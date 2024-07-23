import argparse
import os
import numpy as np
import torch
from PIL import Image
import open_clip
import time
import onnxruntime as ort

from torch import nn
from open_clip.transformer import text_global_pool


def _change_file_extension(file_name, new_extension):
    s = os.path.splitext(file_name)
    return s[0] + new_extension


def encode_text(model, runtime, text_transformer, text):  # this code is copied from open_clip.model
    with torch.no_grad():
        cast_dtype = model.transformer.get_cast_dtype()

        x = model.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        x = x + model.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        if runtime == "onnx":
            model_inputs = text_transformer.get_inputs()
            outputs = text_transformer.run(None, {model_inputs[0].name: x.numpy(), model_inputs[1].name: model.attn_mask.numpy()})
            x = torch.tensor(outputs[0])
        elif runtime == "acl":
            text_data = x.numpy()
            attn_mask_data = model.attn_mask.numpy()
            inputs = (text_data, attn_mask_data) if text_transformer.input_info[0].name == "tokens" else (attn_mask_data, text_data)
            output = text_transformer.run(*inputs)
            x = torch.tensor(output)
        else:
            x = model.transformer(x, attn_mask=model.attn_mask)
        
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = model.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        x, _ = text_global_pool(x, text, model.text_pool_type)
        if model.text_projection is not None:
            if isinstance(model.text_projection, nn.Linear):
                x = model.text_projection(x)
            else:
                x = x @ model.text_projection

        return x


def run(runtime, device, img_name, text, verbose, profiling):
    if device.startswith("npu"):
        try:
            import torch_npu  # must go after cv2 otherwise "ImportError: /lib/aarch64-linux-gnu/libGLdispatch.so.0: cannot allocate memory in static TLS block"
        except ImportError:
            raise RuntimeError("'torch-npu' library is required for Ascend NPU devices")
    if runtime == "acl":
        try:
            from acl_runtime import AclRuntime
        except ImportError:
            raise RuntimeError("'acl' library is required for ACL native runtime")
    print(f"Using device: {device}")

    pt_device = torch.device(device)
    device_index = pt_device.index or 0
    if device.startswith("npu"):
        providers = [
            ("CANNExecutionProvider", {"device_id": device_index, "dump_om_model": True},),
        ]
    elif device.startswith("cuda"):
        providers = [
            ("CUDAExecutionProvider", {"device_id": device_index,},),
        ]
    else:
        providers = ["CPUExecutionProvider"]
        
    options = ort.SessionOptions()
    options.enable_profiling = profiling
    if verbose:
        options.log_severity_level = 0
        options.log_verbosity_level = 0
    
    # Encode image
    image_transformer = None
    if runtime == "onnx":    
        image_transformer = ort.InferenceSession(image_transformer_onnx_file, sess_options=options, providers=providers)
    elif runtime == "acl":
        image_transformer = AclRuntime(model_path=_change_file_extension(image_transformer_onnx_file, ".om"), device_id=device_index)
        print(f"ACL model is loaded: {image_transformer}")
    else:
        model.to(pt_device)
    
    image = preprocess(Image.open(img_name)).unsqueeze(0)

    start = time.perf_counter_ns()
    if runtime == "onnx":
        img_data = image.numpy().astype(np.float32)
        model_inputs = image_transformer.get_inputs()
        outputs = image_transformer.run(None, {model_inputs[0].name: img_data})
        image_features = torch.tensor(outputs[0])
    elif runtime == "acl":
        img_data = image.numpy().astype(np.float32)
        outputs = image_transformer.run(img_data)
        image_features = torch.tensor(outputs)
    else:
        image_features = model.encode_image(image.to(pt_device))
        
    print("Image features:", image_features.shape, image_features.dtype)
    torch.set_printoptions(threshold=50)
    print(image_features)
    print(f"Image encoding time: {(time.perf_counter_ns() - start) / 10**6} ms")

    # Encode text
    text_transformer = None
    if runtime == "onnx":
        text_transformer = ort.InferenceSession(text_transformer_onnx_file, sess_options=options, providers=providers)
    elif runtime == "acl":
        text_transformer = AclRuntime(model_path=_change_file_extension(text_transformer_onnx_file, ".om"), device_id=device_index)
        print(f"ACL model is loaded: {text_transformer}")

    tokens = tokenizer(text.split(","))
    if runtime == "pytorch":
        tokens = tokens.to(pt_device)
    
    start = time.perf_counter_ns()
    text_features = encode_text(model, runtime, text_transformer, tokens)
    print("Text features:", text_features.shape, text_features.dtype)
    print(text_features)
    print(f"Text encoding time: {(time.perf_counter_ns() - start) / 10**6} ms")

    # Correlation        
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
    print("Label probs:", text_probs)


if __name__ == "__main__":
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True, help="Path to input image.")
    parser.add_argument("--text", type=str, required=True, help="Comma-separated strings.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for tensor operations.")
    parser.add_argument("--runtime", type=str, default="onnx", choices=["onnx", "pytorch", "acl"], help="What runtime to use: ONNX, pytorch or native ACL.")
    parser.add_argument("--verbose", action='store_true', help="Enable verbose logging.")
    parser.add_argument("--profiling", action='store_true', help="Enable ONNX profiling")
    args = parser.parse_args()

    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    
    # ONNX model files
    image_transformer_onnx_file = "open_clip_image_transformer.onnx"
    text_transformer_onnx_file = "open_clip_text_transformer.onnx"

    run(args.runtime, args.device, args.img, args.text, args.verbose, args.profiling)
