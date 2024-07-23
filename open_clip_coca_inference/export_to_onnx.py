import argparse
import onnx
import onnxscript
import open_clip
import torch
from typing import Optional, Tuple
from google.protobuf.json_format import MessageToDict
from onnx import version_converter
from onnx import inliner
from onnxscript import opset18 as op
from onnxscript.onnx_types import INT64
from onnxscript.function_libs.torch_lib.tensor_typing import TReal


parser = argparse.ArgumentParser()
parser.add_argument("--exporter", type=str, default="dynamo", choices=["dynamo", "legacy"], help="What exporter to use.")
parser.add_argument("--backport", action='store_true', help="Backport model to opset 15 (only used for dynamo exporter).")
args = parser.parse_args()


# Export parameters
export_image_encoder = True
export_text_encoder = False
export_text_decoder = False
export_text_generator = True

exporter = args.exporter


# ONNX model files
image_transformer_onnx_file = "open_clip_coca_image_transformer.onnx"
text_encoder_onnx_file = "open_clip_coca_text_encoder.onnx"
text_decoder_onnx_file = "open_clip_coca_text_decoder.onnx"
text_generator_onnx_file = "open_clip_coca_text_generator.onnx"


def print_model_info(onnx_model, title):
    print(f"Model {title}: {onnx_model.doc_string}")
    print(f"IR version: {onnx_model.ir_version}")
    opset_ids = ', '.join(sorted(f"{x.domain or '<default>'}:{x.version}" for x in onnx_model.opset_import))
    print(f"Opsets: {opset_ids}")
    for _input in onnx_model.graph.input:
        print("Input:", MessageToDict(_input))
    for _output in onnx_model.graph.output:
        print("Output:", MessageToDict(_output))
        
        
def backport_model(model, *, target_opset, inline_functions):
    converted_model = model
    if inline_functions:
        converted_model = inliner.inline_local_functions(model, True)
    converted_model = version_converter.convert_version(converted_model, target_opset)

    def keep_default_opset_only(model):
        default_oi = None
        for oi in model.opset_import:
            if oi.domain == "":
                default_oi = oi
                break
        del model.opset_import[:]
        model.opset_import.append(default_oi)
        
    keep_default_opset_only(converted_model)
    return converted_model

custom_ort = onnxscript.values.Opset(domain="com.ish", version=1)

axes = [-1]  # from the model exported by the legacy exporter, will be hardcoded 

@onnxscript.script(custom_ort)
def custom_layer_norm(
    input: TReal,
    normalized_shape: INT64,
    weight: Optional[TReal] = None,
    bias: Optional[TReal] = None,
    eps: float = 1e-05,
) -> Tuple[TReal, TReal, TReal]:
    x = input
    mean = op.ReduceMean(x, axes=axes)
    d = op.Sub(x, mean)
    
    dd = op.Pow(d, 2)
    var = op.ReduceMean(dd, axes=axes)
    var_eps = op.Add(var, eps)
    std_dev = op.Sqrt(var_eps)
    
    normalized = op.Div(d, std_dev)
    normalized_scaled = op.Mul(normalized, weight)
    y = op.Add(normalized_scaled, bias)

    return y, mean, std_dev

onnx_registry = torch.onnx.OnnxRegistry()
if args.backport:
    onnx_registry.register_op(
        namespace="aten", op_name="native_layer_norm", overload="default", function=custom_layer_norm)


# Load model
model, _, _ = open_clip.create_model_and_transforms(
  model_name="coca_ViT-L-14",
  pretrained="mscoco_finetuned_laion2B-s13B-b90k"
)
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active


# Export image encoder
if export_image_encoder:
    print("Export image transformer --------------")
    image_encoder = model.visual

    image_input_shape = [2, 3] + list(image_encoder.image_size)  # batch, channels==3, width, height
    print(f"Image input shape: {image_input_shape}")

    model_inputs = {
        "l_x_": torch.randn(*image_input_shape),  # name matches Dynamo's auto-generated name 
    }
    output_names = ["logits"]
    dynamic_axes = {
        "l_x_": {0: "batch_size"},
    }

    if exporter == "legacy":
        torch.onnx.export(
            image_encoder,
            tuple(model_inputs.values()),
            f=image_transformer_onnx_file,
            export_params=True,
            verbose=False,
            opset_version=15,
            do_constant_folding=True,
            input_names=list(model_inputs.keys()),
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
    else:
        diagnostic_options = torch.onnx.DiagnosticOptions(verbosity_level=20, warnings_as_errors=False)
        export_options = torch.onnx.ExportOptions(onnx_registry=onnx_registry, dynamic_shapes=True, diagnostic_options=diagnostic_options)
        onnx_program = torch.onnx.dynamo_export(image_encoder, *model_inputs.values(), export_options=export_options)
        onnx_program.save(image_transformer_onnx_file)        
        print("Model is exported successfully")

        if args.backport:
            backported = backport_model(onnx_program.model_proto, inline_functions=True, target_opset=15)
            onnx.save(backported, image_transformer_onnx_file)
            print("Model is backported successfully")

    onnx_image_transformer = onnx.load_model(image_transformer_onnx_file)
    onnx.checker.check_model(onnx_image_transformer, full_check=True)
    print_model_info(onnx_image_transformer, image_transformer_onnx_file)


# Export text encoder
if export_text_encoder:
    print("Export text encoder --------------")

    # model.text is TextTransformer, it contains a transformer of type Transformer
    text_encoder = model.text.transformer

    cast_dtype = text_encoder.get_cast_dtype()
    text_input_shape = (model.text.context_length + 1, 3, model.text.token_embedding.embedding_dim)

    attn_mask_shape = model.text.attn_mask.shape

    print(f"Text input shape: {text_input_shape}")
    print(f"Attention mask shape: {attn_mask_shape}")

    model_inputs = {
        "tokens": torch.randn(*text_input_shape, dtype=cast_dtype),
        "attention_mask": torch.randn(*attn_mask_shape)
    }
    output_names = ["logits"]
    dynamic_axes = {
        'tokens': {0: 'context_length', 1: 'batch_size'},
        'attention_mask': {0: 'context_length', 1: 'context_length'},
        'logits': {0: 'context_length', 1: 'batch_size'},
    }

    if exporter == "legacy":
        torch.onnx.export(
            text_encoder,
            tuple(model_inputs.values()),
            f=text_encoder_onnx_file,
            export_params=True,
            verbose=True,
            opset_version=15,
            do_constant_folding=True,
            input_names=list(model_inputs.keys()),
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
    else:
        diagnostic_options = torch.onnx.DiagnosticOptions(verbosity_level=20, warnings_as_errors=False)
        export_options = torch.onnx.ExportOptions(dynamic_shapes=True, diagnostic_options=diagnostic_options)
        onnx_program = torch.onnx.dynamo_export(text_encoder, *model_inputs.values(), export_options=export_options)
        onnx_program.save(text_encoder_onnx_file)

    onnx_text_encoder = onnx.load(text_encoder_onnx_file)
    onnx.checker.check_model(onnx_text_encoder, full_check=True)
    print_model_info(onnx_text_encoder, text_encoder_onnx_file)


# Export text decoder transformer
if export_text_decoder:
    print("Export text decoder --------------")
    text_decoder = model.text_decoder

    token_embedding_input_shape = (3, 5, 768)  # 3 and 5 are random values to trick dynamic shape detection
    print(f"Text input shape: {token_embedding_input_shape}")

    model_inputs = {
        "image_embs": torch.randn(3, 255, 768, dtype=torch.float32),
        "token_embs": torch.randn(*token_embedding_input_shape, dtype=torch.float32)
    }
    output_names = ["logits"]
    dynamic_axes = {
        'image_embs': {0: 'batch_size'},
        'token_embs': {0: 'batch_size', 1: 'context_len'},
        'logits': {0: 'batch_size', 1: 'context_len'},
    }

    if exporter == "legacy":
        torch.onnx.export(
            text_decoder,
            tuple(model_inputs.values()),
            f=text_decoder_onnx_file,
            export_params=True,
            verbose=True,
            opset_version=15,
            do_constant_folding=False,
            export_modules_as_functions=False,
            input_names=list(model_inputs.keys()),
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
    else:
        diagnostic_options = torch.onnx.DiagnosticOptions(verbosity_level=20, warnings_as_errors=False)
        export_options = torch.onnx.ExportOptions(dynamic_shapes=True, diagnostic_options=diagnostic_options)
        onnx_program = torch.onnx.dynamo_export(text_decoder, *model_inputs.values(), export_options=export_options)
        onnx_program.save(text_decoder_onnx_file)

    onnx_text_decoder = onnx.load(text_decoder_onnx_file)
    print_model_info(onnx_text_decoder, text_decoder_onnx_file)


# Export generator (encoder + decoder) -----------
if export_text_generator:
    print("Export text generator --------------")
    class TextGeneratorModel(torch.nn.Sequential):
        def __init__(self):
            super().__init__()
            self.encoder = model.text.transformer
            self.decoder = model.text_decoder
        
        def forward(self, tokens, attn_mask, image_embs):  # this function replicates CoCa model forward()
            tokens = tokens.permute(1, 0, 2)  # NLD -> LND
            token_embs = self.encoder(tokens, attn_mask=attn_mask)
            token_embs = token_embs.permute(1, 0, 2)  # LND -> NLD (cause MultimodalTransformer expects NLD)

            token_embs = token_embs[:, :-1]  # _, tokens = text_global_pool(x, pool_type='last')
            
            return self.decoder.forward(image_embs, token_embs)  # MultimodalTransformer    
            
    tgm = TextGeneratorModel()
        
    text_encoder = model.text.transformer

    cast_dtype = text_encoder.get_cast_dtype()
    
    if exporter == "legacy":
        text_input_shape = (3, model.text.context_length + 1, model.text.token_embedding.embedding_dim)
        attn_mask_shape = model.text.attn_mask.shape
    else:
        text_input_shape = (3, 5, model.text.token_embedding.embedding_dim)  
        attn_mask_shape = (5, 5)  # numbers are chosen to trick dynamic shapes detector in dynamo exporter

    print(f"Text input shape: {text_input_shape}")
    print(f"Attention mask shape: {attn_mask_shape}")

    model_inputs = {
        "tokens": torch.randn(*text_input_shape, dtype=cast_dtype),
        "attention_mask": torch.randn(*attn_mask_shape),
        "image_embs": torch.randn(3, 255, 768, dtype=torch.float32),
    }
    output_names = ["logits"]
    dynamic_axes = {
        'tokens': {0: 'batch_size', 1: 'context_length'},
        'attention_mask': {0: 'context_length', 1: 'context_length'},
        'image_embs': {0: 'batch_size'},
    }

    if exporter == "legacy":
        torch.onnx.export(
            tgm,
            tuple(model_inputs.values()),
            f=text_generator_onnx_file,
            export_params=True,
            verbose=False,
            opset_version=15,
            do_constant_folding=False,
            export_modules_as_functions=False,
            input_names=list(model_inputs.keys()),
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
    else:
        diagnostic_options = torch.onnx.DiagnosticOptions(verbosity_level=20, warnings_as_errors=False)
        export_options = torch.onnx.ExportOptions(onnx_registry=onnx_registry, dynamic_shapes=True, diagnostic_options=diagnostic_options)
        onnx_program = torch.onnx.dynamo_export(tgm, *model_inputs.values(), export_options=export_options)
        onnx_program.save(text_generator_onnx_file)
        print("Model is exported successfully")

        if args.backport:
            backported = backport_model(onnx_program.model_proto, inline_functions=True, target_opset=15)
            onnx.save(backported, text_generator_onnx_file)
            print("Model is backported successfully")

    onnx_text_generator = onnx.load(text_generator_onnx_file)
    onnx.checker.check_model(onnx_text_generator, full_check=True)
    print_model_info(onnx_text_generator, text_generator_onnx_file)
