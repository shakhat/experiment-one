import argparse
import onnx
import onnxscript
import open_clip
import torch
from google.protobuf.json_format import MessageToDict
from onnx import version_converter
from onnx import inliner
from onnxscript import opset18 as op
from onnxscript.onnx_types import INT64
from onnxscript.function_libs.torch_lib.tensor_typing import TReal
from typing import Optional, Tuple


parser = argparse.ArgumentParser()
parser.add_argument("--exporter", type=str, default="dynamo", choices=["dynamo", "legacy"], help="What exporter to use.")
parser.add_argument("--backport", action='store_true', help="Backport model to opset 15 (only used for dynamo exporter).")
args = parser.parse_args()

# Export parameters
exporter = args.exporter

# ONNX model files
image_transformer_onnx_file = "open_clip_image_transformer.onnx"
text_transformer_onnx_file = "open_clip_text_transformer.onnx"

# Helpers
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
        
    if inline_functions:  # if we don't inline there will be an additional opset import for functions
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


# Pytorch model
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# Export image transformer
print("Exporting image transformer")
image_transformer = model.visual

image_input_shape = [4, 3] + list(image_transformer.image_size)
print(f"Image input shape: {image_input_shape}")

model_inputs = {
    "image": torch.randn(*image_input_shape),
}
output_names = ["logits"]
dynamic_axes = {
    'image': {0: 'batch_size'},
}

if exporter == "legacy":
    torch.onnx.export(
        image_transformer,
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
    onnx_program = torch.onnx.dynamo_export(image_transformer, *model_inputs.values(), export_options=export_options)
    onnx_program.save(image_transformer_onnx_file)        
    print("Model is exported successfully")

    if args.backport:
        backported = backport_model(onnx_program.model_proto, inline_functions=True, target_opset=15)
        onnx.save(backported, image_transformer_onnx_file)
        print("Model is backported successfully")
        
onnx_model = onnx.load_model(image_transformer_onnx_file)
onnx.checker.check_model(onnx_model, full_check=True)
print_model_info(onnx_model, image_transformer_onnx_file)        


# Export text transformer
print("Exporting text transformer")
text_transformer = model.transformer

cast_dtype = model.transformer.get_cast_dtype()
text_input_shape = (model.context_length, 3, model.token_embedding.embedding_dim)
print(f"Text input shape: {text_input_shape}")

model_inputs = {
    "tokens": torch.randn(*text_input_shape, dtype=cast_dtype),
    "attention_mask": torch.randn(*model.attn_mask.shape)
}
output_names = ["logits"]
dynamic_axes = {
    'tokens': {0: 'context_length', 1: 'batch_size'},
    'attention_mask': {0: 'context_length', 1: 'context_length'},
    'logits': {0: 'context_length', 1: 'batch_size'},
}

if exporter == "legacy":
    torch.onnx.export(
        text_transformer,
        tuple(model_inputs.values()),
        f=text_transformer_onnx_file,
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
    onnx_program = torch.onnx.dynamo_export(text_transformer, *model_inputs.values(), export_options=export_options)
    onnx_program.save(text_transformer_onnx_file)        
    print("Model is exported successfully")

    if args.backport:
        backported = backport_model(onnx_program.model_proto, inline_functions=True, target_opset=15)
        onnx.save(backported, text_transformer_onnx_file)
        print("Model is backported successfully")

onnx_model = onnx.load_model(text_transformer_onnx_file)
onnx.checker.check_model(onnx_model, full_check=True)
print_model_info(onnx_model, text_transformer_onnx_file)        
