# This script reimplements layer normalization operator in Opset 15

from typing import Optional, Tuple, TypeVar, Union
import torch
import torch._dynamo
import torch.nn as nn
import onnx
import onnxruntime as ort
from onnx import version_converter
from onnx import inliner
import onnxscript
from onnxscript import opset18 as op
from onnxscript.onnx_types import INT64, BFLOAT16, DOUBLE, FLOAT, FLOAT16
from onnxscript.function_libs.torch_lib.tensor_typing import TReal
from typing_extensions import TypeAlias


def backport_model(model):
    converted_model = inliner.inline_local_functions(model, True)
    converted_model = version_converter.convert_version(converted_model, 15)

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

# we provide a custom implementation of aten::native_layer_norm function
# code is based on https://pytorch.org/tutorials/beginner/onnx/onnx_registry_tutorial.html
custom_ort = onnxscript.values.Opset(domain="com.ish", version=1)

# NOTE: The function signature must match the signature of the unsupported ATen operator.
# https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml
# NOTE: All attributes must be annotated with type hints.

T_LayerNormalization = TypeVar("T_LayerNormalization", BFLOAT16, DOUBLE, FLOAT, FLOAT16)
U_LayerNormalization: TypeAlias = Union[BFLOAT16, FLOAT]

# In Opset 17 layer normalization is implemented as ONNX LayerNormalization operator
# We first translate it to older operators (e.g. by observing a graph compiled by 
# a legacy exporter and then verifying by function declaration)
# However, ReduceMean from Opset 13 doesn't accept axis as an input, so we have
# to check hardcode value to match what is used by LayerNormalization

axes = [-2, -1]  # from the model exported by legacy exporter

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
    

# Register our custom operator 
onnx_registry = torch.onnx.OnnxRegistry()
onnx_registry.register_op(
    namespace="aten", op_name="native_layer_norm", overload="default", function=custom_layer_norm)
export_options = torch.onnx.ExportOptions(onnx_registry=onnx_registry, op_level_debug=True, dynamic_shapes=True)

# Sample model
class CustomModel(torch.nn.Module):
    def __init__(self, *embedding_dim):
        super().__init__()
        self.m_norm = nn.LayerNorm(embedding_dim)

    def forward(self, input_x):
        y = input_x
        y = self.m_norm(y)
        return y
    
dims = (1, 2)
cm = CustomModel(*dims)

input_x = torch.randn(4, 1, 2)

# Finding out what ATEN operation corresponds to torch function may not be trivial
# To do this we compile our torch code to ATEN IR and see that LayerNorm model
# actually calls aten::native_layer_norm function
# Refs: 
#   https://github.com/pytorch/pytorch/blob/v2.2.0/torch/fx/README.md
#   Colab https://colab.research.google.com/drive/1Zh-Uo3TcTH8yYJF-LLo5rjlHVMtqvMdf#scrollTo=9onie0auHyfD


from torch._decomp import core_aten_decompositions
# Backends can further finetune the decompositions if needed
# Available decompositions can be found in
# torch/_decomp/decompositions.py and torch/_refs/__init__.py
decompositions = core_aten_decompositions()
decompositions.update(
    torch._decomp.get_decompositions([
    ])
)

from torch._functorch.aot_autograd import aot_module_simplified

def toy_backend(gm, sample_inputs):
    def my_compiler(gm, sample_inputs):
        print("Decomposed fx Graph in Aten IR:")
        gm.print_readable()
        return gm

    return aot_module_simplified(
        gm,
        sample_inputs,
        decompositions=decompositions,
        fw_compiler=my_compiler
    )

torch._dynamo.reset()
fn = torch.compile(backend=toy_backend, dynamic=True)(cm)
# triggers compilation of forward graph on the first run
out = fn(input_x)

# Now export to ONNX
# Dynamo produces model in Opset 18
onnx_program = torch.onnx.dynamo_export(cm, input_x, export_options=export_options)
print(onnx_program.model_proto)
onnx_program.save("model.onnx")

# Model exported by legacy exporter is in Opset 15
# We can use it to verify that our implementation of LayerNormalization is correct
model_inputs = {
    "y": input_x,
}
output_names = ["y"]

torch.onnx.export(
    cm,
    tuple(model_inputs.values()),
    f="model-legacy.onnx",
    export_params=True,
    # verbose=True,
    opset_version=15,
    do_constant_folding=False,
    export_modules_as_functions=True,
    input_names=list(model_inputs.keys()),
    output_names=output_names,
)

# Backport the model to Opset 15
backported = backport_model(onnx_program.model_proto)
print(backported)
onnx.save(backported, "model.onnx")

# Check ONNX model
saved_onnx_model = onnx.load_model("model.onnx")
onnx.checker.check_model(saved_onnx_model, full_check=True)
print("Model check passed")

# Calculate the model natively
etalon = cm(input_x)

# Calculate the model with ONNX
s = ort.InferenceSession("model.onnx", providers=['CPUExecutionProvider'])
r = torch.tensor(s.run(None, {"y": input_x.numpy()})[0])

# Check that the results are the same
print('Pytorch result:')
print(etalon)
print('ONNX result:')
print(r)
print(f"ONNX model result is {'' if torch.allclose(etalon, r, atol=1e-07) else 'not '}equal to pytorch model")
