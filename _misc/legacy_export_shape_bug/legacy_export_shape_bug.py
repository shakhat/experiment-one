# Legacy ONNX export inserts constant for shapes of tensors returned by 'unflatten' function

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
from google.protobuf.json_format import MessageToDict


class BuggyModel(nn.Module):
    def forward(self, query: torch.Tensor):
        n = query.shape[2]  # here ONNX will contain Shape operator
        p = query.unflatten(-1, (2, n // 2))
        d = p.shape[0]  # here ONNX will contain a Constant node instead of a Shape operator 
        k = p.view(d, 4, 2)  # actually Reshape operator
        return k
   
# Instantiate the model
model = BuggyModel()

# Export the model to ONNX 
tensor_x = torch.rand((3, 1, 8))
model_inputs = {
    "l_query_": tensor_x,
}
output_names = ["y"]
dynamic_axes = {
    'l_query_': {0: 'context_length', 1: 'batch_size'},
}

torch.onnx.export(
    model,
    tuple(model_inputs.values()),
    f="buggy_model.onnx",
    export_params=True,
    verbose=True,
    opset_version=15,
    do_constant_folding=False,
    input_names=list(model_inputs.keys()),
    output_names=output_names,
    dynamic_axes=dynamic_axes,
)

# Check the model is exported with correct dynamic shape
loaded_model = onnx.load("buggy_model.onnx")
for _input in loaded_model.graph.input:
    print("Input:", MessageToDict(_input))


# Calculate the model natively
x = torch.rand((3, 1, 8))
etalon = model(x)

# Calculate the model with ONNX
s = ort.InferenceSession("buggy_model.onnx", providers=['CPUExecutionProvider'])
r = torch.tensor(s.run(None, {"l_query_": x.numpy()})[0])

# Check results are the same
print(f"ONNX model result {'equals' if torch.all(r == etalon).item() else 'is not equal'} to pytorch model")

# Calculate the model with a different shape
# this will fail despite that the first dimension is dynamic 
# RUNTIME_EXCEPTION : Non-zero status code returned while running Reshape node
# The input tensor cannot be reshaped to the requested shape. Input shape:{4,1,2,4}, requested shape:{3,4,2}
x = torch.rand((4, 1, 8))
r = torch.tensor(s.run(None, {"l_query_": x.numpy()})[0])
