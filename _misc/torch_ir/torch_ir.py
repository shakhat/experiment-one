# Finding out what ATEN operation corresponds to torch function may not be trivial
# To do this we compile our torch code to ATEN IR and see that LayerNorm model
# actually calls aten::native_layer_norm function
# Refs: 
#   https://github.com/pytorch/pytorch/blob/v2.2.0/torch/fx/README.md
#   Colab https://colab.research.google.com/drive/1Zh-Uo3TcTH8yYJF-LLo5rjlHVMtqvMdf#scrollTo=9onie0auHyfD

import torch
import torch.nn as nn
from torch._decomp import core_aten_decompositions
from torch._functorch.aot_autograd import aot_module_simplified


# Sample model
class CustomModel(torch.nn.Module):
    def __init__(self, *embedding_dim):
        super().__init__()
        self.m_norm = nn.LayerNorm(embedding_dim)

    def forward(self, input_x):
        y = input_x
        y = self.m_norm(y)
        return y
    

decompositions = core_aten_decompositions()
decompositions.update(
    torch._decomp.get_decompositions([
    ])
)

def printing_backend(gm, sample_inputs):
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

model = CustomModel(1, 2)
print(model)

input_x = torch.randn(4, 1, 2)

fn = torch.compile(backend=printing_backend, dynamic=True)(model)
fn(input_x)
