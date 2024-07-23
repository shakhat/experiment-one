import torch
import torch_npu

# this works as expected
dev = "cpu"
r = torch.tensor([1], dtype=torch.int64, device=dev)
s = torch.tensor([[0]*252], dtype=torch.int64, device=dev)
print(dev, s.dtype, r[s])


# this also works (int32)
dev = "npu"
r = torch.tensor([1], dtype=torch.int64, device=dev)
s = torch.tensor([[0]*252], dtype=torch.int32, device=dev)
print(dev, s.dtype, r[s])


# but this crashes with "E40021: Failed to compile Op [Index1]" 
# reproducible on Kunpeng + Ascend 910 ProB + ascend-toolkit/7.0.0/aarch64-linux
# (though works on x86 + Ascend 310 + ascend-toolkit/7.0.1.1/x86_64-linux)
dev = "npu"
r = torch.tensor([1], dtype=torch.int64, device=dev)
s = torch.tensor([[0]*252], dtype=torch.int64, device=dev)
print(dev, s.dtype, r[s])
