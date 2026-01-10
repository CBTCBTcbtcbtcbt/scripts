import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
print(f"CUDA 版本: {torch.version.cuda}")
print(f"GPU 设备: {torch.cuda.get_device_name(0)}")
def test(a,b,c=None):
    if c:
        print("c is not None")
    else:
        print("c is None")
    pass
if_c=False
test(1,2, 3 if if_c else None)