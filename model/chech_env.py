import torch
import torchvision
import sys

print("=" * 50)
print("环境检查结果")
print("=" * 50)

# PyTorch信息
print(f"PyTorch版本: {torch.__version__}")
print(f"Torchvision版本: {torchvision.__version__}")
print(f"Python版本: {sys.version}")

# CUDA信息
print(f"CUDA是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU设备数量: {torch.cuda.device_count()}")
    print(f"当前GPU设备: {torch.cuda.current_device()}")
    print(f"GPU设备名称: {torch.cuda.get_device_name()}")
else:
    print("CUDA不可用，将使用CPU进行训练")

# 设备信息
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"将使用的设备: {device}")

print("=" * 50)
