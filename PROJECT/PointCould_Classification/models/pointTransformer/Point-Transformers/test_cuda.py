import torch

print(torch.__version__)
if not torch.cuda.is_available():
    print("CUDA NOT AVAILABLE")
else:
    print("Cuda available")
