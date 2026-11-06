import os
import sys
import platform
import psutil
import torch

def get_env_info():
    info = []
    info.append("# Local Environment Information (ENV_LOCAL.md)\n")
    info.append(f"- **OS**: {platform.system()} {platform.release()} ({platform.version()})")
    info.append(f"- **Machine**: {platform.machine()}")
    info.append(f"- **Python Version**: {sys.version.split()[0]}")
    info.append(f"- **Python Executable**: {sys.executable}")
    info.append(f"- **PyTorch Version**: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    info.append(f"- **CUDA Available**: {cuda_available}")
    if cuda_available:
        info.append(f"- **CUDA Version**: {torch.version.cuda}")
        info.append(f"- **GPU Count**: {torch.cuda.device_count()}")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_bytes = torch.cuda.get_device_properties(0).total_memory
        gpu_mem_gb = gpu_mem_bytes / (1024 ** 3)
        info.append(f"- **GPU 0 Name**: {gpu_name}")
        info.append(f"- **GPU 0 Total Memory**: {gpu_mem_gb:.2f} GB")
        info.append(f"- **BF16 Supported**: {torch.cuda.is_bf16_supported()}")
    else:
        info.append("- **GPU**: None (CPU Mode)")
        
    ram_bytes = psutil.virtual_memory().total
    ram_gb = ram_bytes / (1024 ** 3)
    info.append(f"- **System RAM**: {ram_gb:.2f} GB\n")

    return "\n".join(info)

if __name__ == "__main__":
    env_info_md = get_env_info()
    print(env_info_md)
    
    docs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
    os.makedirs(docs_dir, exist_ok=True)
    out_path = os.path.join(docs_dir, "ENV_LOCAL.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(env_info_md)
    print(f"Saved local environment info to: {out_path}")
