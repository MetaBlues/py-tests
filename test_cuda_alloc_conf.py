# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python test_mem_flag.py

import torch
import os
from contextlib import contextmanager

@contextmanager
def fixed_cuda_allocator():
    # 保存原来的 allocator 设置
    original_settings = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    print(original_settings)
    
    try:
        # 设置 allocator 为 expandable_segments:False
        torch.cuda.memory._set_allocator_settings('expandable_segments:False')
        yield
    finally:
        # 恢复到原来的 allocator 设置
        torch.cuda.memory._set_allocator_settings(original_settings)

def create_shared_tensor(size):
    # 创建一个共享的 CUDA tensor
    tensor = torch.randn(size, device='cuda')
    shared_tensor = tensor.share_memory_()  # 将其放入共享内存
    return shared_tensor

if __name__ == "__main__":
    # 使用上下文管理器创建共享 Tensor
    with fixed_cuda_allocator():
        shared_tensor = create_shared_tensor((3, 3))
        print("Created shared tensor in fixed allocator context:")
        print(shared_tensor)

    # 在此处，你可以将 shared_tensor 传递给其他进程进行 IPC
    # 例如，使用 multiprocessing.Queue、Pipe 或者其他 IPC 方法。
    
    # 以下为示例打印共享 tensor 的值
    print("Shared tensor after exiting context:")
    print(shared_tensor)
