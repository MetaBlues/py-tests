import os
import time
import torch

"""
nsys profile -t cuda,nvtx -s none --cpuctxsw none \
mpirun --allow-run-as-root -np 8 -x CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 test_memcpy.py
"""

world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
rank = int(os.environ["OMPI_COMM_WORLD_RANK"])

torch.cuda.set_device(rank % torch.cuda.device_count())


def set_ideal_affinity_for_current_gpu():
    import cuda.cuda
    import cuda.cudart
    import pynvml
    import uuid
    err, device_id = cuda.cudart.cudaGetDevice()
    assert err == cuda.cudart.cudaError_t.cudaSuccess
    err, device_uuid = cuda.cuda.cuDeviceGetUuid(device_id)
    assert err == cuda.cuda.CUresult.CUDA_SUCCESS
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByUUID("GPU-" + str(uuid.UUID(bytes=device_uuid.bytes)))
    pynvml.nvmlDeviceSetCpuAffinity(handle)

set_ideal_affinity_for_current_gpu()

torch.distributed.init_process_group(
    backend="nccl",
    world_size=world_size, rank=rank,
    init_method=f"tcp://127.0.0.1:6002")

size = 128 * (2 ** 20)
buffer_cpu = torch.empty(size, dtype=torch.uint8, pin_memory=True)
buffer_gpu = torch.empty(size, dtype=torch.uint8, device="cuda")

buffer_cpu.copy_(torch.rand(size // 2, dtype=torch.bfloat16).view(torch.uint8))
buffer_gpu.copy_(torch.rand(size // 2, dtype=torch.bfloat16, device="cuda").view(torch.uint8))

run_times = 10
speed_DtoH_list = []
speed_HtoD_list = []

for _ in range(20):
    torch.distributed.barrier()
    t1 = time.time()
    for _ in range(run_times):
        buffer_cpu.copy_(buffer_gpu, non_blocking=True)
    torch.distributed.barrier()
    t2 = time.time()
    speed = size * run_times / (t2 - t1)
    if rank == 0:
        print(f"memcpy DtoH {speed / 2 ** 30:.1f} GiB/s")
    speed_DtoH_list.append(speed)

    torch.distributed.barrier()
    t1 = time.time()
    for _ in range(run_times):
        buffer_gpu.copy_(buffer_cpu, non_blocking=True)
    torch.distributed.barrier()
    t2 = time.time()
    speed = size * run_times / (t2 - t1)
    if rank == 0:
        print(f"memcpy HtoD {speed / 2 ** 30:.1f} GiB/s")
    speed_HtoD_list.append(speed)

if rank == 0:
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')} memcpy DtoH max {max(speed_DtoH_list) / 2 ** 30:.1f} GiB/s")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')} memcpy HtoD max {max(speed_HtoD_list) / 2 ** 30:.1f} GiB/s")