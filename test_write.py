import os
import sys
import time
import torch

"""
单机测试
mpirun --allow-run-as-root -np 8 -x CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 test_write.py

多机测试
HOSTFILE=/etc/mpi/hostfile_seq
NP=${NP:-$(cat "$HOSTFILE" | grep -v \# | cut -d'=' -f2 | awk '{sum += $0} END {print sum}')}
mpirun --allow-run-as-root --hostfile "$HOSTFILE" --np $NP -x MASTER_ADDR="$MY_NODE_IP" -x NCCL_IB_DISABLE=1 python3 test_write.py 2>&1 | tee write_bw_${NP}.log
"""

world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
master_port = "6002"

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
    init_method = 'tcp://' + master_addr + ':' + master_port)


speed_write_hdd = []
speed_write_ssd = []
speed_write_local_ssd = []

SSD_DIR = '/path_to_ssd/fast_save/test_write/'
HDD_DIR = '/path_to_hdd/fast_save/test_write/'
LOCAL_SSD_DIR = '/root/ckpt'

def write_to_file(tgt: str, obj: object, fname: str):
    if tgt == 'SSD':
        file_path = os.path.join(SSD_DIR, fname)
    elif tgt == 'HDD':
        file_path = os.path.join(HDD_DIR, fname)
    else:
        file_path = os.path.join(LOCAL_SSD_DIR, fname)

    fd = os.open(file_path, os.O_WRONLY|os.O_CREAT|os.O_SYNC)
    with os.fdopen(fd, "wb") as fo:
        torch.save(obj, fo)

MB = 2 ** 20
run_times = 8

for i in range(run_times):
    # size = 128 * (2**i) # 128 256 512 1024 2048 4096 8192
    size = 512 * (i+1) # 512 1024 1536 2048 2560 3072 3584 4096
    buffer_cpu = torch.rand(size * MB // 2, dtype=torch.bfloat16, pin_memory=True)

    if SSD_DIR:
        if not os.path.exists(SSD_DIR):
            if local_rank == 0:
                os.makedirs(SSD_DIR)
        torch.distributed.barrier()
        t1 = time.time()
        write_to_file('SSD', buffer_cpu, f'rank_{rank}.pt')
        t2 = time.time()
        speed = size * MB / (t2 - t1) / 2 ** 30
        speed_write_ssd.append({f"{size} MB": f"{speed:.4f} GiB/s"})

    if LOCAL_SSD_DIR:
        if not os.path.exists(LOCAL_SSD_DIR):
            if local_rank == 0:
                os.makedirs(LOCAL_SSD_DIR)
        torch.distributed.barrier()
        t1 = time.time()
        write_to_file('LOCAL_SSD', buffer_cpu, f'rank_{rank}.pt')
        t2 = time.time()
        speed = size * MB / (t2 - t1) / 2 ** 30
        speed_write_local_ssd.append({f"{size} MB": f"{speed:.4f} GiB/s"})

    if HDD_DIR:
        if not os.path.exists(HDD_DIR):
            if local_rank == 0:
                os.makedirs(HDD_DIR)
        torch.distributed.barrier()
        t1 = time.time()
        write_to_file('HDD', buffer_cpu, f'rank_{rank}.pt')
        t2 = time.time()
        speed = size * MB / (t2 - t1) / 2 ** 30
        speed_write_hdd.append({f"{size} MB": f"{speed:.4f} GiB/s"})
    
    torch.distributed.barrier()
    if rank == 0:
        print(f"iter {i} finished")

if SSD_DIR:
    torch.distributed.barrier()
    print(f"Rank={rank} write to SSD: {speed_write_ssd}")

if LOCAL_SSD_DIR:
    torch.distributed.barrier()
    print(f"Rank={rank} write to LOCAL_SSD: {speed_write_local_ssd}")

if HDD_DIR:
    torch.distributed.barrier()
    print(f"Rank={rank} write to HDD: {speed_write_hdd}")