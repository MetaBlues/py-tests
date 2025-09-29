import torch
import time

torch.cuda.init()
assert torch.cuda.device_count() >= 3, "need at least three GPUs"

N = 2 * 1024 * 1024 * 1024
dtype = torch.float32

# Host Tensor (pinned for async)
host_h2d_tensor = torch.randn(N, dtype=dtype, pin_memory=True)
host_d2h_tensor = torch.empty(N, dtype=dtype, pin_memory=True)

# Device 0 tensor
p2p_send_tensor0 = torch.randn(N, dtype=dtype, device="cuda:0")

# Device 1 tensors
p2p_recv_tensor1 = torch.empty_like(p2p_send_tensor0, device="cuda:1")
p2p_send_tensor1 = torch.randn(N, dtype=dtype, device="cuda:1")
device_h2d_tensor1 = torch.empty(N, dtype=dtype, device="cuda:1")
device_d2h_tensor1 = torch.randn(N, dtype=dtype, device="cuda:1")
d2d_from_tensor1 = torch.randn(N, dtype=dtype, device="cuda:1")
d2d_to_tensor1 = torch.empty_like(d2d_from_tensor1, device="cuda:1")

# Device 2 tensor
p2p_recv_tensor2 = torch.empty_like(p2p_send_tensor1, device="cuda:2")

stream_h2d = torch.cuda.Stream(device=1)
stream_d2h = torch.cuda.Stream(device=1)
stream_p2p_send = torch.cuda.Stream(device=1)
stream_p2p_recv = torch.cuda.Stream(device=1)
stream_d2d = torch.cuda.Stream(device=1)
torch.cuda.synchronize()

for i in range(3):
    # Only H2D
    start = time.time()
    with torch.cuda.stream(stream_h2d):
        device_h2d_tensor1.copy_(host_h2d_tensor, non_blocking=False)
    torch.cuda.synchronize()
    t_h2d = time.time() - start
    print(f"only H2D cost: {t_h2d:.3f} seconds")

    # Only D2H
    start = time.time()
    with torch.cuda.stream(stream_d2h):
        host_d2h_tensor.copy_(device_d2h_tensor1, non_blocking=False)
    torch.cuda.synchronize()
    t_d2h = time.time() - start
    print(f"only D2H cost: {t_d2h:.3f} seconds")

    # Only P2P send
    start = time.time()
    with torch.cuda.stream(stream_p2p_send):
        p2p_recv_tensor2.copy_(p2p_send_tensor1, non_blocking=False)
    torch.cuda.synchronize()
    t_p2p_send = time.time() - start
    print(f"only P2P send cost: {t_p2p_send:.3f} seconds")

    # Only P2P recv
    start = time.time()
    with torch.cuda.stream(stream_p2p_recv):
        p2p_recv_tensor1.copy_(p2p_send_tensor0, non_blocking=False)
    torch.cuda.synchronize()
    t_p2p_recv = time.time() - start
    print(f"only P2P recv cost: {t_p2p_recv:.3f} seconds")

    # Only D2D
    start = time.time()
    with torch.cuda.stream(stream_d2d):
        d2d_to_tensor1.copy_(d2d_from_tensor1, non_blocking=False)
    torch.cuda.synchronize()
    t_d2d = time.time() - start
    print(f"only D2D cost: {t_d2d:.3f} seconds")

    # overlap H2D + P2P send
    start = time.time()
    with torch.cuda.stream(stream_h2d):
        device_h2d_tensor1.copy_(host_h2d_tensor, non_blocking=False)     # Host → GPU1
    with torch.cuda.stream(stream_p2p_send):
        p2p_recv_tensor2.copy_(p2p_send_tensor1, non_blocking=False)     # GPU0 → GPU1
    torch.cuda.synchronize()
    t_both = time.time() - start
    print(f"overlap H2D + P2P cost: {t_both:.3f} seconds")

    # overlap H2D + P2P recv
    start = time.time()
    with torch.cuda.stream(stream_h2d):
        device_h2d_tensor1.copy_(host_h2d_tensor, non_blocking=False)     # Host → GPU1
    with torch.cuda.stream(stream_p2p_recv):
        p2p_recv_tensor1.copy_(p2p_send_tensor0, non_blocking=False)     # GPU0 → GPU1
    torch.cuda.synchronize()
    t_both = time.time() - start
    print(f"overlap H2D + P2P cost: {t_both:.3f} seconds")

    # overlap D2D + P2P send
    start = time.time()
    with torch.cuda.stream(stream_d2d):
        d2d_to_tensor1.copy_(d2d_from_tensor1, non_blocking=False)     # GPU1 → GPU1
    with torch.cuda.stream(stream_p2p_send):
        p2p_recv_tensor2.copy_(p2p_send_tensor1, non_blocking=False)     # GPU1 → GPU2
    torch.cuda.synchronize()
    t_both = time.time() - start
    print(f"overlap D2D + P2P cost: {t_both:.3f} seconds")

    # overlap D2D + P2P recv
    start = time.time()
    with torch.cuda.stream(stream_d2d):
        d2d_to_tensor1.copy_(d2d_from_tensor1, non_blocking=False)     # GPU1 → GPU1
    with torch.cuda.stream(stream_p2p_recv):
        p2p_recv_tensor1.copy_(p2p_send_tensor0, non_blocking=False)     # GPU0 → GPU1
    torch.cuda.synchronize()
    t_both = time.time() - start
    print(f"overlap D2D + P2P cost: {t_both:.3f} seconds")

    # overlap H2D + D2H
    start = time.time()
    with torch.cuda.stream(stream_h2d):
        device_h2d_tensor1.copy_(host_h2d_tensor, non_blocking=True)     # Host → GPU1
    with torch.cuda.stream(stream_d2h):
        host_d2h_tensor.copy_(device_d2h_tensor1, non_blocking=True)     # GPU1 → Host
    torch.cuda.synchronize()
    t_both = time.time() - start
    print(f"overlap H2D + D2H cost: {t_both:.3f} seconds")

    # overlap D2D + D2H + H2D
    start = time.time()
    with torch.cuda.stream(stream_d2d):
        d2d_to_tensor1.copy_(d2d_from_tensor1, non_blocking=True)     # GPU1 → GPU1
    with torch.cuda.stream(stream_d2h):
        host_d2h_tensor.copy_(device_d2h_tensor1, non_blocking=True)     # GPU1 → Host
    with torch.cuda.stream(stream_h2d):
        device_h2d_tensor1.copy_(host_h2d_tensor, non_blocking=True)     # Host → GPU1
    torch.cuda.synchronize()
    t_both = time.time() - start
    print(f"overlap D2D + D2H + H2D cost: {t_both:.3f} seconds")

    # overlap D2D + P2P send + P2P recv
    start = time.time()
    with torch.cuda.stream(stream_d2d):
        d2d_to_tensor1.copy_(d2d_from_tensor1, non_blocking=True)     # GPU1 → GPU1
    with torch.cuda.stream(stream_p2p_send):
        p2p_recv_tensor2.copy_(p2p_send_tensor1, non_blocking=True)     # GPU1 → GPU2
    with torch.cuda.stream(stream_p2p_recv):
        p2p_recv_tensor1.copy_(p2p_send_tensor0, non_blocking=True)     # GPU0 → GPU1
    torch.cuda.synchronize()
    t_both = time.time() - start
    print(f"overlap D2D + P2P send + P2P recv cost: {t_both:.3f} seconds")

    # overlap D2D + H2D + P2P recv
    start = time.time()
    with torch.cuda.stream(stream_d2d):
        d2d_to_tensor1.copy_(d2d_from_tensor1, non_blocking=True)     # GPU1 → GPU1
    with torch.cuda.stream(stream_h2d):
        device_h2d_tensor1.copy_(host_h2d_tensor, non_blocking=True)     # Host → GPU1
    with torch.cuda.stream(stream_p2p_recv):
        p2p_recv_tensor1.copy_(p2p_send_tensor0, non_blocking=True)     # GPU0 → GPU1
    torch.cuda.synchronize()
    t_both = time.time() - start
    print(f"overlap D2D + H2D + P2P recv cost: {t_both:.3f} seconds")

    # overlap D2D + P2P send + P2P recv + H2D + D2H
    start = time.time()
    with torch.cuda.stream(stream_d2d):
        d2d_to_tensor1.copy_(d2d_from_tensor1, non_blocking=True)     # GPU1 → GPU1
    with torch.cuda.stream(stream_p2p_send):
        p2p_recv_tensor2.copy_(p2p_send_tensor1, non_blocking=True)     # GPU1 → GPU2
    with torch.cuda.stream(stream_p2p_recv):
        p2p_recv_tensor1.copy_(p2p_send_tensor0, non_blocking=True)     # GPU0 → GPU1
    with torch.cuda.stream(stream_h2d):
        device_h2d_tensor1.copy_(host_h2d_tensor, non_blocking=True)     # Host → GPU1
    with torch.cuda.stream(stream_d2h):
        host_d2h_tensor.copy_(device_d2h_tensor1, non_blocking=True)     # GPU1 → Host
    torch.cuda.synchronize()
    t_both = time.time() - start
    print(f"overlap D2D + P2P send + P2P recv + H2D + D2H cost: {t_both:.3f} seconds")
    print("--------------------------------------------------")