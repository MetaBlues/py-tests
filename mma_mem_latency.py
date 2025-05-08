sm90_flops_per_sm_cycle = 4096
sm100_flops_per_sm_cycle = 4096 * 2
# half precision
sm90_fma_half_per_sm_cycle = sm90_flops_per_sm_cycle // 2
sm100_fma_half_per_sm_cycle = sm100_flops_per_sm_cycle // 2

smem_throughput_per_sm_cycle = 128 # byte
# l2_throughput_per_cycle = 3942.4 # byte, float4
l2_throughput_per_cycle = 4472.3 # byte, float

# smem pressure
m64n256k16_comp_latency = (64*256*16) // sm90_fma_half_per_sm_cycle # 128 cycle
m64n256k16_wgmma_smem_latency = 2 * (64*16+256*16) // 128 # 80 cycle
m64n256k16_mma_smem_latency = 4 * (32*16+128*16) * 2 // 128 # 160 cycle

# l2 pressure
BLOCK_M = [128, 256]
BLOCK_N = [64, 128, 256]
BLOCK_K = 128

sizeof_bf16 = 2
sizeof_fp8 = 1

M = 256
N = 8192

for m in BLOCK_M:
    for n in BLOCK_N:
        tiles = (M // m) * (N // n)
        tiles = tiles if tiles < 132 else 132 # TODO: 还是需要了解 gemm pipeline
        fma_cycle = m * n // sm90_fma_half_per_sm_cycle
        l2_cycle = (m + n) * sizeof_bf16 // (l2_throughput_per_cycle / tiles)
        print(f"{m=} {n=} {tiles=} overlap_raito={fma_cycle/l2_cycle:.02f}")
