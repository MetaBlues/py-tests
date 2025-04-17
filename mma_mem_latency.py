sm90_fma_per_sm_cycle = 2048
sm100_fma_per_sm_cycle = 4096
smem_throughput_per_sm_cycle = 128 # byte

# smem pressure
m64n256k16_comp_latency = (64*256*16)//2048 # 128 cycle
m64n256k16_wgmma_smem_latency = 2 * (64*16+256*16) // 128 # 80 cycle
m64n256k16_mma_smem_latency = 4 * (32*16+128*16) * 2 // 128 # 160 cycle

# l2 pressure
l2_throughput_per_cycle = 3942.4 # byte
l2_throughput_per_sm_cycle = l2_throughput_per_cycle / 132\

# deepgemm fp8
BLOCK_M = [128, 256]
BLOCK_N = [128, 144, 160]
BLOCK_K = 128

for m in BLOCK_M:
    for n in BLOCK_N:
        fma_cycle = m * n  // sm90_fma_per_sm_cycle
        l2_cycle = (m + n) // l2_throughput_per_sm_cycle
        print(f"{m=} {n=} {fma_cycle/l2_cycle}")