sm90_fma_per_sm_cycle = 2048
sm100_fma_per_sm_cycle = 4096
smem_throughput_per_sm_cycle = 128 # byte

# smem pressure
m64n256k16_comp_latency = (64*256*16)//2048 # 128 cycle
m64n256k16_wgmma_smem_latency = 2 * (64*16+256*16) // 128 # 80 cycle
m64n256k16_mma_smem_latency = 4 * (32*16+128*16) * 2 // 128 # 160 cycle

# l2 pressure
# l2_throughput_per_cycle = 3942.4 # byte # float4
l2_throughput_per_cycle = 4472.3 # byte # float
l2_throughput_per_sm_cycle = l2_throughput_per_cycle / 132

# deepgemm fp8
BLOCK_M = [64, 128, 256]
BLOCK_N = [64, 128, 144, 160]
BLOCK_K = 128

for m in BLOCK_M:
    for n in BLOCK_N:
        fma_cycle = m * n  // sm90_fma_per_sm_cycle
        l2_cycle = (m + n) // l2_throughput_per_sm_cycle
        print(f"{m=} {n=} {fma_cycle/l2_cycle}")

# m=64 n=64 0.6666666666666666
# m=64 n=128 0.8
# m=64 n=144 0.6666666666666666
# m=64 n=160 0.8333333333333334
# m=128 n=64 0.8
# m=128 n=128 1.1428571428571428
# m=128 n=144 1.125
# m=128 n=160 1.25
# m=256 n=64 0.8888888888888888
# m=256 n=128 1.4545454545454546
# m=256 n=144 1.6363636363636365
# m=256 n=160 1.6666666666666667