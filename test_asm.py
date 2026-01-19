from collections import defaultdict

import torch
import triton
import triton.language as tl
from triton.language.target_info import cuda_capability_geq

# class RbitsCache:
#     seed: int = 1234
#     cache: dict = defaultdict(list)
#     @classmethod
#     @triton.jit
#     def get_rbits(cls, SIZE):
#         if len(cls.cache[SIZE]) == 0:
#             cls.cache[SIZE] = list(tl.randint4x(cls.seed, tl.arange(0, SIZE)))
#             cls.seed += 1
#         return cls.cache[SIZE].pop()

@triton.jit
def cvt_to_f8x4_with_stochastic_rounding(
    X,
    Y,
    seed,
    x_numel,
    SR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tensor = tl.load(X + tl.arange(0, BLOCK), mask=tl.arange(0, BLOCK) < x_numel, other=0.0)
    tensor = tensor.to(tl.float32)
    if SR:
        #FIXME:
        rbits = tl.randint(seed, tl.arange(0, BLOCK))
        rbits = (rbits & 0xFF).to(tl.uint8)
        out_tensor = tl.inline_asm_elementwise(
            asm=
            """
            {
            cvt.rs.satfinite.e4m3x4.f32 $0, {$4, $3, $2, $1}, $5;
            }
            """,
            constraints=(
                # output: 1×uint32 containing 4 packed FP8 values
                "=r,"
                # inputs: 4×f32 + 1xuint32 containing 4 packed uint8 values
                "r,r,r,r,r"
            ),
            args=[tensor, rbits],
            dtype=tl.uint8,
            is_pure=True,
            pack=4,
        )
        tl.store(Y + tl.arange(0, BLOCK), out_tensor.to(tl.float8e4nv, bitcast=True), mask=tl.arange(0, BLOCK) < x_numel)
    else:
        tl.store(Y + tl.arange(0, BLOCK), tensor.to(tl.float8e4nv, fp_downcast_rounding="rtne"), mask=tl.arange(0, BLOCK) < x_numel)
    # return fp8_out



def run_once(seed, SR=True):
    assert cuda_capability_geq(10, 0)
    x_fp32 = torch.tensor([
        0.124, 0.126, 0.128, 0.130,
        1.003, 1.004, 1.005, 1.006,
        -0.501, -0.499, -0.498, -0.497,
        3.9, 4.0, 4.1, 4.2
    ], dtype=torch.float32, device="cuda")

    y_fp8 = torch.empty_like(x_fp32, dtype=torch.float8_e4m3fn, device='cuda')

    grid = (1,)
    BLOCK = triton.next_power_of_2(x_fp32.numel())
    cvt_to_f8x4_with_stochastic_rounding[grid](
        x_fp32,
        y_fp8,
        seed,
        x_fp32.numel(),
        SR=SR, # SR Flag
        BLOCK=BLOCK,
    )
    return y_fp8


out1 = run_once(seed=123)
out2 = run_once(seed=456)
out_rtn = run_once(seed=123, SR=False)

print("FP8 (seed=123):")
print(out1)

print("\nFP8 (seed=456):")
print(out2)

print("\nFP8 (RTNE):")
print(out_rtn)