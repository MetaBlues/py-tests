import torch
import time

# 设置矩阵大小
matrix_size_list = [4096, 8192, 16384, 32768]

for matrix_size in matrix_size_list:
    # 设置迭代次数
    num_iterations = 100

    # 创建随机矩阵
    matrix1 = torch.randn(matrix_size, matrix_size, dtype=torch.float32).cuda()
    matrix2 = torch.randn(matrix_size, matrix_size, dtype=torch.float32).cuda()

    # FP32性能测试
    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(num_iterations):
        result = torch.matmul(matrix1, matrix2)

    torch.cuda.synchronize()
    end_time = time.time()

    fp32_time = end_time - start_time
    fp32_tflops = (2 * matrix_size ** 3 * num_iterations) / (fp32_time * 1e12)

    # BF16性能测试
    matrix1_bf16 = matrix1.bfloat16()
    matrix2_bf16 = matrix2.bfloat16()

    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(num_iterations):
        result = torch.matmul(matrix1_bf16, matrix2_bf16)

    torch.cuda.synchronize()
    end_time = time.time()

    bf16_time = end_time - start_time
    bf16_tflops = (2 * matrix_size ** 3 * num_iterations) / (bf16_time * 1e12)

    print(f"Matrix size: {matrix_size}, FP32 Performance: {fp32_tflops:.2f} TFLOPS, BF16 Performance: {bf16_tflops:.2f} TFLOPS")