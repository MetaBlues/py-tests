# Author: kevin.xie  zhihu@kaiyuan

import torch
from torch import nn
from datetime import datetime

B = 1
S = 2048 * 8
D = 512
A = 8
def train(num_iter=5, device="cuda:0"):
    decoder_layer = nn.TransformerDecoderLayer(d_model=D, nhead=A, dtype=torch.bfloat16)
    model = nn.TransformerDecoder(decoder_layer, num_layers=6).to(device=device)
    x = torch.randn(size=(B, S, D), dtype=torch.bfloat16, device=device)
    tgt = torch.rand(size=(B, S, D), dtype=torch.bfloat16, device=device)
    model.train()
    labels = torch.rand_like(model(x, tgt))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters())
    for _ in range(num_iter):
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            y = model(x, tgt)
        loss = criterion(y, labels)
        loss.backward()
        print(loss.item())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


def run():
    # Start recording memory snapshot history
    torch.cuda.memory._record_memory_history()
    # training running:
    train()

    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    file_name = f"visual_mem_{timestamp}.pickle"
    # save record:
    torch.cuda.memory._dump_snapshot(file_name)
    # file_name = f"rank_{torch.distributed.get_rank()}.pickle"
    # if torch.distributed.get_rank() == 0:
    #     torch.cuda.memory._dump_snapshot(file_name)

    # Stop recording memory snapshot history:
    torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == "__main__":
    run()