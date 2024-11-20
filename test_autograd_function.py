import torch
import nvtx

class MyFunction(torch.autograd.Function):
    @staticmethod
    @nvtx.annotate("forward", color="green")
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input + 1, input * 2

    @staticmethod
    @nvtx.annotate("backward", color="purple")
    def backward(ctx, *grad_output):
        input, = ctx.saved_tensors
        return grad_output[0] + grad_output[1]

prof = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=5,warmup=5,active=10,repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        with_modules=True
        )
# Profiling starts here
# prof.start()

a = torch.rand([4], device=torch.device('cuda'), requires_grad=True)
with nvtx.annotate("back_prop", color="orange"):
    b, bb = MyFunction.apply(a)
    c = b + bb
    c.sum().backward() # a.grad.fill_(0)
print(f"{c.grad_fn=}, {b.grad_fn=}, {bb.grad_fn=}, {a.grad_fn=}") 
print(c.grad_fn.next_functions) # ((<torch.autograd.function.MyFunctionBackward object at 0x7f1318ee0340>, 0), (<torch.autograd.function.MyFunctionBackward object at 0x7f1318ee0340>, 1)) 

c_grad = torch.tensor((1.0,))
b_grad, bb_grad = c.grad_fn(c_grad)
print(f"{b_grad=}, {bb_grad=}")

# prof.step()
# prof.stop()
