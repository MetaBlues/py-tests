from contextlib import contextmanager
import torch


class MyClass:
    a = 1

    def __del__(self):
        print("finilize obj")


@contextmanager
def mycontext():
    print("enter context")
    a = list()
    a.append(MyClass())

    def pack_hook(x):
        a.append(x)
        return x

    def unpack_hook(x):
        print(a[0].a) # if #L33 is commented, a will be realeased at unpack
        return x


    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        yield

    del a
    # a.clear()
    print("exit context")


x = torch.randn((3, 4), requires_grad=True)

def func():
    with mycontext():
        y = x.pow(2)
        print(f" in context")
    print(f" outside context")

    y.backward(torch.ones_like(y))
    print(f" end backward")

func()

print(f" outside function")
