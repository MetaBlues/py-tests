import torch.multiprocessing as multiprocessing
import time

def worker_process(queue1, queue2, shared_status):
    while True:
        obj = queue1.get()
        if obj == 'sync':
            queue2.put('done')
            continue
        else:
            # 模拟工作状态
            with shared_status.get_lock():
                shared_status.value = 1  # busy
            print(obj, flush=True)
            # 模拟空闲状态
            with shared_status.get_lock():
                shared_status.value = 0  # idle

def main():
    # 创建共享变量
    ctx = multiprocessing.get_context('spawn')
    queue1 = ctx.Queue()
    queue2 = ctx.Queue()
    shared_status = ctx.Value('i', 0) # must created by ctx.Value(), not by multiprocessing.Value()

    # 创建并启动子进程
    p = ctx.Process(target=worker_process, args=(queue1, queue2, shared_status))
    p.start()
    queue1.put(12345)

    def sync():
        queue1.put('sync')
        queue2.get()  # 等待子进程确认同步

    while True:
        with shared_status.get_lock():
            status = shared_status.value
        if status == 0:
            sync()  # 进行同步
            break
        time.sleep(2)

    queue1.put(23456)
    sync()  # 再次进行同步

    p.terminate()
    p.join()

if __name__ == '__main__':
    main()
