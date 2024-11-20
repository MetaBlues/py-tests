import queue
import threading
import atexit

num_worker_threads = 1

def do_work(item):
    print(item)

def source():
    return range(10)

def worker():
    while True:
        item = q.get()
        if item is None:
            break
        do_work(item)
        q.task_done()

q = queue.Queue()

threads = []

for i in range(num_worker_threads):
    t = threading.Thread(target=worker)
    t.start()
    threads.append(t)

for item in source():
    q.put(item)

def sync():
    q.join()
    print('all tasks in queue are done')

sync_thread = threading.Thread(target=sync)
sync_thread.start()

def cleanup_and_shutdown_threads():
    sync_thread.join()
    for i in range(num_worker_threads):
        q.put(None)
    for t in threads:
        t.join()

def monitor_main_thread():
    main_thread = threading.main_thread()
    main_thread.join()
    cleanup_and_shutdown_threads()

monitor = threading.Thread(target=monitor_main_thread)
monitor.start()