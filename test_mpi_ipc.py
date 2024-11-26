# /opt/conda/bin/conda install -n py310 -c conda-forge gcc_linux-64 gxx_linux-64 gfortran_linux-64
from mpi4py import MPI
import pickle

def send_dict(comm, data, dest):
    """将字典序列化后发送"""
    serialized_data = pickle.dumps(data)  # 序列化字典
    comm.Send([serialized_data, MPI.BYTE], dest=dest)

def recv_dict(comm, source):
    """接收字典并反序列化"""
    status = MPI.Status()
    # 先接收数据的大小
    comm.Probe(source=source, tag=0, status=status)
    size = status.Get_count(MPI.BYTE)
    serialized_data = bytearray(size)
    comm.Recv(serialized_data, source=source, tag=0)
    data = pickle.loads(serialized_data)  # 反序列化字典
    return data

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size < 2:
        print("至少需要两个进程才能运行这个示例")
        return

    if rank == 0:
        # 进程 0 构造一个字典
        my_dict = {'name': 'Alice', 'age': 30, 'city': 'New York'}
        print(f"进程 {rank} 发送字典: {my_dict}")
        send_dict(comm, my_dict, dest=1)  # 发送给进程 1

    elif rank == 1:
        # 进程 1 接收字典
        received_dict = recv_dict(comm, source=0)
        print(f"进程 {rank} 接收到字典: {received_dict}")

if __name__ == "__main__":
    main()
