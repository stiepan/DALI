import multiprocessing as mp
from multiprocessing import shared_memory

def modify(buf_name):
    shm = shared_memory.SharedMemory(buf_name)
    shm.buf[0:50] = b"b" * 50
    shm.close()

if __name__ == "__main__":
    shm = shared_memory.SharedMemory(create=True, size=100)

    try:
        shm.buf[0:100] = b"a" * 100
        proc = mp.Process(target=modify, args=(shm.name,))
        proc.start()
        proc.join()
        name = input("Enter your name: ")
        print(bytes(shm.buf[:100]))
        print(dir(shm))
    finally:
        shm.close()
        shm.unlink()