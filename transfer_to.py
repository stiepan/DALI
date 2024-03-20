import socket
import os, os.path
import time
from collections import deque
import multiprocessing

import torch
import ctypes
from math import prod

from nvidia.dali import backend as _b

def as_tensor(pointer, shape, torch_type):
    arr = (pointer._type_ * prod(shape)).from_address(
        ctypes.addressof(pointer.contents))

    return torch.frombuffer(arr, dtype=torch_type).view(*shape)

if os.path.exists("my_asdf_socket.s"):
  os.remove("my_asdf_socket.s")

shape = (4 * 1024, 1024, 1024)
vol = prod(shape)

sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.bind("my_asdf_socket.s")
sock.listen(1)
conn_sock, addr = sock.accept()
shm_handle = multiprocessing.reduction.recv_handle(conn_sock)
shm = _b.SharedMem(shm_handle, vol)
print(shm.get_raw_ptr())
print(shm.cuda_attr())
shm.pin()
print(shm.cuda_attr())
p = ctypes.cast(shm.get_raw_ptr(), ctypes.POINTER(ctypes.c_uint8))
t = as_tensor(p, shape, torch.uint8)


time.sleep(1)
print(t)
