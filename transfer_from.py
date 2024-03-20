import socket
import os, os.path
import time
import multiprocessing

import torch
import ctypes
from math import prod

from nvidia.dali import backend as _b

shape = (4 * 1024, 1024, 1024)
vol = prod(shape)

shm = _b.SharedMem(-1, vol)

print(shm.get_raw_ptr())

print("create tensor")
t_gpu = torch.arange(0, vol, dtype=torch.uint8).reshape(shape).to("cuda:0")
print("created tensor, creating socket")
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect("my_asdf_socket.s")
multiprocessing.reduction.send_handle(sock, shm.handle, None)
print("sent")


print("starting async copy")
shm.pin()  # comment me out for the copy to take longer
s = time.time()
shm.mem_async_cpy_from_device(t_gpu.data_ptr())
e = time.time()
print(e - s)
print("finished async copy")

