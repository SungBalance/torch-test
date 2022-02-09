import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import time
import argparse

import torch.nn.functional as f

from contextlib import contextmanager

@contextmanager
def time_check(name):
    s = time.time()
    yield
    e = time.time()
    total_time = e-s
    print(f"{name} : {total_time}")

SIZE1 = 4000
SIZE2 = 4000
send_tensors = [] # 
send_tensors2 = [] # 
recv_tensors = [] # 
recv_tensors2 = [] # 

matmul_out_send = []

def run(idx):
    forward_m1 = torch.nn.functional.linear(send_tensors[idx], send_tensors[idx])
    send_req1 = dist.send(forward_m1, dst=1)

    rr = torch.nn.functional.relu(DUMMY)
    forward_m2 = torch.nn.functional.linear(send_tensors2[idx], send_tensors2[idx])
    send_req2 = dist.send(forward_m2, dst=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()
    torch.cuda.set_device(args.local_rank)

    #backend = "gloo"
    backend = "nccl"
    init_method = "tcp://localhost:8009"
    dist.init_process_group(backend, rank=args.local_rank, world_size=2, init_method=init_method)

    DUMMY = torch.rand((1,1)).cuda()

    LOOP_CNT = 5
    for i in range(LOOP_CNT):
        send_tensors.append(torch.rand((SIZE1,SIZE2)).cuda()) # for send
        send_tensors2.append(torch.rand((SIZE1,SIZE2)).cuda()) # for send
        #recv_tensors.append(torch.empty([SIZE1,SIZE2], device=torch.cuda.current_device()) ) # for recv
        recv_tensors.append(torch.empty([SIZE1,SIZE2]).cuda()) # for recv
        recv_tensors2.append(torch.empty([SIZE1,SIZE2]).cuda()) # for recv


    # before
    for i in range(LOOP_CNT):
        r = recv_tensors[i]
        print( f"before recv idx:{i} val:{r[0][5]}" )

    # before
    for i in range(LOOP_CNT):
        s = send_tensors[i]
        print( f"before send idx:{i} val:{s[0][5]}" )


    # send, recv
    for idx in range(LOOP_CNT):
        s = time.time()
        run(idx)
        ed = time.time()
        print("######################################")
    # after
    for idx in range(5):
       r = recv_tensors[idx]
       print( f"after recv idx:{i} val:{r[0][5]}" )