import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import time
import argparse

import torch.nn.functional as f

SIZE1 = 4000
SIZE2 = 4000
send_tensors = [] # 
recv_tensors = [] # 
send_tensors2 = [] # 
recv_tensors2 = [] # 


matmul_out_send = []

def run(rank, idx):
    s = time.time()
    recv_req1 = dist.recv(tensor=recv_tensors[idx], src=0)
    e = time.time()
    print("recv1 time : ", e-s)

    forward_m1 = torch.nn.functional.linear(recv_tensors[idx], recv_tensors[idx])

    s = time.time()
    recv_req2 = dist.recv(tensor=recv_tensors2[idx], src=0)
    e = time.time()
    print("recv2 time : ", e-s)
    forward_m2 = torch.nn.functional.linear(recv_tensors2[idx], recv_tensors2[idx])

    torch.cuda.synchronize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--node_rank", type=int)
    #import pdb
    #pdb.set_trace()
    args = parser.parse_args()
    torch.cuda.set_device(args.node_rank)

    #backend = "gloo"
    backend = "nccl"
    init_method = "tcp://localhost:8009"
    dist.init_process_group(backend, rank=args.node_rank, world_size=2, init_method=init_method)

    LOOP_CNT = 5
    for i in range(LOOP_CNT):
        send_tensors.append(torch.rand((SIZE1,SIZE2)).cuda()) # for send
        send_tensors2.append(torch.rand((SIZE1,SIZE2)).cuda()) # for send
        recv_tensors.append(torch.empty([SIZE1,SIZE2]).cuda()) # for recv
        recv_tensors2.append(torch.empty([SIZE1,SIZE2]).cuda()) # for recv


    # before
    for idx in range(LOOP_CNT):
        r = recv_tensors[idx]
        print( f"before recv idx:{idx} val:{r[0][5]}" )

    # before
    for idx in range(LOOP_CNT):
        s = send_tensors[idx]
        print( f"before send idx:{idx} val:{s[0][5]}" )


    # send, recv
    for idx in range(LOOP_CNT):
        s = time.time()
        run(idx)
        ed = time.time()
        print("############################################")

    # after
    for idx in range(5):
       r = recv_tensors[idx]
       print( f"after recv idx:{idx} val:{r[0][5]}" )