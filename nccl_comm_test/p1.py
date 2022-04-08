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

def run(rank, idx):
    #O1 = torch.nn.functional.linear(recv_tensors[idx], recv_tensors[idx])
    forward_m1 = torch.nn.functional.linear(send_tensors[idx], send_tensors[idx])
    #torch.cuda.synchronize()
    #with time_check("first send"):
    send_req1 = dist.isend(forward_m1, dst=1)

    rr = torch.nn.functional.relu(DUMMY)
    #send_req1.wait()
    #print("call wait")
    #send_req1.wait()
    #print("call wait end")

    #time.sleep(0.010)
    #torch.cuda.synchronize()
    forward_m2 = torch.nn.functional.linear(send_tensors2[idx], send_tensors2[idx])
    #with time_check("second send"):
    send_req2 = dist.isend(forward_m2, dst=1)

    #recv_req1 = dist.irecv(tensor=recv_tensors[idx], src=1)
    #while( recv_req1.is_completed() == False ):
    #    continue

    #m1_backward = torch.nn.functional.linear(recv_tensors[idx], recv_tensors[idx])

    #recv_req2 = dist.irecv(tensor=recv_tensors2[idx], src=1)
    #while( recv_req2.is_completed() == False ):
    #    continue

    #m2_backward = torch.nn.functional.linear(recv_tensors[idx], recv_tensors[idx])


    torch.cuda.synchronize()
    time.sleep(0.1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()
    my_rank = args.local_rank
    print("my_rank : ", my_rank)
    torch.cuda.set_device(my_rank)

    #backend = "gloo"
    backend = "nccl"
    init_method = "tcp://localhost:8009"
    dist.init_process_group(backend, rank=my_rank, world_size=2, init_method=init_method)

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
    for i in range(LOOP_CNT):
        s = time.time()
        run(my_rank, i)
        ed = time.time()
        print("######################################")
        #print(f"{my_rank} out loop time : {ed-s} ")

    # after
    for i in range(5):
       r = recv_tensors[i]
       print( f"after recv idx:{i} val:{r[0][5]}" )

    # after
    #for i in range(5):
    #   r = matmul_out_send[i]
    #   print( f"after matmul out idx:{i} val:{r[0][5]}" )