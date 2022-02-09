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
    recv_req1 = dist.irecv(tensor=recv_tensors[idx], src=0)
    e = time.time()
    print("recv1 time : ", e-s)
    #while( recv_req1.is_completed() == False ):
    #    continue
    forward_m1 = torch.nn.functional.linear(recv_tensors[idx], recv_tensors[idx])

    s = time.time()
    recv_req2 = dist.irecv(tensor=recv_tensors2[idx], src=0)
    e = time.time()
    print("recv2 time : ", e-s)

    #bacward_m1 = torch.nn.functional.linear(forward_m1,forward_m1)
    #send_req1 = dist.isend(tensor=bacward_m1, dst=0)

    while( recv_req2.is_completed() == False ):
        continue
    forward_m2 = torch.nn.functional.linear(recv_tensors2[idx], recv_tensors2[idx])
    #bacward_m2 = torch.nn.functional.linear(forward_m2,forward_m2)
    #send_req2 = dist.isend(tensor=bacward_m2, dst=0)


    #send_req2.wait()
    #time.sleep(0.1)

    #send_req = dist.isend(tensor=send_tensors[idx], dst=0)
    #matmul_out_send.append(O1)


    #while( send_req.is_completed() == False ):
    #    continue

    #recv_req.wait()
    #send_req.wait()


    #while( send_req.is_completed() == False ):
    #    continue

    torch.cuda.synchronize()
    #ed = time.time()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--node_rank", type=int)
    #import pdb
    #pdb.set_trace()
    args = parser.parse_args()
    #my_rank = args.local_rank
    my_rank = args.node_rank
    print("my_rank : ", my_rank)
    torch.cuda.set_device(my_rank)

    #backend = "gloo"
    backend = "nccl"
    init_method = "tcp://localhost:8009"
    dist.init_process_group(backend, rank=my_rank, world_size=2, init_method=init_method)

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
        #print(f"{my_rank} out loop time : {ed-s} ")
        print("############################################")

    # after
    for i in range(5):
       r = recv_tensors[i]
       print( f"after recv idx:{i} val:{r[0][5]}" )

    # after
    #for i in range(5):
    #   r = matmul_out_send[i]
    #   print( f"after matmul out idx:{i} val:{r[0][5]}" )