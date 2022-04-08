import argparse, time

import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel,
    CPUOffload,
)

import numpy as np

import hooks

from captum.attr import IntegratedGradients

torch.manual_seed(123)
np.random.seed(123)

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fsdp', type=bool, default=False)

    return parser.parse_args()

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(256, 512)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(512, 512)

    def forward(self, input):
        return self.lin2(self.relu(self.lin1(input)))

def do_experiment(model):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
    
    input = torch.rand(256, 256).cuda()
    baseline = torch.zeros(256, 256).cuda()
    
    ig = IntegratedGradients(model)

    start_time = time.time()
    attributions, delta = ig.attribute(input, baseline, target=0, return_convergence_delta=True)
    end_time = time.time()

    print(f'peak_gpu_memory: {torch.cuda.max_memory_allocated(torch.cuda.current_device())/1024} KB')
    print(f'execution time: {end_time - start_time}')
    # print('IG Attributions:', attributions)
    # print('Convergence Delta:', delta)


if __name__ == '__main__':

    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    model = ToyModel().cuda()

    print("\n\n\n\n\n\nbaseline")
    do_experiment(model)

    print("\n\nwith saved_tensor to CPU")
    with torch.autograd.graph.save_on_cpu(pin_memory=True):
        do_experiment(model)


    # Test for Big Model
    print("\n\n\n\n\n\n===========TEST FOR BIG MODEL===========")

    from .models.beyond_the_spectrum.model import BeyondtheSpectrumModel
    from .models.beyond_the_spectrum.options import Options
    model = BeyondtheSpectrumModel(opt=Options('pixel'))
    print("\n\nbaseline")

    try:     
        do_experiment(model)
    except:
        print("FAILED")

    print("\n\nwith saved_tensor to CPU")

    try:     
        with torch.autograd.graph.save_on_cpu(pin_memory=True):
            do_experiment(model)
    except:
        print("FAILED")

    # model_fsdp = FullyShardedDataParallel(
    #     model,
    #     cpu_offload=CPUOffload(offload_params=True)
    # )

    # print("\n\nwith FSDP")
    # try:
    #     do_experiment(model_fsdp)
    # except:
    #     print("FAILED")