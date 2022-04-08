import torch
import torch.nn as nn

def loading_hook_fn(module, input):
    module.cuda(torch.cuda.current_device())

def offloading_hook_fn(module, input, output):
    module.cpu()


# def get_offloaded_model(model):
#     for idx, module in model.named_parameters():
#         module.