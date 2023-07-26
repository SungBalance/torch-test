import os

import torch
import torch.distributed as dist

# torchrun args

# LOCAL_RANK - The local rank.
# RANK - The global rank.
# GROUP_RANK - The rank of the worker group. A number between 0 and max_nnodes. When running a single worker group per node, this is the rank of the node.
# ROLE_RANK - The rank of the worker across all the workers that have the same role. The role of the worker is specified in the WorkerSpec.
# LOCAL_WORLD_SIZE - The local world size (e.g. number of workers running locally); equals to --nproc-per-node specified on torchrun.
# WORLD_SIZE - The world size (total number of workers in the job).
# ROLE_WORLD_SIZE - The total number of workers that was launched with the same role specified in WorkerSpec.
# MASTER_ADDR - The FQDN of the host that is running worker with rank 0; used to initialize the Torch Distributed backend.
# MASTER_PORT - The port on the MASTER_ADDR that can be used to host the C10d TCP store.
# TORCHELASTIC_RESTART_COUNT - The number of worker group restarts so far.
# TORCHELASTIC_MAX_RESTARTS - The configured maximum number of restarts.
# TORCHELASTIC_RUN_ID - Equal to the rendezvous run_id (e.g. unique job id).
# PYTHON_EXEC - System executable override. If provided, the python user script will use the value of PYTHON_EXEC as executable. The sys.executable is used by default.


local_rank = int(os.environ["LOCAL_RANK"])
global_rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])


tp_groups = [
    (1,2)
    ]
pp_groups = [
    list(range(world_size))
    for tp_group in tp_groups for tp_idx in tp_group
]


dist.init_process_group(backend="nccl")


if local_rank in [0,1]:
    tp_group = torch.distributed.new_group(ranks=[0,1], backend='nccl', pg_options=None)
    another_tp_group = torch.distributed.new_group(ranks=[2,3], backend='nccl', pg_options=None)
else:
    tp_group = torch.distributed.new_group(ranks=[2,3], backend='nccl', pg_options=None)
    another_tp_group = torch.distributed.new_group(ranks=[0,1], backend='nccl', pg_options=None)

tp_group_rank = torch.distributed.get_group_rank(tp_group, global_rank)
tp_global_rank = torch.distributed.get_global_rank(tp_group, tp_group_rank)
tp_process_group_rank = torch.distributed.get_process_group_ranks(tp_group)

if local_rank in [0,2]:
    pp_group = torch.distributed.new_group(ranks=[0,2], backend='nccl', pg_options=None)
else:
    pp_group = torch.distributed.new_group(ranks=[1,3], backend='nccl', pg_options=None)

pp_group_rank = torch.distributed.get_group_rank(pp_group, global_rank)
pp_global_rank = torch.distributed.get_global_rank(pp_group, pp_group_rank)
pp_process_group_rank = torch.distributed.get_process_group_ranks(pp_group)

output_dict = {
    'local_rank': local_rank,
    'global_rank': global_rank,
    'world_size': world_size,

    'tp_group_rank': tp_group_rank,
    'tp_global_rank': tp_global_rank,
    'tp_process_group_rank': tp_process_group_rank,

    'tp_group_rank': tp_group_rank,
    'tp_global_rank': tp_global_rank,
    'tp_process_group_rank': tp_process_group_rank,

    'another_tp_process_group_rank': torch.distributed.get_process_group_ranks(another_tp_group),

    'pp_group_rank': pp_group_rank,
    'pp_global_rank': pp_global_rank,
    'pp_process_group_rank': pp_process_group_rank,

    'torch.world_size': torch.distributed.get_world_size(),
    'tp_torch.world_size': torch.distributed.get_world_size(tp_group),
    'pp_torch.world_size': torch.distributed.get_world_size(pp_group),
}

print(f"{output_dict}\n", flush=True)


