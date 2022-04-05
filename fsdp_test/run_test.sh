export GPU_NUM=3

export GPUS_PER_NODE=1
export MASTER_ADDR=localhost
export MASTER_PORT=8013
export NNODES=1
export NODE_RANK=0
export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

CUDA_VISIBLE_DEVICES=$GPU_NUM python -m torch.distributed.launch $DISTRIBUTED_ARGS test.py