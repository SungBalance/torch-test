NUM_TRAINERS=4

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=$NUM_TRAINERS \
    group_test.py