sudo docker run -d -it \
    -v ${PWD}:/workspace/mlsys \
    --name sk-torch-test \
    --net=host \
    --ipc=host \
    --gpus all \
    nvcr.io/nvidia/pytorch:23.06-py3 bash