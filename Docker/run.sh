sudo docker run -d -it \
    -v ${PWD}:/workspace/bdsl \
    --name sk-torch-test \
    --net=host \
    --ipc=host \
    --gpus all \
    nvcr.io/nvidia/pytorch:21.12-py3 bash