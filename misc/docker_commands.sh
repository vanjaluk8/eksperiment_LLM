

# tesorRT container with GPU support
docker run --gpus all --rm -it -v /raid/models/hf:/workspace/hf nvcr.io/nvidia/tensorrt:25.06-py3