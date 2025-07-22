FROM nvcr.io/nvidia/tritonserver:25.06-py3

RUN pip install torch torchvision torchaudio --pre --index-url https://download.pytorch.org/whl/nightly/cu128 --break-system-packages && \
    pip install onnx numpy transformers safetensors sentencepiece accelerate --break-system-packages && \
    pip install timm

RUN apt-get update && \
    apt-get install -y libopenmpi-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

EXPOSE 8000 8001 8002
CMD ["tritonserver"]