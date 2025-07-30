# NVIDIA TRITON INFERENCE SERVER

## Folder struktura za pojedini model
```bash 
├── hgf_models
│    └── Mistral-7B-Instruct-v0.3
├── model_repo
├── models
└── triton_model_repo
    └── mistral
        └── 1
```

## Skripta za preuzimanje modela sa huggingface-a
```python
    from huggingface_hub import snapshot_download
    
    snapshot_download(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        local_dir="/raid/models/hgf_models/Mistral-7B-Instruct-v0.3",
        local_dir_use_symlinks=False  # ensures actual files, not symlinks
    )
```

## Konfiguracija za model korištenjem pytorch-a
```python
    import triton_python_backend_utils as pb_utils
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    import numpy as np
    
    torch.set_float32_matmul_precision("high")
    
    
    class TritonPythonModel:
        def initialize(self, args):
            self.tokenizer = AutoTokenizer.from_pretrained("/models/hf/mistral")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                "/models/hf/mistral",
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            self.model.eval()
    
        def execute(self, requests):
            responses = []
            for request in requests:
                input_tensor = pb_utils.get_input_tensor_by_name(request, "text_input")
                input_array = input_tensor.as_numpy()  # shape: (batch_size, 1)
                input_texts = [x[0].decode("utf-8") for x in input_array]
    
                # Tokenize as a batch
                inputs = self.tokenizer(input_texts, return_tensors="pt", padding=True).to("cuda")
                outputs = self.model.generate(**inputs, max_new_tokens=128)
                decoded = [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
    
                output_tensor = pb_utils.Tensor(
                    "text_output", np.array([d.encode('utf-8') for d in decoded], dtype=object)
                )
                responses.append(pb_utils.InferenceResponse([output_tensor]))
            return responses

```

## Konzolni ispis pokretanja Triton Inference Servera
```bash 
I0714 06:49:34.597221 1 model_lifecycle.cc:473] "loading: gemma3:1"
I0714 06:49:34.597239 1 model_lifecycle.cc:473] "loading: meta-llama:1"
I0714 06:49:34.597250 1 model_lifecycle.cc:473] "loading: mistral:1"
...
I0714 06:55:31.200517 1 server.cc:681]
+------------+---------+--------+
| Model      | Version | Status |
+------------+---------+--------+
| gemma3     | 1       | READY  |
| meta-llama | 1       | READY  |
| mistral    | 1       | READY  |
+------------+---------+--------+


I0715 16:28:08.551653 1 model_lifecycle.cc:473] "loading: mistral:1"
I0715 16:28:15.631006 1 python_be.cc:2289] "TRITONBACKEND_ModelInstanceInitialize: mistral_0_0 (GPU device 0)"
I0715 16:28:15.631084 1 python_be.cc:2289] "TRITONBACKEND_ModelInstanceInitialize: mistral_1_0 (GPU device 1)"
I0715 16:28:15.631151 1 python_be.cc:2289] "TRITONBACKEND_ModelInstanceInitialize: mistral_3_0 (GPU device 3)"
I0715 16:28:15.631170 1 python_be.cc:2289] "TRITONBACKEND_ModelInstanceInitialize: mistral_2_0 (GPU device 2)"
I0715 16:28:15.631205 1 python_be.cc:2289] "TRITONBACKEND_ModelInstanceInitialize: mistral_4_0 (GPU device 4)"
I0715 16:28:15.631235 1 python_be.cc:2289] "TRITONBACKEND_ModelInstanceInitialize: mistral_5_0 (GPU device 5)"
I0715 16:28:15.631288 1 python_be.cc:2289] "TRITONBACKEND_ModelInstanceInitialize: mistral_6_0 (GPU device 6)"
I0715 16:28:15.631419 1 python_be.cc:2289] "TRITONBACKEND_ModelInstanceInitialize: mistral_7_0 (GPU device 7)"
715 16:30:27.666494 1 python_be.cc:2289] "TRITONBACKEND_ModelInstanceInitialize: meta-llama_1_0 (GPU device 1)"
I0715 16:30:27.667271 1 python_be.cc:2289] "TRITONBACKEND_ModelInstanceInitialize: meta-llama_0_0 (GPU device 0)"
I0715 16:30:27.668023 1 python_be.cc:2289] "TRITONBACKEND_ModelInstanceInitialize: meta-llama_2_0 (GPU device 2)"
I0715 16:30:27.668809 1 python_be.cc:2289] "TRITONBACKEND_ModelInstanceInitialize: meta-llama_4_0 (GPU device 4)"
I0715 16:30:27.669749 1 python_be.cc:2289] "TRITONBACKEND_ModelInstanceInitialize: meta-llama_6_0 (GPU device 6)"
I0715 16:30:27.670984 1 python_be.cc:2289] "TRITONBACKEND_ModelInstanceInitialize: meta-llama_3_0 (GPU device 3)"
I0715 16:30:27.672273 1 python_be.cc:2289] "TRITONBACKEND_ModelInstanceInitialize: meta-llama_5_0 (GPU device 5)"
I0715 16:30:27.732455 1 python_be.cc:2289] "TRITONBACKEND_ModelInstanceInitialize: meta-llama_7_0 (GPU device 7)"
I0715 16:30:27.848899 1 python_be.cc:2289] "TRITONBACKEND_ModelInstanceInitialize: gemma3_0_0 (GPU device 0)"
```

## Osnovni config.pbtxt za model gemma3
U ovom primjeru koristimo ONNX model gemma3, koji je već konvertiran i spreman za korištenje u Triton Inference Serveru.
```protobuf
config.pbtxt
name: "rt-gemma3"
platform: "onnxruntime_onnx"
max_batch_size: 8

input [
  {
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [ -1 ]
  },
  {
    name: "onnx::Gather_1"
    data_type: TYPE_INT64
    dims: [ 1 ]
  }
]

output [
  {
    name: "logits"
    data_type: TYPE_FP32   # Or whatever exactly your ONNX provides!
    dims: [ -1 ]
  }
]
```