# folder struktura

├── hgf_models
│    └── Mistral-7B-Instruct-v0.3
├── model_repo
├── models
└── triton_model_repo
    └── mistral
        └── 1


#triton 3 LLMa
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

# tensorrt

###### tensorrt dio

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

# gemma problemi kod pokretanja testova sa user strane
E0724 06:29:07.070951 1 pb_stub.cc:736] "Failed to process the request(s) for model 'gemma3_0_0', message: FailOnRecompileLimitHit: recompile_limit reached with one_graph=True or error_on_graph_break=True. Excessive recompilations can degrade performance due to the compilation overhead of each recompilation. To monitor recompilations, enable TORCH_LOGS=recompiles. If recompilations are expected, consider increasing torch._dynamo.config.cache_size_limit to an appropriate value