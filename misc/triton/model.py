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
