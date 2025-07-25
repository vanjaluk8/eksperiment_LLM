import pandas as pd
from transformers import AutoTokenizer

def get_token_counter(model_name):
    MODEL_MAP = {
        "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
        "gemma": "google/gemma-7b",
        "meta-llama": "meta-llama/Llama-3.1-8B-Instruct",
    }

    name = model_name.lower()
    for prefix, hf_id in MODEL_MAP.items():
        if prefix in name:
            tokenizer = AutoTokenizer.from_pretrained(hf_id)

            def count(text):
                return len(tokenizer.encode(text, add_special_tokens=False))
            return count

    raise ValueError(f"Model {model_name} is not supported in get_token_counter")

def build_metrics(test_name, model, prompt, output_text, prompt_idx, response, latency, count_tokens):
    ''' gradi metrike iz odgovora i prompta koje se spremaju u df'''
    num_tokens = count_tokens(output_text)
    prompt_len = count_tokens(prompt)
    ttft = response.elapsed.total_seconds()
    tpot = latency / max(num_tokens, 1)
    tokens_per_sec = num_tokens / max(latency, 1e-6)
    return {
        "timestamp": pd.Timestamp.now(),
        "test_name": test_name,
        "prompt_index": prompt_idx,
        "model": model,
        "prompt": prompt,
        "ttft": ttft,
        "tpot": tpot,
        "tokens_per_sec": tokens_per_sec,
        "latency": latency,
        "prompt_tokens": prompt_len,
        "output_tokens": num_tokens,
        "output_text": output_text,
        "status_code": response.status_code
    }