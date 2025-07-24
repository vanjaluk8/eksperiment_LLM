import json
import time
import httpx
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


def send_infer_request(url, prompt, count_tokens):
    headers = {"Content-Type": "application/json"}
    payload = {
        "inputs": [
            {
                "name": "text_input",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [prompt]
            }
        ]
    }

    start_time = time.time()
    response = httpx.post(url, headers=headers, json=payload, timeout=180.0)
    end_time = time.time()

    total_time = end_time - start_time
    try:
        output_text = json.loads(response.text)["outputs"][0]["data"][0]
    except Exception:
        output_text = ""

    num_tokens = count_tokens(output_text)
    prompt_len = count_tokens(prompt)
    ttft = response.elapsed.total_seconds()
    generation_time = total_time

    tpot = generation_time / max(num_tokens, 1) # seconds/token
    tokens_per_sec = num_tokens / max(generation_time, 1e-6)

    # Console log for humans
    print(f"Model URL: {url}")
    print(f"Prompt (truncated): {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
    print(f"Output (truncated): {output_text[:100]}{'...' if len(output_text) > 100 else ''}")
    print(f"TTFT: {ttft:.5f}s, TPOT: {tpot * 1000:.3f} ms/tok, Throughput: {tokens_per_sec:.1f} tok/s")
    print(f"Prompt tokens: {prompt_len}, Output tokens: {num_tokens}")
    print(f"Generation time: {generation_time:.3f}s")
    print(f"Total time: {total_time:.3f}s")
    print("-" * 40)

    return {
        "response": response,
        "num_tokens": num_tokens,
        "prompt_len": prompt_len,
        "ttft": ttft,
        "tpot": tpot,  # seconds/token
        "tokens_per_sec": tokens_per_sec,
        "latency": total_time,
        "output_text": output_text
    }


def collect_metrics(llm, triton_url, prompt, is_stress_test=False, count_tokens=None):
    result = send_infer_request(
        f"{triton_url}/v2/models/{llm}/infer", prompt, count_tokens
    )
    return {
        "timestamp": pd.Timestamp.now(),
        "test_name": "stress",
        "prompt_index": 0,
        "model": llm,
        "prompt": f"{prompt}" if is_stress_test else prompt,
        "ttft": result["ttft"],
        "tpot": result["tpot"],          # ms/token
        "tokens_per_sec": result["tokens_per_sec"],
        "latency": result["latency"],
        "num_tokens": result["num_tokens"],
        "status_code": result["response"].status_code,
    }


def run_chat_like_test(llm, triton_url, initial_prompt, count_tokens, n_turns=5):
    """Simulate a chat with N turns, appending each assistant output to the prompt."""
    chat_history = ""
    prompt_start = initial_prompt
    metrics_results = []

    for t in range(n_turns):
        if t == 0:
            full_prompt = f"User: {prompt_start}\nAssistant:"
        else:
            user_input = f"Turn {t}: OK, elaborate more."
            full_prompt = chat_history + f"\nUser: {user_input}\nAssistant:"

        result = send_infer_request(
            f"{triton_url}/v2/models/{llm}/infer",
            full_prompt,
            count_tokens
        )

        metrics = {
            "timestamp": pd.Timestamp.now(),
            "test_name": "chat",
            "prompt_index": t,
            "model": llm,
            "prompt": full_prompt,
            "ttft": result["ttft"],
            "tpot": result["tpot"],        # ms/token
            "tokens_per_sec": result["tokens_per_sec"],
            "latency": result["latency"],
            "prompt_tokens": result["prompt_len"],
            "output_tokens": result["num_tokens"],
            "assistant_output": result["output_text"]
        }
        metrics_results.append(metrics)

        if t == 0:
            chat_history = f"User: {prompt_start}\nAssistant: {result['output_text']}"
        else:
            chat_history += f"\nUser: Turn {t}: Tell me something new.\nAssistant: {result['output_text']}"

    return pd.DataFrame(metrics_results)


def run_summarize_document_test(llm, triton_url, document, count_tokens):
    """Run a summarization test on a given document and return a DataFrame."""
    prompt = f"Summarize the following document in 1500 words:\n\n{document}\n\nSummary:"
    result = send_infer_request(f"{triton_url}/v2/models/{llm}/infer", prompt, count_tokens)

    metrics = {
        "timestamp": pd.Timestamp.now(),
        "test_name": "summarizer",
        "prompt_index": 0,
        "model": llm,
        "prompt": prompt,
        "ttft": result["ttft"],
        "tpot": result["tpot"],         # ms/token
        "tokens_per_sec": result["tokens_per_sec"],
        "latency": result["latency"],
        "num_tokens": result["num_tokens"],
        "status_code": result["response"].status_code,
        "summary_output": result["output_text"]
    }

    return pd.DataFrame([metrics])

def run_batch_prompt_test(llm, url, prompts, count_tokens):
    headers = {"Content-Type": "application/json"}

    payload = {
        "inputs": [
            {
                "name": "text_input",
                "shape": [len(prompts), 1],  # ← SIMULATE batch size here
                "datatype": "BYTES",
                "data": prompts               # List of N strings
            }
        ]
    }

    start_time = time.time()
    response = httpx.post(url, headers=headers, json=payload, timeout=180.0)
    end_time = time.time()

    try:
        outputs = json.loads(response.text)["outputs"][0]["data"]
    except Exception:
        outputs = [""] * len(prompts)

    total_time = end_time - start_time
    ttft = response.elapsed.total_seconds()  # Shared for all prompts

    results = []
    for idx, (prompt, output) in enumerate(zip(prompts, outputs)):
        output_tokens = count_tokens(output)
        prompt_tokens = count_tokens(prompt)
        tpot = total_time / max(output_tokens, 1)
        tokens_per_sec = output_tokens / max(total_time, 1e-6)

        results.append({
            "timestamp": pd.Timestamp.now(),
            "test_name": "batch",
            "prompt_index": idx,
            "model": llm,
            "prompt": prompt,
            "output_text": output[:200],
            "ttft": ttft,
            "tpot": tpot,
            "tokens_per_sec": tokens_per_sec,
            "latency": total_time,
            "num_tokens": output_tokens,
            "prompt_tokens": prompt_tokens,
            "status_code": response.status_code
        })

        print(f"\n[Prompt {idx}] → Tokens/sec: {tokens_per_sec:.1f} | TPOT: {tpot*1000:.2f} ms | Tokens: {output_tokens}")

    return pd.DataFrame(results)
