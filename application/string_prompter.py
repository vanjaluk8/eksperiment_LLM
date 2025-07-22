import json
import os
import time

import httpx
import pandas as pd
from dotenv import load_dotenv
from transformers import AutoTokenizer
from application.prompts import PROMPTS

load_dotenv()

TRITON_URL = os.getenv("TRITON_URL")
LLM = ["gemma3", "mistral", "meta-llama"]

LLM_URL = f"{TRITON_URL}/v2/models/{LLM[1]}/infer"
stress_test_prompts = PROMPTS[-5:]

tokenizer = AutoTokenizer.from_pretrained("gpt2")

def collect_metrics(llm, prompt, is_stress_test=False):
    result = send_infer_request(f"{TRITON_URL}/v2/models/{llm}/infer", prompt)
    return {
        "timestamp": pd.Timestamp.now(),
        "model": llm,
        "prompt": f"STRESS_TEST: {prompt}" if is_stress_test else prompt,
        "ttft": result["ttft"],
        "tpot": result["tpot"],
        "tpt": result["tpt"],
        "latency": result["latency"],
        "num_tokens": result["num_tokens"],
        "output_text": result["output_text"],
        "status_code": result["response"].status_code,
    }

def count_tokens(text):
    return len(tokenizer.encode(text))

def send_infer_request(url, prompt):
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
    response = httpx.post(url, headers=headers, json=payload, timeout=120.0)
    end_time = time.time()
    total_time = end_time - start_time
    try:
        output_text = json.loads(response.text)["outputs"][0]["data"][0]
    except Exception:
        output_text = ""

    num_tokens = count_tokens(output_text)
    prompt_len = count_tokens(prompt)
    ttft = response.elapsed.total_seconds()
    generation_time = total_time - ttft if total_time > ttft else 0.0001

    tpot = num_tokens / generation_time if generation_time > 0 else 0
    tpt = generation_time / num_tokens if num_tokens > 0 else 0

    metrics = f"TTFT: {ttft:.2f}s, TPOT: {tpot:.2f} tok/s, TPT: {tpt:.4f} s/tok"
    print(f"Model URL: {url}")
    print(f"Prompt (truncated): {prompt[:60]}{'...' if len(prompt)>60 else ''}")
    print(f"Output (truncated): {output_text[:100]}{'...' if len(output_text)>100 else ''}")
    print(metrics)
    print(f"Prompt tokens: {prompt_len}, Output tokens: {num_tokens}")
    print("-" * 40)

    return {
        "response": response,
        "output_text": output_text,
        "num_tokens": num_tokens,
        "prompt_len": prompt_len,
        "ttft": ttft,
        "tpot": tpot,
        "tpt": tpt,
        "latency": total_time,
    }


