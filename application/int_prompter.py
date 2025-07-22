import json
import os
import time

import httpx
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from transformers import AutoTokenizer

# ===========================
# 🔧 Postavke
# ===========================
load_dotenv()

TRITON_URL = os.getenv("TRITON_URL")
LLM = ["rt-mistral"]
MAX_LEN = 8
VOCAB_SIZE = 32768  # iz model_metadata

PROMPTS = [
    "Discuss the 4-3-3 formation in football. Please answer in one sentence.",
    "Explain the offside rule. Please answer in one sentece.",
    "Who won the 2018 World Cup? Please answer in one word.",
]

# ===========================
# 📦 Tokenizer
# ===========================
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer.pad_token = tokenizer.eos_token

# ===========================
# 📤 Payload za Triton
# ===========================
def build_payload(prompt):
    tokens = tokenizer(
        prompt,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="np"
    )
    input_ids = tokens["input_ids"].astype(np.int64)  # Oblik: (1, MAX_LEN)

    print("📥 Tokenized input_ids:", input_ids)
    print("Shape:", input_ids.shape)

    payload = {
        "inputs": [
            {
                "name": "input_ids",
                "shape": list(input_ids.shape),  # [1, 8]
                "datatype": "INT64",
                "data": input_ids.tolist()
            }
        ],
        "outputs": [
            {
                "name": "logits",
                "parameters": {"binary_data": True}
            }
        ]
    }
    return payload

# ===========================
# 📨 Slanje zahtjeva + mjerenje performansi
# ===========================
def send_int_request(url, prompt):
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/octet-stream"
    }
    payload = build_payload(prompt)

    start_time = time.time()
    response = httpx.post(url, headers=headers, json=payload, timeout=60.0)
    ttft = time.time() - start_time  # ⏱️ Time to First Token

    if response.status_code != 200:
        print("❌ Greška:", response.status_code, response.content.decode(errors="ignore"))
        return "", 0, ttft, 0

    # 📦 Parsiranje binarnog odgovora (FP16)
    try:
        logits_buffer = np.frombuffer(response.content, dtype=np.float16)
        total_logits = logits_buffer.size
        expected_per_token = VOCAB_SIZE
        batch_size = 1  # uvijek slanje jednog prompta

        # 👇 Automatski izračun stvarnog seq_len
        if total_logits % (batch_size * expected_per_token) != 0:
            print("⚠️ Upozorenje: logits veličina nije djeljiva s vokabularom. Ostatak:", total_logits % VOCAB_SIZE)

        seq_len = total_logits // (batch_size * expected_per_token)

        # 👇 Oblikuj logits u [batch, seq_len, vocab_size]
        logits = logits_buffer[:batch_size * seq_len * expected_per_token].reshape(batch_size, seq_len, expected_per_token)
        pred_token_ids = np.argmax(logits, axis=-1)[0]  # uzimamo batch=0
        num_tokens = len(pred_token_ids)

        # 🧠 Dekodirani tekst
        decoded_text = tokenizer.decode(pred_token_ids, skip_special_tokens=True)

        # 📊 Metrike
        tpot = ttft / num_tokens if num_tokens > 0 else 0
        tokens_per_sec = num_tokens / ttft if ttft > 0 else 0

        print(f"✅ TTFT: {ttft:.2f}s | TPOT: {tpot:.4f}s/token | Token/s: {tokens_per_sec:.2f}")
        print(f"📝 Odgovor: {decoded_text}")

        return decoded_text, num_tokens, ttft, tpot

    except Exception as e:
        print("❌ Greška kod parsiranja binarnog odgovora:", str(e))
        return "", 0, ttft, 0

# ===========================
# 🔁 Evaluacija
# ===========================
metrics_list = []

for prompt in PROMPTS:
    for llm in LLM:
        llm_url = f"{TRITON_URL}/v2/models/{llm}/infer"
        print(f"\n🚀 Model: {llm} | Prompt: {prompt}")
        try:
            output_text, num_tokens, ttft, tpot = send_int_request(llm_url, prompt)
            metrics_list.append({
                "model": llm,
                "prompt": prompt,
                "output_text": output_text,
                "num_tokens": num_tokens,
                "ttft": round(ttft, 4),
                "tpot": round(tpot, 4),
                "tokens_per_sec": round(num_tokens / ttft if ttft > 0 else 0, 2)
            })
        except Exception as e:
            print(f"❌ Pogreška za model '{llm}' prompt='{prompt}': {e}")

# ===========================
# 💾 Spremi rezultate
# ===========================
df = pd.DataFrame(metrics_list)
df.to_csv("llm_int_metrics_fp16.csv", index=False)

print("\n📊 Rezultat testiranja:")
print(df)
