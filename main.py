import random
import os
from application.helpers import get_token_counter
from application.load_tester import (send_batch_prompt_test, send_chat_like_test,
                                     send_summarize_document_test, send_prompt_request)
import pandas as pd
from dotenv import load_dotenv
from datasets import load_dataset

load_dotenv()

TRITON_URL = os.getenv("TRITON_URL")
LLM = ["gemma3", "mistral", "meta-llama"]

def get_model_infer_url(base_url, model_name):
    return f"{base_url}/v2/models/{model_name}/infer"


# Koristi Alpaca dataset sa HG
alpaca = load_dataset('tatsu-lab/alpaca', split='train')
alpaca_prompts = [
    ex["instruction"] + (f" {ex['input']}" if ex["input"] else "")
    for ex in alpaca
]


# Promptovi za testiranje, uzmi na random 20 iz Alpaca dataset-a
NUM_PROMPTS = 175
N_TURNS = 3
BATCH_SIZE = 8
NUM_BATCHES = 8
TOTAL_PROMPTS = BATCH_SIZE * NUM_BATCHES

stress_test_prompts = random.sample(alpaca_prompts, NUM_PROMPTS)
chat_test_prompts = random.sample(alpaca_prompts, N_TURNS)
batch_test_prompts = random.sample(alpaca_prompts, BATCH_SIZE)

# Dokument za sumiranje, neki tekst o povijesti nogometa
with open('sumiraj_me.txt', 'r') as f:
    document = f.read()

def main():
    single_results = []
    chat_results = []
    summarize_results = []
    batch_results = []

    for model_name in LLM:
        model_url = get_model_infer_url(TRITON_URL, model_name)
        # Obican prompt test
        print(f"\n=== Test slanja {NUM_PROMPTS} promptova: {model_name} ===")
        count_tokens = get_token_counter(model_name)
        for idx, prompt in enumerate(stress_test_prompts):
            metrics_df = send_prompt_request(model_name, model_url, prompt, count_tokens)
            metrics = metrics_df.iloc[0].to_dict()
            single_results.append(metrics)

        # Kao Chat test
        print(f"\n=== Test slanja konverzacija u {N_TURNS} interacija : {model_name} ===")
        for prompt in chat_test_prompts:
            chat_metrics = send_chat_like_test(model_name, model_url, prompt, count_tokens, n_turns=N_TURNS)
            chat_results.extend(chat_metrics.to_dict(orient="records"))

        # Sumiranje dokumenta test
        print(f"\n=== Test sumiranja dokumenta: {model_name} ===")
        summarize_df = send_summarize_document_test(model_name, model_url, document, count_tokens)
        summarize_results.extend(summarize_df.to_dict(orient="records"))

        print(f"\n=== Test batch prompta veličine {BATCH_SIZE * NUM_BATCHES}: {model_name} ===")
        batch_test_prompts = random.sample(alpaca_prompts, TOTAL_PROMPTS)
        for i in range(0, TOTAL_PROMPTS, BATCH_SIZE):
            batch = batch_test_prompts[i:i + BATCH_SIZE]
            batch_df = send_batch_prompt_test(model_name, model_url, batch, count_tokens)
            batch_results.extend(batch_df.to_dict(orient='records'))

    pd.DataFrame(single_results).to_csv("metrike/prompts_single/1_prompt_test_results.csv", index=False)
    pd.DataFrame(chat_results).to_csv("metrike/prompts_single/2_chat_test_results.csv", index=False)
    pd.DataFrame(summarize_results).to_csv("metrike/prompts_single/3_summarize_test_results.csv", index=False)
    pd.DataFrame(batch_results).to_csv("metrike/prompts_single/4_batch_test_results.csv", index=False)
    print("\nGotovo, rezultati su spremljeni u 'metrike/prompts_single/' folder.")

if __name__ == "__main__":
    main()