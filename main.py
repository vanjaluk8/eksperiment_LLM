import random
import os
from application.string_prompter import get_token_counter, run_chat_like_test, run_summarize_document_test
from application.string_prompter import collect_metrics
from application.string_prompter import run_batch_prompt_test
import pandas as pd
from dotenv import load_dotenv
from datasets import load_dataset

load_dotenv()


TRITON_URL = os.getenv("TRITON_URL")
LLM = ["gemma3", "mistral", "meta-llama"]

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
        # Obican prompt test
        print(f"\n=== Running stress test for model: {model_name} ===")
        count_tokens = get_token_counter(model_name)
        for idx, prompt in enumerate(stress_test_prompts):
            metrics = collect_metrics(model_name, TRITON_URL, prompt, True, count_tokens)
            single_results.append(metrics)

        # Kao Chat test
        print(f"\n=== Running chat-like test for model: {model_name} ===")
        for prompt in chat_test_prompts:
            chat_metrics = run_chat_like_test(model_name, TRITON_URL, prompt, count_tokens, n_turns=N_TURNS)
            chat_results.extend(chat_metrics.to_dict(orient="records"))

        # Sumiranje dokumenta test
        print(f"\n=== Running document summarization for model: {model_name} ===")
        summarize_df = run_summarize_document_test(model_name, TRITON_URL, document, count_tokens)
        summarize_results.extend(summarize_df.to_dict(orient="records"))

        print(f"\n=== Running batch prompt test for model: {model_name} ===")
        batch_test_prompts = random.sample(alpaca_prompts, TOTAL_PROMPTS)
        for i in range(0, TOTAL_PROMPTS, BATCH_SIZE):
            batch = batch_test_prompts[i:i + BATCH_SIZE]
            batch_df = run_batch_prompt_test(model_name, f"{TRITON_URL}/v2/models/{model_name}/infer", batch, count_tokens)
            batch_results.extend(batch_df.to_dict(orient='records'))

    #print("Metrics results:", chat_results)
    pd.DataFrame(single_results).to_csv("metrike/prompts/llm_stress_test_results.csv", index=False)
    pd.DataFrame(chat_results).to_csv("metrike/prompts/llm_chat_test_results.csv", index=False)
    pd.DataFrame(summarize_results).to_csv("metrike/prompts/llm_summarize_test_results.csv", index=False)
    pd.DataFrame(batch_results).to_csv("metrike/prompts/llm_batch_test_results.csv", index=False)
    print("\nGotovo, rezultati su spremljeni u 'metrike/prompts' folder.")

if __name__ == "__main__":
    main()