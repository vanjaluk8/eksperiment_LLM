from locust import HttpUser, task, between
import random
from datasets import load_dataset
from application.string_prompter import (
    collect_metrics,
    run_chat_like_test,
    run_summarize_document_test,
    get_token_counter
)

# Load Alpaca prompts once at startup
alpaca = load_dataset('tatsu-lab/alpaca', split='train')
alpaca_prompts = [
    ex["instruction"] + (f" {ex['input']}" if ex["input"] else "")
    for ex in alpaca
]

class LLMUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        self.model_name = random.choice(["gemma3", "mistral", "meta-llama"])
        self.triton_url = "http://10.41.24.210:8000"
        self.count_tokens = get_token_counter(self.model_name)
        with open('sumiraj_me.txt', 'r') as f:
            self.document = f.read()

    @task
    def single_prompt_test(self):
        prompt = random.choice(alpaca_prompts)
        collect_metrics(self.model_name, self.triton_url, prompt, True, self.count_tokens)

    @task
    def chat_like_test(self):
        prompt = random.choice(alpaca_prompts)
        run_chat_like_test(self.model_name, self.triton_url, prompt, self.count_tokens, n_turns=5)

    @task
    def summarize_document_test(self):
        run_summarize_document_test(self.model_name, self.triton_url, self.document, self.count_tokens)