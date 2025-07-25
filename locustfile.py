import pandas as pd
import os
from locust import HttpUser, task, between
from urllib.parse import urlparse
from urllib import response

from application.load_tester import send_batch_prompt_test, send_summarize_document_test, send_chat_like_test, send_prompt_request
from main import (
    get_model_infer_url, get_token_counter, LLM, TRITON_URL,
    stress_test_prompts, chat_test_prompts, batch_test_prompts,
    N_TURNS, BATCH_SIZE, NUM_BATCHES, TOTAL_PROMPTS
)
import random
import json


class ModelUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        self.model_name = random.choice(LLM)
        self.model_url = get_model_infer_url(TRITON_URL, self.model_name)
        self.count_tokens = get_token_counter(self.model_name)
        parsed = urlparse(self.model_url)
        self.endpoint_path = parsed.path
        print(f"Endpoint path: {self.endpoint_path}")

    @task
    def prompt_test(self):
        prompt = random.choice(stress_test_prompts)
        payload = {
            "inputs": [{
                "name": "text_input",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [prompt]
            }]
        }
        headers = {"Content-Type": "application/json"}
        self.client.post(
            self.endpoint_path,
            data=json.dumps(payload),
            headers=headers,
            name=f"{self.model_name}/infer"
        )

        metrics_df = send_prompt_request(self.model_name, self.model_url, prompt, self.count_tokens)
        metrics = metrics_df.iloc[0].to_dict()
        print(metrics)  # Or write to CSV, etc.
        file_exists = os.path.isfile("metrike/prompts/1_prompt_test_results.csv")
        df = pd.DataFrame([metrics])
        df.to_csv("metrike/prompts/1_prompt_test_results.csv", mode="a", index=False, header=not file_exists)


    @task
    def summarize_test(self):
        with open('sumiraj_me.txt', 'r', encoding='utf-8') as f:
            document = [line.strip() for line in f if line.strip()]

        prompt = f"Summarize this document in 1500 words:\n\n{document}\n\nSummary:"
        payload = {
            "inputs": [{
                "name": "text_input",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [prompt]
            }]
        }
        headers = {"Content-Type": "application/json"}
        self.client.post(
            self.endpoint_path,
            data=json.dumps(payload),
            headers=headers,
            name=f"{self.model_name}/infer"
        )

        metrics_df = send_summarize_document_test(self.model_name, self.model_url, prompt, self.count_tokens)
        metrics = metrics_df.iloc[0].to_dict()
        #print(metrics)  # Or write to CSV, etc.
        file_exists = os.path.isfile("metrike/prompts/3_summarize_test_results.csv")
        df = pd.DataFrame([metrics])
        df.to_csv("metrike/prompts/3_summarize_test_results.csv", mode="a", index=False, header=not file_exists)

    @task
    def batch_test(self):
        prompt = random.sample(batch_test_prompts, 8)
        payload = {
            "inputs": [{
                "name": "text_input",
                "shape": [8, 1],
                "datatype": "BYTES",
                "data": [prompt]
            }]
        }
        headers = {"Content-Type": "application/json"}
        self.client.post(
            self.endpoint_path,
            data=json.dumps(payload),
            headers=headers,
            name=f"{self.model_name}/infer"
        )


        metrics_df = send_batch_prompt_test(self.model_name, self.model_url, prompt, self.count_tokens)
        file_exists = os.path.isfile("metrike/prompts/4_batch_test_results.csv")
        metrics_df.to_csv("metrike/prompts/4_batch_test_results.csv", mode="a", index=False, header=not file_exists)

    @task
    def chat_test(self):
        initial_prompt = random.choice(chat_test_prompts)
        metrics_df = send_chat_like_test(self.model_name, self.model_url, initial_prompt, self.count_tokens, n_turns=N_TURNS)
        agg_metrics = {
            "timestamp": pd.Timestamp.now(),
            "test_name": "chat",

            "prompt_index": N_TURNS,
            "model": self.model_name,
            "ttft": metrics_df["ttft"].sum(),
            "tpot": metrics_df["tpot"].mean(),
            "tokens_per_sec": metrics_df["tokens_per_sec"].mean(),
            "latency": metrics_df["latency"].sum() if "latency" in metrics_df else None,
            "prompt_tokens": metrics_df["prompt_tokens"].sum() if "tokens" in metrics_df else None,
            "output_tokens": metrics_df["output_tokens"].sum(),
            "output_text": " ".join(metrics_df["output_text"].tolist()),
            "status_code": metrics_df["status_code"].mode()[0] if "status_code" in metrics_df else None
        }
        file_exists = os.path.isfile("metrike/prompts/2_chat_test_results.csv")
        pd.DataFrame([agg_metrics]).to_csv(
            "metrike/prompts/2_chat_test_results.csv",
            mode="a", index=False, header=not file_exists
        )

