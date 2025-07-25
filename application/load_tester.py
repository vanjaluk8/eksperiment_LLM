import json
import time
import httpx
import pandas as pd

# metrike koje se mjere
# - TTFT: Total Time For Test (ukupno vrijeme za testiranje)
# - TPOT: Time Per Output Token (vrijeme po tokenu izlaza)
# - Tokens Per Second (broj tokena po sekundi)
# - Latency (latencija)
# - Throughput/TPS (broj tokena po sekundi)
# - Prompt Tokens (broj tokena u promptu)
# - Output Tokens (broj tokena u izlazu)

def send_request(url, payload):
    ''' Salje HTTP POST na URL sa payloadom i vraca odgovor i latenciju '''
    headers = {"Content-Type": "application/json"}
    start_time = time.time()
    response = httpx.post(url, headers=headers, json=payload, timeout=180.0)
    end_time = time.time()
    latency = end_time - start_time
    return response, latency


def send_prompt_request(model, url, prompt, count_tokens):
    ''' Salje obican prompt na model i vraca metrike (koriste ju i druge funkcije) '''
    from application.helpers import build_metrics
    payload = {
        "inputs": [{
            "name": "text_input",
            "shape": [1, 1],
            "datatype": "BYTES",
            "data": [prompt]
        }]
    }
    response, latency = send_request(url, payload)
    try:
        output_text = json.loads(response.text)["outputs"][0]["data"][0]
    except Exception:
        output_text = ""
    metrics = build_metrics("single", model, prompt, output_text, 0, response, latency, count_tokens)
    print(f"\nPrompt: {prompt[:60]}...| Output: {output_text[:100]}... "
          f"\n| TTFT: {metrics['ttft']:.3f}s | TPOT: {metrics['tpot']*1000:.3f} ms/tok "
          f"\n| Throughput: {metrics['tokens_per_sec']:.2f} tok/s"
          f"\n| Tokens: {metrics['output_tokens']} | Latency: {metrics['latency']:.3f}s"
          f"\n| Status: {response.status_code}")

    return pd.DataFrame([metrics])

def send_chat_like_test(model, url, initial_prompt, count_tokens, n_turns=5):
    ''' Simulacija chat interakcije sa modelom'''
    chat_history = ""
    results = []
    for t in range(n_turns):
        if t == 0:
            full_prompt = f"User: {initial_prompt}\nAssistant:"
        else:
            user_input = f"Turn {t}: OK, elaborate more."
            full_prompt = chat_history + f"\nUser: {user_input}\nAssistant:"
        df = send_prompt_request(model, url, full_prompt, count_tokens)
        metrics = df.iloc[0].to_dict()
        metrics["test_name"] = "chat"
        metrics["prompt_index"] = t
        results.append(metrics)
        if t == 0:
            chat_history = f"User: {initial_prompt}\nAssistant: {metrics['output_text']}"
        else:
            chat_history += f"\nUser: Turn {t}: Tell me something new.\nAssistant: {metrics['output_text']}"
    return pd.DataFrame(results)

def send_summarize_document_test(model, url, document, count_tokens):
    ''' Salje dokument za sumiranje i vraca metrike '''
    prompt = f"Summarize the following document in 1500 words:\n\n{document}\n\nSummary:"
    df = send_prompt_request(model, url, prompt, count_tokens)
    metrics = df.iloc[0].to_dict()
    metrics["test_name"] = "summarizer"
    metrics["prompt_index"] = 0
    return pd.DataFrame([metrics])

def send_batch_prompt_test(model, url, prompts, count_tokens):
    ''' Salje batch promptova (8 komada) na model i vraca metrike '''
    from application.helpers import build_metrics
    payload = {
        "inputs": [{
            "name": "text_input",
            "shape": [len(prompts), 1],
            "datatype": "BYTES",
            "data": prompts
        }]
    }
    response, latency = send_request(url, payload)
    try:
        outputs = json.loads(response.text)["outputs"][0]["data"]
    except Exception:
        outputs = [""] * len(prompts)

    results = []
    for idx, (prompt, output) in enumerate(zip(prompts, outputs)):
        metrics = build_metrics("batch", model, prompt, output, idx, response, latency, count_tokens)
        results.append(metrics)
        print(f"[Prompt {idx}]"
          f"| TTFT: {metrics['ttft']:.3f}s | TPOT: {metrics['tpot']*1000:.3f} ms/tok "
          f"| Throughput: {metrics['tokens_per_sec']:.2f} tok/s"
          f"| Tokens: {metrics['output_tokens']} | Latency: {metrics['latency']:.3f}s"
          f"| Status: {response.status_code}")
    return pd.DataFrame(results)


