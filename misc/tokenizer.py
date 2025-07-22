from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
prompt = "On which stadium plays NK Istra 1961? Tell me only the name of the stadium."
input_ids = tokenizer(prompt, return_tensors="np")["input_ids"].tolist()
print(input_ids)