# Python
from application.prompts import PROMPTS
from application.string_prompter import collect_metrics, LLM, stress_test_prompts
import pandas as pd

def main():
    metrics_list = [
        collect_metrics(llm, prompt)
        for llm in LLM
        for prompt in PROMPTS[:-5]
    ] + [
        collect_metrics(llm, prompt, is_stress_test=True)
        for llm in LLM
        for prompt in stress_test_prompts
    ]

    df = pd.DataFrame(metrics_list)
    df.to_csv("metrike/llm_metrics.csv", index=False)
    print(df)

if __name__ == "__main__":
    main()