# eksperiment_LLM
Kod i materijali za diplomski rad Učinkovito izvođenje velikih jezičnih moodela u raspodijeljenim sustavima


# testiranje tritona

```bash
    curl -v -X POST \
        -H "Content-Type: application/json" \
        -d @payload.json \
        http://10.41.24.210:8000/v2/models/mistral/infer
```