# Učinkovito izvođenje velikih jezičnih modela u raspodijeljenim sustavima - popratni materijali uz diplomski rad

## O repozitoriju i svrha
U ovom repozitoriju nalaze se skripte i materijali za testiranje i benchmarking velikih jezičnih modela (LLM-ova) koristeći *NVIDIA Triton Inference Server*. Ovaj repozitorij sadrži sav materijal korišten u parktičnom dijelu diplomskog rada, uključujući skripte za izvođenje eksperimenata, testiranje performansi, metrike i analizu rezultata.

Svrha eksperimenta je dokazati da veliki jezični modeli u raspodijeljenjim sustavima mogu postizati visoke performanse uz pravilno konfigurirnje postavki služenja modela i optimizaciju infrastrukture.

## Sadržaj

### Triton i perf_analyzer
- U misc i triton folderima se nalaze popratni materijali koji su korišteni u testiranju i podizanju triton i perf_analyzer instanci:
- `Dockerfile` i `docker-compose.yml` datoteke pokretanje Triton servera
- `config.pbtxt` protobuf konfiguracijska datoteka za Triton server, koja definira modele i njihove postavke
- `perf_analyzer.md` datoteka s osnovnim uputama za korištenje `perf_analyzer` alata
- `test_script.sh` bash skripta za pokretanje testova performansi na Triton serveru, uključujući slanje zahtjeva i prikupljanje rezult
- dijagram arhitekture sustava i tijek rada eksperimenta, koji vizualizira interakciju između komponenti i protok podataka

Niže se nalazi osnovni primjer `curl` naredbi za slanje inference zahtjeva, te bash skripta za pokretanje automatiziranih testova.

```bash
    curl -v -X POST \
        -H "Content-Type: application/json" \
        -d @payload.json \
        http://10.41.24.210:8000/v2/models/mistral/infer
```

### Eksperimenti i benchmarkiranje LLM-ova
- Python skripte u application datoteci služe za izvođenje eksperimenata i benchmarkinga s LLM-ovima sa korisničke strane. Skripte su dizajnirane za interakciju s Triton serverom i izvođenje različitih testova performansi sa osnovnim funkcionalnostima.
- locustfile.py sadrži skriptu za pokretanje Locust testova, koji simuliraju korisničke zahtjeve prema Triton serveru i mjere performanse.

## Tipičan tijek rada

1. Priprema testnih zahtjeva i slanje prema Triton serveru koristeći priloženu bash skriptu
2. Prikupljanje rezultata i osnovnih mejernih podataka
3. Analiza rezultata i mjernih podataka, uključujući:
   - sa serverske strane:
     - Inferences/Second
     - p99 Latency
     - p95 Latency
     - p90 Latency
     - p50 Latency
     - računanje benchmarka `troughput to latency ratio`
   - s korisničke strane (koristeći smoke test skripte, te locust za stres testove):
     - TIME TO FIRST TOKEN (TTFT)
     - TIME PER OUTPUT TOKEN (TPOT)
     - TOKEN PER SECOND (TPS)
     - LATENCY (latency)
   
4. Vizualizacija i analiza mjernih podataka radi usporedbe različitih modela i vrsta testiranja koristeći jupyter notebook

## Korištene tehnologije

- Python, hugging face alati i pytorch biblioteke za rad s velikim jezičnim modelima
- Docker i Docker Compose za upravljanje kontejnerima i pokretanje Triton server
- Nvidia Triton Inference Server i perf_analyzer alat
- Biblioteke za analizu podataka (`pandas`, `numpy`, `seaborn`, `matplotlib`)
- Shell skripting za interakciju s API-jem
