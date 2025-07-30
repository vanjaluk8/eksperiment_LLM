# NVIDIA Triton Inference Server - Performance Analyzer (perf_analyzer)

## Pokreće se putem docker  run komande
```bash
  docker run --rm -d --net=host -it nvcr.io/nvidia/tritonserver:24.05-py3-sdk
```


## Osnovni testovi pokretnuti iz konzole
### A. Baseline Latency/Throughput
```bash
  perf_analyzer -m mistral -u localhost:8000 --concurrency-range 1:1
  perf_analyzer -m gemma3 -u localhost:8000 --concurrency-range 1:1
  perf_analyzer -m meta-llama -u localhost:8000 --concurrency-range 1:1
```
### B. Scaling Concurrency
```bash
  perf_analyzer -m mistral -u localhost:8000 --concurrency-range 1:4
  perf_analyzer -m gemma3 -u localhost:8000 --concurrency-range 1:16:2
```
### C. Batch Size Scaling
```bash
  perf_analyzer -m meta-llama -u localhost:8000 --batch-size 4 --concurrency-range 1:4
```
### D. Vary Sequence Length / Prompt Size
```bash
  perf_analyzer -m gemma3 -u localhost:8000 --concurrency-range 1:4 --percentile 95
```

### Ispis testa u terminalu
```bash
   Request count: 478
    Throughput: 5.90113 infer/sec
    p50 latency: 693918210 usec
    p90 latency: 695533683 usec
    p95 latency: 724755898 usec
    p99 latency: 749308680 usec
    Avg HTTP time: 695116827 usec (send/recv 7820 usec + response wait 695109007 usec)
  Server:
    Inference count: 3824
    Execution count: 478
    Successful request count: 478
    Avg request latency: 695099710 usec (overhead 1 usec + queue 692389668 usec + compute input 18 usec + compute infer 2709978 usec + compute output 43 usec)
```