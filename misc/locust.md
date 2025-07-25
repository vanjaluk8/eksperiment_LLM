# random testiranje na tri modela za sva 4 scenarija
# concurrent users 250, workera 4, trajanje 1h
 ```bash
locust -f locustfile.py \
      --host=http://localhost:8000 \
      --users=100 \
      --spawn-rate=25 \
      --run-time=1h 
      --csv=metrike/locust/locust_results --html=metrike/locust/locust_results.html --loglevel=INFO 
```
Kod pokretanja locusta radili su samo mistral i meta-llama modeli, dok gemma3 nije uspio pratiti opterećenje.



Type     Name                                                                          # reqs      # fails |    Avg     Min     Max    Med |   req/s  failures/s
--------|----------------------------------------------------------------------------|-------|-------------|-------|-------|-------|-------|--------|-----------
POST     meta-llama/infer                                                                 504     0(0.00%) |  43917   10185  108798  32000 |    0.24        0.00
POST     mistral/infer                                                                    511     0(0.00%) |  32995    5565  100458  23000 |    0.24        0.00
--------|----------------------------------------------------------------------------|-------|-------------|-------|-------|-------|-------|--------|-----------
         Aggregated                                                                      1015     0(0.00%) |  38419    5565  108798  27000 |    0.48        0.00

Response time percentiles (approximated)
Type     Name                                                                                  50%    66%    75%    80%    90%    95%    98%    99%  99.9% 99.99%   100% # reqs
--------|--------------------------------------------------------------------------------|--------|------|------|------|------|------|------|------|------|------|------|------
POST     meta-llama/infer                                                                    32000  55000  67000  72000  92000  97000 101000 105000 109000 109000 109000    504
POST     mistral/infer                                                                       23000  35000  45000  54000  68000  89000  93000  95000 100000 100000 100000    511
--------|--------------------------------------------------------------------------------|--------|------|------|------|------|------|------|------|------|------|------|------
         Aggregated                                                                          27000  40000  56000  63000  85000  94000  98000 101000 108000 109000 109000   1015

