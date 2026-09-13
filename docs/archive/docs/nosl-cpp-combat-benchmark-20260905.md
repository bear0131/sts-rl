# Native NOSL combat benchmark (2026-09-05)

The Python/PyTorch NOSL run was stopped with `SIGINT` before this benchmark.
The saved checkpoint remains at:

```text
/root/autodl-tmp/sts-rl/runs/nosl-is-az-inner-s128d3-group4-long/20260905-010608/interrupted.zip
```

The C++ executable uses the same simulator and the extracted NOSL
redeterminization, but does not use the neural model. It uses uniform/online
tabular policy and HP-ratio leaf values, so these numbers are execution
benchmarks rather than model-quality comparisons.

| machine/config | episodes | decisions/s | simulations/s | native transitions/s | elapsed |
| --- | ---: | ---: | ---: | ---: | ---: |
| local, 1 thread, 128x3 | 20 | 3,687 | 471,904 | 1.39M | 0.162 s |
| local, 4 threads, 128x3 | 80 | 13,776 | 1.76M | 5.19M | 0.167 s |
| remote, 1 thread, 128x3 | 500 | 4,696 | 601,113 | 1.77M | 3.086 s |
| remote, 25 threads, 128x3 | 500 | 31,378 | 4.02M | 11.83M | 0.462 s |
| remote, 1 thread, training updates | 500 | 4,572 | 585,199 | 1.72M | 3.044 s |

The 25-thread result is about 6.7x faster than the remote one-thread result,
not 25x, which indicates CPU quota/placement contention. The tabular updates
reduce throughput by only about 3% in this run. `--threads N` uses independent
tables per worker; dumps are written as `PATH.worker-N` for a multi-worker run.
