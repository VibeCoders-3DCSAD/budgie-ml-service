# Extraction validation — 2026-10-21

## Behavior and packaging

- 96 tests passed, including inherited classification/rule/forecast/optimization tests, signed request ownership checks, real forecasts from all 12 category artifacts, checksum rejection, readiness, request limits, and responsive health checks during forecasting.
- Ruff lint and format checks passed; mypy passed for all 24 application source files; `bash -n start.sh` passed.
- For the same synthetic inputs and forecast date (2026-10-21), original Odin-ML and the extracted service produced identical JSON for all 12 category forecast points, classification results, and budget recommendations. Response IDs, execution timing, and generated-at timestamps were excluded by comparing the business results directly.
- A clean Python 3.14.4 environment loaded every original SARIMA artifact. PyArrow was discovered as a serialization dependency and is included in the runtime lock.
- Docker image built successfully. All three primary inference endpoints returned HTTP 200/SUCCESS in the running container, with the original HMAC contract.
- Container ran as UID 10001, under a 1536 MiB memory cap, one CPU, and a 128-process limit, without an OOM kill. Runtime image contains no PyTorch, scikit-learn, training package, or pytest.
- The upstream Starlette TestClient emits one deprecation warning about its httpx integration. The tests still pass; this warning does not affect the production server.

## Local process memory

The production `start.sh` launcher was profiled on the development Linux host. RSS was sampled from `/proc/<pid>/status` every 10 ms; VmHWM supplied the kernel-maintained peak. Readiness polling determines startup duration, so timing includes polling overhead. These measurements cover the Uvicorn/Python process, not total host memory.

| Measurement | Result |
| --- | ---: |
| Startup through readiness | 1.463 seconds |
| Startup memory high-water mark | 248.5 MiB |
| RSS immediately after readiness | 248.5 MiB |
| Peak high-water mark after requests | 321.1 MiB |
| RSS at end | 305.8 MiB |
| Monitoring duration | 2.439 seconds |
| Samples | 236 |

Workload: eight signed forecasts, each with 10,000 synthetic transactions and all 12 categories, submitted with client concurrency four; then five signed recommendations with 100 categories. The shared limiter serializes numerical computation. The service remained healthy and was stopped after profiling.

A separate short container smoke test reported 182.3 MiB of container memory after representative requests. Container accounting and host-process RSS are different measures, and its workload was smaller; do not compare those values directly.

This is a short functional and memory smoke test, not a sustained-load, leak, CPU-capacity, or latency benchmark. A dedicated 2 GB target is supported by these results, but production headroom depends on the OS, other processes, and actual traffic. No target-machine deployment has been performed.
