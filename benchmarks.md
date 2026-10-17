# Inference benchmarks

## Early single-request engine comparison

- Model: Qwen/Qwen2.5-0.5B-Instruct
- GPU: RTX 5070
- Dtype: bf16
- Batch size: 1

| Method | Prompt tokens | Output tokens | TTFT s | Latency s | Decode TPS | Output throughput tok/s |
|---|---:|---:|---:|---:|---:|---:|
| Uncached | | 302 | 0.2245 | 5.0515 | 62.3574 | 59.7843 |
| KV cache | | 307 | 0.2242 | 4.3442 | 74.2721 | 70.6693 |

## Controlled engine benchmark: 2026-10-17

- Model: Qwen/Qwen2.5-0.5B-Instruct; BF16; NVIDIA GeForce RTX 5070.
- Environment: PyTorch 2.14.1+cu130, Transformers 5.18.0, NVIDIA driver 610.88.
- Workload: the same 76-token chat prompt repeated within each batch; greedy decoding; 128 generated tokens per request.
- One 32-token warmup per configuration, followed by three measured runs per configuration. Configuration order was reversed on the second measured pass.
- Measurements below are medians. The existing loop timers include model execution and generation bookkeeping, but exclude tokenization, final text decoding, HTTP, and queue waiting.
- Batch sizes were passed directly to the decoding loop; this was not an HTTP server benchmark.

Prompt used for both the controlled engine and HTTP benchmarks:

> Write a detailed technical tutorial of at least 1000 words explaining how an autoregressive transformer serves language model requests. Cover tokenization, prefill, token decoding, key-value caching, batching, request scheduling, and latency versus throughput. Explain each topic with concrete examples.

| Method | Batch size | Generated tokens per run | Batch latency s | Engine TTFT s | Aggregate decode tok/s | Output throughput tok/s |
|---|---:|---:|---:|---:|---:|---:|
| Uncached | 1 | 128 | 2.6052 | 0.0224 | 49.2798 | 49.1321 |
| KV cache | 1 | 128 | 2.1455 | 0.0170 | 59.6176 | 59.6592 |
| KV cache | 3 | 384 | 2.0385 | 0.0145 | 188.3743 | 188.3707 |
| KV cache | 16 | 2048 | 1.9229 | 0.0283 | 1072.5051 | 1065.0578 |

At batch size one, KV caching improved median decode rate by 1.2098x and reduced time per decoded token from 20.2923 ms to 16.7736 ms, a 17.34% reduction. These values use decode rate after the first token, rather than total output throughput.

Raw engine measurements:

| Method | Batch size | Run | Batch latency s | Engine TTFT s | Aggregate decode tok/s | Output throughput tok/s |
|---|---:|---:|---:|---:|---:|---:|
| Uncached | 1 | 1 | 2.0348 | 0.0200 | 63.0330 | 62.9046 |
| Uncached | 1 | 2 | 3.0121 | 0.0224 | 42.4802 | 42.4958 |
| Uncached | 1 | 3 | 2.6052 | 0.0281 | 49.2798 | 49.1321 |
| KV cache | 1 | 1 | 1.6758 | 0.0170 | 76.5648 | 76.3835 |
| KV cache | 1 | 2 | 2.3087 | 0.0174 | 55.4271 | 55.4436 |
| KV cache | 1 | 3 | 2.1455 | 0.0153 | 59.6176 | 59.6592 |
| KV cache | 3 | 1 | 2.2195 | 0.0130 | 172.6649 | 173.0088 |
| KV cache | 3 | 2 | 1.8952 | 0.0145 | 202.5837 | 202.6130 |
| KV cache | 3 | 3 | 2.0385 | 0.0160 | 188.3743 | 188.3707 |
| KV cache | 16 | 1 | 1.9229 | 0.0283 | 1072.5051 | 1065.0578 |
| KV cache | 16 | 2 | 1.8367 | 0.0301 | 1124.7583 | 1115.0610 |
| KV cache | 16 | 3 | 2.2097 | 0.0275 | 931.1743 | 926.8177 |

## HTTP server benchmark: 2026-10-17

### Configuration and timing

- Same model, hardware, dtype, software environment, and prompt as the controlled engine benchmark above.
- Repository base revision: `1a353e1`, with the local worker batch limit changed from 3 to 16.
- Actual Uvicorn server on localhost, using one process, access logging disabled, and Uvicorn log level `warning`. Existing engine metrics and output printing remained enabled.
- Existing `POST /generate` endpoint, asyncio queue, background worker, and model generation code were exercised without source instrumentation.
- Maximum batch size: 16. Maximum batch collection window: 50 ms. Maximum generated tokens per request: 512.
- The client sent either one request or a burst of 16 simultaneous requests using a shared HTTP connection pool.
- Startup and one warmup wave at each concurrency were excluded. Five measured waves were run at each concurrency, alternating configuration order between passes.
- All 85 measured HTTP requests succeeded. Every request generated 512 token IDs. Every measured 16-request wave formed one actual batch of 16.
- Token counts came from the existing server's `Tokens generated` metrics, rather than re-tokenizing response strings. These counts include EOS if generated.
- Each wave was timed on the client from releasing its requests until all response bodies were received and parsed. This includes HTTP, queue collection/waiting, tokenization, generation, text decoding, and response transfer.

Aggregate output throughput for each wave:

```text
sum(generated token IDs across completed requests) / client wave wall time
```

### Results

| Concurrent requests | Observed batch size | Measured requests | Tokens per wave | Median output tok/s | Output tok/s range | Median wave latency s | Request latency p50 s |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 5 | 512 | 42.3838 | 41.6845-53.8526 | 12.0801 | 12.0801 |
| 16 | 16 | 80 | 8192 | 768.3404 | 672.6478-839.9506 | 10.6619 | 10.6574 |

Median aggregate HTTP output throughput increased by 18.1281x between concurrency one and concurrency 16. This is an aggregate throughput comparison, not an 18x improvement in individual-request token speed.

Raw measured waves, excluding warmups:

| Pass | Concurrency | Observed batch size | Generated tokens | Client wave wall time s | Aggregate output tok/s |
|---|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 512 | 12.080078 | 42.3838 |
| 1 | 16 | 16 | 8192 | 12.178736 | 672.6478 |
| 2 | 16 | 16 | 8192 | 11.772288 | 695.8715 |
| 2 | 1 | 1 | 512 | 12.282737 | 41.6845 |
| 3 | 1 | 1 | 512 | 12.248853 | 41.7998 |
| 3 | 16 | 16 | 8192 | 10.661941 | 768.3404 |
| 4 | 16 | 16 | 8192 | 9.752955 | 839.9506 |
| 4 | 1 | 1 | 512 | 9.507442 | 53.8526 |
| 5 | 1 | 1 | 512 | 9.697355 | 52.7979 |
| 5 | 16 | 16 | 8192 | 9.882147 | 828.9697 |

### Scope and resume wording

These are warmed localhost burst measurements using one repeated prompt. They are not sustained mixed-prompt traffic results or production latency guarantees. The endpoint returns complete strings, so this test reports completion latency rather than streaming client TTFT.

The HTTP workload generated 512 tokens per request, while the controlled engine benchmark generated 128. Differences between those tables cannot be attributed solely to HTTP or queue overhead.

Resume wording supported by this HTTP test:

> Built a FastAPI/Uvicorn inference server with asyncio request scheduling and static batching, scaling aggregate HTTP output throughput from 42 to 768 tokens/sec at batch size 16 on an RTX 5070 in localhost benchmarks.
