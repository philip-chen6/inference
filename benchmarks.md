# Inference benchmarks

- Model: Qwen/Qwen2.5-0.5B-Instruct
- GPU: RTX 5070
- Dtype: bf16
- Batch size: 1

| Method | Prompt tokens | Output tokens | TTFT s | Latency s | Decode TPS | Output throughput tok/s |
|---|---:|---:|---:|---:|---:|---:|
| Uncached | | 302 | 0.2245 | 5.0515 | 62.3574 | 59.7843 |
| KV cache | | 307 | 0.2242 | 4.3442 | 74.2721 | 70.6693 |