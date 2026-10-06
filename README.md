# inference
learning stuff

## Setup

Install uv, then recreate the project environment on each computer:

```powershell
uv sync --locked
```

The project pins Python 3.13 and selects CUDA 13.0 PyTorch wheels on Windows
and Linux. GPU execution requires an NVIDIA GPU and a compatible driver.
Model weights are downloaded into each computer's Hugging Face cache on first use.
Commit `pyproject.toml`, `uv.lock`, and `.python-version`; recreate `.venv` locally.

Check the environment:

```powershell
uv run --locked python -c "import torch, fastapi, uvicorn; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Start the server:

```powershell
uv run --locked uvicorn server:app --host 127.0.0.1 --port 8000 --workers 1
```

The current `/generate` endpoint accepts `prompt` as a query parameter.

To run a command:

```powershell
$payload = @{ prompt = "Explain KV caching." } | ConvertTo-Json

Invoke-RestMethod `
    -Method Post `
    -Uri "http://127.0.0.1:8000/generate" `
    -ContentType "application/json" `
    -Body $payload
```

