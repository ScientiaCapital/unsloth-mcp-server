---
name: model-deployment
description: Export and deploy fine-tuned models to production. Covers GGUF/Ollama, vLLM, HuggingFace Hub, Docker, quantization, and platform selection. Use after fine-tuning when you need to deploy models efficiently.
---

# Model Deployment

Complete guide for exporting, optimizing, and deploying fine-tuned LLMs to production environments.

## Overview

After fine-tuning your model with Unsloth, deploy it efficiently:

- **GGUF export** - For llama.cpp, Ollama, local inference
- **vLLM deployment** - For high-throughput production serving
- **HuggingFace Hub** - For sharing and version control
- **Quantization** - Reduce size while maintaining quality
- **Platform selection** - Choose the right infrastructure
- **Monitoring** - Track performance and costs

## Quick Start

### Export to GGUF (Ollama/llama.cpp)

```python
from unsloth import FastLanguageModel

# Load your fine-tuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    "./fine_tuned_model",
    max_seq_length=2048
)

# Export to GGUF format
model.save_pretrained_gguf(
    "./gguf_output",
    tokenizer,
    quantization_method="q4_k_m"  # 4-bit quantization
)

# Use with Ollama
# ollama create my-model -f ./gguf_output/Modelfile
# ollama run my-model
```

### Deploy with vLLM

```python
from unsloth import FastLanguageModel

# Save for vLLM
model.save_pretrained("./vllm_model")
tokenizer.save_pretrained("./vllm_model")

# Start vLLM server
# python -m vllm.entrypoints.openai.api_server \
#   --model ./vllm_model \
#   --tensor-parallel-size 1 \
#   --dtype bfloat16
```

### Push to HuggingFace Hub

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "./fine_tuned_model",
    max_seq_length=2048
)

# Push to Hub
model.push_to_hub(
    "your-username/model-name",
    token="hf_...",
    private=False
)
tokenizer.push_to_hub(
    "your-username/model-name",
    token="hf_..."
)
```

## Export Formats

### 1. GGUF (llama.cpp / Ollama)

**Best for:** Local deployment, edge devices, CPU inference

```python
# Export with different quantization levels
quantization_methods = {
    "q4_k_m": "4-bit, medium quality (recommended)",
    "q5_k_m": "5-bit, higher quality",
    "q8_0": "8-bit, near-original quality",
    "f16": "16-bit float, full quality",
    "f32": "32-bit float, highest quality"
}

# Export
model.save_pretrained_gguf(
    "./gguf_output",
    tokenizer,
    quantization_method="q4_k_m"
)

# Creates:
# - model-q4_k_m.gguf (quantized model)
# - Modelfile (for Ollama)
```

**Use with Ollama:**

```bash
# Create Ollama model
cd gguf_output
ollama create my-medical-model -f Modelfile

# Run
ollama run my-medical-model "What are the symptoms of pneumonia?"

# API server
ollama serve
# curl http://localhost:11434/api/generate -d '{"model": "my-medical-model", "prompt": "..."}'
```

**Use with llama.cpp:**

```bash
# Build llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Run inference
./main -m ../gguf_output/model-q4_k_m.gguf -p "Your prompt here"

# Server mode
./server -m ../gguf_output/model-q4_k_m.gguf --host 0.0.0.0 --port 8080
```

### 2. vLLM (Production Serving)

**Best for:** High-throughput production, API serving, multi-user

```python
# Prepare model for vLLM
model.save_pretrained("./vllm_model")
tokenizer.save_pretrained("./vllm_model")

# Optional: Merge LoRA weights into base model
model = FastLanguageModel.merge_and_unload(model)
model.save_pretrained("./vllm_model_merged")
```

**Deploy vLLM Server:**

```bash
# Single GPU
python -m vllm.entrypoints.openai.api_server \
  --model ./vllm_model \
  --dtype bfloat16 \
  --max-model-len 4096

# Multi-GPU (tensor parallelism)
python -m vllm.entrypoints.openai.api_server \
  --model ./vllm_model \
  --tensor-parallel-size 4 \
  --dtype bfloat16

# With quantization (AWQ)
python -m vllm.entrypoints.openai.api_server \
  --model ./vllm_model \
  --quantization awq \
  --dtype half
```

**Use vLLM API:**

```python
import openai

# Configure client
openai.api_key = "EMPTY"
openai.api_base = "http://localhost:8000/v1"

# Generate
response = openai.Completion.create(
    model="./vllm_model",
    prompt="Your prompt here",
    max_tokens=512,
    temperature=0.7
)

print(response.choices[0].text)
```

### 3. HuggingFace Hub

**Best for:** Sharing, version control, collaboration

```python
from unsloth import FastLanguageModel
from huggingface_hub import HfApi

# Load and push
model, tokenizer = FastLanguageModel.from_pretrained("./fine_tuned_model")

# Push to Hub
model.push_to_hub(
    "username/model-name",
    token="hf_...",
    private=True,  # or False for public
    commit_message="Initial upload of medical model"
)
tokenizer.push_to_hub("username/model-name", token="hf_...")

# Add model card
api = HfApi()
api.upload_file(
    path_or_fileobj="README.md",
    path_in_repo="README.md",
    repo_id="username/model-name",
    token="hf_..."
)
```

**Download from Hub:**

```python
from unsloth import FastLanguageModel

# Anyone can now load your model
model, tokenizer = FastLanguageModel.from_pretrained(
    "username/model-name",
    max_seq_length=2048
)
```

### 4. Docker Deployment

**Best for:** Reproducible deployments, cloud platforms

**Dockerfile for vLLM:**

```dockerfile
FROM vllm/vllm-openai:latest

# Copy model
COPY ./vllm_model /app/model

# Expose port
EXPOSE 8000

# Run server
CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "/app/model", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
```

**Build and run:**

```bash
# Build
docker build -t my-model-server .

# Run
docker run -d \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/vllm_model:/app/model \
  my-model-server

# Test
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "/app/model", "prompt": "Hello", "max_tokens": 50}'
```

## Quantization Strategies

### Quantization Methods Comparison

| Method | Size | Quality | Speed     | Use Case                  |
| ------ | ---- | ------- | --------- | ------------------------- |
| F32    | 100% | 100%    | Slow      | Baseline, not recommended |
| F16    | 50%  | ~100%   | Fast      | Full quality, GPU         |
| Q8_0   | 25%  | ~99%    | Faster    | Near-full quality         |
| Q5_K_M | 16%  | ~95%    | Very fast | Balanced                  |
| Q4_K_M | 12%  | ~90%    | Fastest   | Recommended default       |
| Q4_0   | 12%  | ~85%    | Fastest   | Low-end devices           |
| Q2_K   | 8%   | ~70%    | Fastest   | Edge devices only         |

### GGUF Quantization

```python
# Export multiple quantization levels
quantization_levels = ["q4_k_m", "q5_k_m", "q8_0"]

for quant in quantization_levels:
    model.save_pretrained_gguf(
        f"./gguf_output_{quant}",
        tokenizer,
        quantization_method=quant
    )
    print(f"Exported {quant}")

# Compare file sizes
# q4_k_m: ~4GB (7B model)
# q5_k_m: ~5GB
# q8_0: ~8GB
```

### GPTQ Quantization

**Best for:** GPU inference with high throughput

```python
from transformers import GPTQConfig

# Configure GPTQ
gptq_config = GPTQConfig(
    bits=4,
    dataset="c4",  # Calibration dataset
    tokenizer=tokenizer,
    group_size=128
)

# Quantize
quantized_model = model.quantize(gptq_config)

# Save
quantized_model.save_pretrained("./gptq_model")
tokenizer.save_pretrained("./gptq_model")
```

### AWQ Quantization

**Best for:** vLLM deployment

```python
from awq import AutoAWQForCausalLM

# Load model
model = AutoAWQForCausalLM.from_pretrained("./fine_tuned_model")

# Quantize
model.quantize(
    tokenizer,
    quant_config={
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4
    }
)

# Save
model.save_quantized("./awq_model")
tokenizer.save_pretrained("./awq_model")

# Use with vLLM
# python -m vllm.entrypoints.openai.api_server \
#   --model ./awq_model --quantization awq
```

## Deployment Platforms

### Local Deployment

**Pros:** Full control, no API costs, data privacy
**Cons:** Limited scale, hardware costs

```bash
# Ollama (easiest)
ollama create my-model -f Modelfile
ollama run my-model

# llama.cpp (most flexible)
./server -m model.gguf --host 0.0.0.0 --port 8080

# vLLM (best performance)
python -m vllm.entrypoints.openai.api_server --model ./model
```

**Hardware Requirements:**

| Model Size | Min RAM | Min VRAM | Recommended GPU |
| ---------- | ------- | -------- | --------------- |
| 1-3B       | 8GB     | 4GB      | RTX 3060        |
| 7B         | 16GB    | 8GB      | RTX 4070        |
| 13B        | 32GB    | 16GB     | RTX 4090        |
| 30B        | 64GB    | 24GB     | A5000           |
| 70B        | 128GB   | 48GB     | 2x A6000        |

### Cloud Platforms

#### Modal

**Best for:** Serverless, pay-per-use

```python
import modal

stub = modal.Stub("my-model")

@stub.function(
    image=modal.Image.debian_slim().pip_install("vllm"),
    gpu="A100",
    timeout=600
)
def generate(prompt: str) -> str:
    from vllm import LLM

    llm = LLM(model="./model")
    output = llm.generate(prompt)
    return output[0].outputs[0].text

# Deploy
# modal deploy app.py
```

**Pricing:** ~$1-3/hour A100, pay only for usage

#### RunPod

**Best for:** Persistent endpoints, GPU pods

```bash
# Deploy via RunPod UI or API
curl -X POST https://api.runpod.io/v2/endpoints \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -d '{
    "name": "my-model",
    "gpu_type": "RTX_4090",
    "docker_image": "vllm/vllm-openai:latest",
    "env": {
      "MODEL_NAME": "./model"
    }
  }'
```

**Pricing:** ~$0.30-0.50/hour RTX 4090, ~$1.50/hour A100

#### Vast.ai

**Best for:** Lowest cost, spot instances

```bash
# Search for instances
vastai search offers 'gpu_name=RTX_4090 num_gpus=1'

# Rent instance
vastai create instance <instance_id> \
  --image vllm/vllm-openai:latest \
  --env MODEL_NAME=./model
```

**Pricing:** ~$0.15-0.30/hour RTX 4090, ~$0.80/hour A100

#### AWS/GCP/Azure

**Best for:** Enterprise, compliance, scale

**AWS SageMaker:**

```python
from sagemaker.huggingface import HuggingFaceModel

# Create model
huggingface_model = HuggingFaceModel(
    model_data="s3://bucket/model.tar.gz",
    role=role,
    transformers_version="4.37",
    pytorch_version="2.1",
    py_version="py310"
)

# Deploy
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.xlarge"
)

# Generate
result = predictor.predict({
    "inputs": "Your prompt here"
})
```

**Pricing:** ~$1-5/hour depending on instance type

### Platform Comparison

| Platform | Setup  | Cost          | Scale   | Best For                   |
| -------- | ------ | ------------- | ------- | -------------------------- |
| Local    | Medium | Hardware only | Limited | Development, privacy       |
| Modal    | Easy   | Pay-per-use   | Auto    | Serverless, experiments    |
| RunPod   | Easy   | Low           | Manual  | Production, cost-sensitive |
| Vast.ai  | Medium | Lowest        | Manual  | Training, batch inference  |
| AWS/GCP  | Hard   | High          | Auto    | Enterprise, compliance     |

## Optimization Strategies

### 1. Merge LoRA Adapters

Before deployment, merge LoRA weights:

```python
from unsloth import FastLanguageModel

# Load with LoRA
model, tokenizer = FastLanguageModel.from_pretrained(
    "./fine_tuned_model",
    max_seq_length=2048
)

# Merge LoRA into base weights
model = FastLanguageModel.merge_and_unload(model)

# Save merged model (no LoRA overhead)
model.save_pretrained("./merged_model")
tokenizer.save_pretrained("./merged_model")
```

**Benefits:**

- Faster inference (no adapter computation)
- Simpler deployment (single model file)
- Broader compatibility

### 2. Enable Flash Attention

```python
# During model loading
model, tokenizer = FastLanguageModel.from_pretrained(
    "model-name",
    max_seq_length=2048,
    use_flash_attention_2=True  # 2-3x faster attention
)

# For vLLM deployment
# vLLM automatically uses flash attention if available
```

### 3. Batch Processing

For high throughput:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="./model")

# Batch prompts
prompts = [
    "Prompt 1",
    "Prompt 2",
    # ... up to 100s of prompts
]

# Generate in batch (much faster than sequential)
outputs = llm.generate(prompts, SamplingParams(temperature=0.7))

for output in outputs:
    print(output.outputs[0].text)
```

### 4. Continuous Batching

vLLM automatically does continuous batching:

```python
# Just configure for optimal throughput
python -m vllm.entrypoints.openai.api_server \
  --model ./model \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 256
```

## Load Testing & Benchmarking

### Benchmark Inference Speed

```python
import time
from vllm import LLM

llm = LLM(model="./model")

# Test prompts
prompts = ["Test prompt"] * 100

# Benchmark
start = time.time()
outputs = llm.generate(prompts)
end = time.time()

total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
tokens_per_sec = total_tokens / (end - start)

print(f"Throughput: {tokens_per_sec:.2f} tokens/sec")
```

### Load Testing with Locust

```python
from locust import HttpUser, task, between

class ModelUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def generate(self):
        self.client.post("/v1/completions", json={
            "model": "./model",
            "prompt": "What is the capital of France?",
            "max_tokens": 50
        })

# Run: locust -f loadtest.py --host http://localhost:8000
```

### Performance Targets

| Metric           | Target     | Excellent   | Notes               |
| ---------------- | ---------- | ----------- | ------------------- |
| Latency (TTFT)   | <500ms     | <200ms      | Time to first token |
| Throughput       | >50 tok/s  | >100 tok/s  | Per user            |
| P99 Latency      | <2s        | <1s         | 99th percentile     |
| Batch throughput | >500 tok/s | >1000 tok/s | Total system        |
| GPU utilization  | >70%       | >85%        | Resource efficiency |

## Monitoring & Observability

### Basic Monitoring

```python
import prometheus_client
from prometheus_client import Counter, Histogram

# Metrics
REQUEST_COUNT = Counter('model_requests_total', 'Total requests')
REQUEST_DURATION = Histogram('model_request_duration_seconds', 'Request duration')
TOKENS_GENERATED = Counter('model_tokens_generated_total', 'Total tokens')

# Instrument your endpoint
@REQUEST_DURATION.time()
def generate(prompt: str):
    REQUEST_COUNT.inc()
    output = model.generate(prompt)
    TOKENS_GENERATED.inc(len(output.token_ids))
    return output

# Expose metrics
prometheus_client.start_http_server(9090)
```

### vLLM Metrics

vLLM exposes metrics automatically:

```bash
curl http://localhost:8000/metrics

# Key metrics:
# - vllm:num_requests_running
# - vllm:num_requests_waiting
# - vllm:gpu_cache_usage_perc
# - vllm:time_to_first_token_seconds
# - vllm:time_per_output_token_seconds
```

### Cost Tracking

```python
class CostTracker:
    def __init__(self, cost_per_hour: float):
        self.cost_per_hour = cost_per_hour
        self.start_time = time.time()
        self.total_tokens = 0

    def track_generation(self, num_tokens: int):
        self.total_tokens += num_tokens

    def get_stats(self):
        hours = (time.time() - self.start_time) / 3600
        total_cost = hours * self.cost_per_hour
        cost_per_1k_tokens = (total_cost / self.total_tokens) * 1000

        return {
            'total_cost': total_cost,
            'total_tokens': self.total_tokens,
            'cost_per_1k_tokens': cost_per_1k_tokens,
            'tokens_per_dollar': self.total_tokens / total_cost
        }

# Usage
tracker = CostTracker(cost_per_hour=1.50)  # A100 pricing
tracker.track_generation(512)
print(tracker.get_stats())
```

## Common Deployment Patterns

### Pattern 1: Quick Local Demo

```bash
# Export to GGUF
python export_gguf.py

# Run with Ollama
ollama create my-demo -f Modelfile
ollama run my-demo

# Share demo
# Users just need: ollama pull username/my-demo
```

### Pattern 2: Production API

```bash
# Merge LoRA weights
python merge_lora.py

# Quantize with AWQ
python quantize_awq.py

# Deploy with vLLM
docker run -d --gpus all -p 8000:8000 \
  -v $(pwd)/model:/model \
  vllm/vllm-openai:latest \
  --model /model --quantization awq

# Load balancer + monitoring
# nginx -> vLLM instances -> Prometheus/Grafana
```

### Pattern 3: Multi-Model Serving

```python
from vllm import LLM

# Load multiple models
models = {
    'medical': LLM(model="./medical_model"),
    'legal': LLM(model="./legal_model"),
    'general': LLM(model="./general_model")
}

# Route based on input
def route_and_generate(text: str, domain: str):
    model = models.get(domain, models['general'])
    return model.generate(text)
```

### Pattern 4: Hybrid Deployment

```python
# Small model locally, large model in cloud
class HybridInference:
    def __init__(self):
        self.local = LLM(model="./small_model")  # 3B
        self.cloud_endpoint = "https://api.cloud.com/large-model"

    def generate(self, prompt: str, complexity: str = 'auto'):
        # Simple queries -> local
        # Complex queries -> cloud
        if complexity == 'auto':
            complexity = self.estimate_complexity(prompt)

        if complexity == 'simple':
            return self.local.generate(prompt)
        else:
            return requests.post(self.cloud_endpoint, json={'prompt': prompt})
```

## Troubleshooting

### Issue: Out of Memory (OOM)

**Solutions:**

```python
# 1. Use smaller quantization
model.save_pretrained_gguf("./output", tokenizer, quantization_method="q4_0")

# 2. Reduce max sequence length
python -m vllm.entrypoints.openai.api_server \
  --model ./model \
  --max-model-len 2048  # Instead of 4096

# 3. Enable CPU offloading
model, tokenizer = FastLanguageModel.from_pretrained(
    "./model",
    device_map="auto",  # Automatic CPU/GPU split
    offload_folder="./offload"
)

# 4. Use tensor parallelism (multi-GPU)
python -m vllm.entrypoints.openai.api_server \
  --model ./model \
  --tensor-parallel-size 2  # Split across 2 GPUs
```

### Issue: Slow Inference

**Solutions:**

```python
# 1. Enable flash attention
model, tokenizer = FastLanguageModel.from_pretrained(
    "./model",
    use_flash_attention_2=True
)

# 2. Use GPTQ/AWQ quantization (faster than GGUF on GPU)
# See quantization section above

# 3. Batch requests
# See batch processing section

# 4. Use vLLM instead of HuggingFace transformers
# vLLM is 10-20x faster for serving
```

### Issue: Model Quality Degradation

**Solutions:**

```python
# 1. Use higher quantization
# q4_k_m -> q5_k_m -> q8_0

# 2. Don't quantize twice
# If model is already quantized (e.g., bnb-4bit), export to f16 or f32

# 3. Test quantization quality
def test_quantization(original_model, quantized_model, test_prompts):
    results = []
    for prompt in test_prompts:
        orig_out = original_model.generate(prompt)
        quant_out = quantized_model.generate(prompt)
        similarity = calculate_similarity(orig_out, quant_out)
        results.append(similarity)
    return np.mean(results)

# Target: >90% similarity for production use
```

### Issue: High Latency

**Solutions:**

```bash
# 1. Use smaller model
# 7B instead of 13B often has similar quality with 2x lower latency

# 2. Reduce max_tokens
# Lower max_tokens = faster generation

# 3. Use local deployment
# Eliminates network latency

# 4. Optimize GPU settings
python -m vllm.entrypoints.openai.api_server \
  --model ./model \
  --gpu-memory-utilization 0.95 \
  --max-num-batched-tokens 8192
```

## Best Practices

### 1. Test Before Production

```python
# Always test quantized models
test_prompts = load_test_prompts()

original = LLM(model="./fine_tuned_model")
quantized = LLM(model="./quantized_model")

for prompt in test_prompts:
    orig_out = original.generate(prompt)
    quant_out = quantized.generate(prompt)

    # Compare quality
    print(f"Original: {orig_out}")
    print(f"Quantized: {quant_out}")
    print(f"Similarity: {calculate_similarity(orig_out, quant_out)}")
```

### 2. Version Your Models

```
models/
├── medical-v1.0.0/
│   ├── full/           # Full precision
│   ├── q4_k_m/        # 4-bit GGUF
│   ├── awq/           # AWQ quantized
│   └── README.md      # Model card
├── medical-v1.1.0/
└── production -> medical-v1.0.0/  # Symlink to deployed version
```

### 3. Monitor Everything

- Latency (P50, P95, P99)
- Throughput (tokens/sec)
- Error rate
- GPU utilization
- Cost per request
- Quality metrics (if available)

### 4. Start Small, Scale Up

```
1. Local testing (Ollama/llama.cpp)
2. Cloud trial (Modal/RunPod single instance)
3. Production (vLLM with load balancer)
4. Scale (Multi-GPU, multi-region)
```

### 5. Document Everything

Create a deployment README:

```markdown
# Model Deployment Guide

## Model Details

- Base: Llama-3.2-7B
- Fine-tuned on: Medical Q&A dataset
- Quantization: Q4_K_M
- Size: 4.2GB

## Deployment

ollama create medical-model -f Modelfile
ollama run medical-model

## Performance

- Latency: ~200ms (TTFT)
- Throughput: 50 tok/s
- Hardware: RTX 4070, 12GB VRAM

## Example Usage

...
```

## Cost Optimization

### Estimate Deployment Costs

```python
def estimate_monthly_cost(
    requests_per_day: int,
    avg_tokens_per_request: int,
    platform: str
):
    """
    Estimate monthly deployment costs
    """
    # Platform costs (per hour)
    costs = {
        'local_rtx4090': 0.20,  # Electricity + amortized hardware
        'vast_rtx4090': 0.25,
        'runpod_rtx4090': 0.40,
        'runpod_a100': 1.50,
        'modal_a100': 2.00,
        'aws_g5_xlarge': 1.20
    }

    hourly_cost = costs.get(platform, 1.0)

    # Estimate throughput
    tokens_per_sec = 50  # Conservative estimate
    seconds_per_request = avg_tokens_per_request / tokens_per_sec

    # Calculate usage
    daily_seconds = requests_per_day * seconds_per_request
    daily_hours = daily_seconds / 3600

    # For serverless, only count actual usage
    # For dedicated, count 24/7
    if platform.startswith('modal'):
        monthly_cost = daily_hours * 30 * hourly_cost
    else:
        monthly_cost = 24 * 30 * hourly_cost  # Always-on

    return {
        'monthly_cost': monthly_cost,
        'cost_per_request': monthly_cost / (requests_per_day * 30),
        'daily_hours': daily_hours
    }

# Example
cost = estimate_monthly_cost(
    requests_per_day=10000,
    avg_tokens_per_request=256,
    platform='runpod_rtx4090'
)
print(f"Monthly cost: ${cost['monthly_cost']:.2f}")
print(f"Per request: ${cost['cost_per_request']:.4f}")
```

### Cost Optimization Strategies

1. **Use spot instances** (Vast.ai) - 50-70% cheaper
2. **Scale down during off-peak** - 30-50% savings
3. **Batch requests** - Better GPU utilization
4. **Use smaller models** - 7B vs 13B often similar quality
5. **Aggressive quantization** - Q4 often sufficient
6. **Multi-tenancy** - Share GPU across models

## Additional Resources

- **Training**: See `unsloth-finetuning` skill for model training
- **Tokenizers**: See `superbpe` and `unsloth-tokenizer` skills
- **Optimization**: See `training-optimization` skill for training details
- **Datasets**: See `dataset-engineering` skill for data preparation

## Summary

Model deployment workflow:

1. ✓ Fine-tune with Unsloth
2. ✓ Merge LoRA adapters
3. ✓ Choose export format (GGUF/vLLM/HF)
4. ✓ Quantize appropriately (Q4_K_M recommended)
5. ✓ Select deployment platform
6. ✓ Deploy and monitor
7. ✓ Optimize costs and performance

Start with local Ollama deployment for testing, then scale to cloud for production.
