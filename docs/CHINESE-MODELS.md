# Chinese LLM Fine-Tuning Expertise

> **The premier fine-tuning service for Chinese open-source LLMs**

We specialize in fine-tuning the top 20 Chinese language models with Unsloth's 2x speed advantage and 80% VRAM reduction.

---

## Top 20 Supported Chinese Models

### Tier 1: Flagship Models (Production-Ready)

| #   | Model             | Developer   | Parameters        | Specialty       | VRAM (4-bit) | Status  |
| --- | ----------------- | ----------- | ----------------- | --------------- | ------------ | ------- |
| 1   | **DeepSeek-R1**   | DeepSeek    | 671B (37B active) | Reasoning, Math | MoE          | ✅ Full |
| 2   | **DeepSeek-V3.1** | DeepSeek    | 671B (37B active) | General, Coding | MoE          | ✅ Full |
| 3   | **Qwen3-235B**    | Alibaba     | 235B              | General Purpose | MoE          | ✅ Full |
| 4   | **Qwen3-72B**     | Alibaba     | 72B               | Reasoning       | 48GB         | ✅ Full |
| 5   | **Qwen3-32B**     | Alibaba     | 32B               | Balanced        | 24GB         | ✅ Full |
| 6   | **Kimi K2**       | Moonshot AI | 1T+               | Agents, Tools   | MoE          | ✅ Full |
| 7   | **GLM-4.5**       | Zhipu AI    | MoE               | Agentic, Tools  | MoE          | ✅ Full |

### Tier 2: Specialized Models

| #   | Model                 | Developer | Parameters | Specialty       | VRAM (4-bit) | Status  |
| --- | --------------------- | --------- | ---------- | --------------- | ------------ | ------- |
| 8   | **DeepSeek-Coder-V2** | DeepSeek  | 236B       | Coding          | MoE          | ✅ Full |
| 9   | **Qwen3-Coder-32B**   | Alibaba   | 32B        | Coding          | 24GB         | ✅ Full |
| 10  | **DeepSeek-OCR**      | DeepSeek  | Various    | Document OCR    | 8-16GB       | ✅ Full |
| 11  | **Qwen2.5-VL-72B**    | Alibaba   | 72B        | Vision-Language | 48GB         | ✅ Full |
| 12  | **Qwen2.5-VL-7B**     | Alibaba   | 7B         | Vision-Language | 8GB          | ✅ Full |
| 13  | **Yi-1.5-34B**        | 01.AI     | 34B        | Bilingual       | 24GB         | ✅ Full |
| 14  | **Yi-1.5-9B**         | 01.AI     | 9B         | Efficient       | 8GB          | ✅ Full |

### Tier 3: Efficient & Domain Models

| #   | Model                            | Developer | Parameters | Specialty             | VRAM (4-bit) | Status  |
| --- | -------------------------------- | --------- | ---------- | --------------------- | ------------ | ------- |
| 15  | **Baichuan-4**                   | Baichuan  | Various    | Chinese Culture, Law  | 16-24GB      | ✅ Full |
| 16  | **Baichuan-Omni**                | Baichuan  | Various    | Multimodal            | 24GB         | ✅ Full |
| 17  | **DeepSeek-R1-Distill-Qwen-32B** | DeepSeek  | 32B        | Reasoning (Distilled) | 24GB         | ✅ Full |
| 18  | **DeepSeek-R1-Distill-Qwen-14B** | DeepSeek  | 14B        | Reasoning (Distilled) | 12GB         | ✅ Full |
| 19  | **DeepSeek-R1-Distill-Qwen-7B**  | DeepSeek  | 7B         | Reasoning (Distilled) | 8GB          | ✅ Full |
| 20  | **Qwen3-4B**                     | Alibaba   | 4B         | Edge/Mobile           | 4GB          | ✅ Full |

---

## Why Chinese Models?

### Market Opportunity

| Metric                 | Value                         | Source            |
| ---------------------- | ----------------------------- | ----------------- |
| Chinese LLM downloads  | **#1 globally**               | Hugging Face 2025 |
| DeepSeek market share  | **32%** of enterprise AI      | Menlo Ventures    |
| Qwen monthly downloads | **#1 open model**             | NVIDIA GTC 2025   |
| Cost advantage         | **30-60x cheaper** than GPT-4 | Industry analysis |

### Real-World Adoption

- **Airbnb** uses Qwen for AI customer service (13 model ensemble)
- **Cursor** and **Cognition** build on DeepSeek infrastructure
- **Weibo** fine-tuned Qwen for $7,800 (vs $294K for R1 from scratch)

---

## Model Selection Guide

### By Use Case

| Use Case           | Recommended Models               | Why                          |
| ------------------ | -------------------------------- | ---------------------------- |
| **General Chat**   | Qwen3-32B, DeepSeek-V3.1         | Best quality/cost balance    |
| **Reasoning/Math** | DeepSeek-R1, Qwen3-72B           | SOTA reasoning benchmarks    |
| **Coding**         | DeepSeek-Coder-V2, Qwen3-Coder   | Top coding benchmarks        |
| **Chinese NLP**    | Baichuan-4, GLM-4.5              | Native Chinese optimization  |
| **Vision+Text**    | Qwen2.5-VL, Baichuan-Omni        | Multimodal capabilities      |
| **Agents/Tools**   | Kimi K2, GLM-4.5                 | Best function calling        |
| **Edge/Mobile**    | Qwen3-4B, DeepSeek-R1-Distill-7B | Fits consumer hardware       |
| **Document OCR**   | DeepSeek-OCR                     | 89%+ accuracy Chinese/Arabic |

### By Hardware

| GPU       | VRAM | Best Models                                    |
| --------- | ---- | ---------------------------------------------- |
| RTX 4090  | 24GB | Qwen3-32B, Yi-1.5-34B, DeepSeek-R1-Distill-32B |
| RTX 3090  | 24GB | Same as 4090                                   |
| A10       | 24GB | Same, optimized for cloud                      |
| A100 40GB | 40GB | Qwen3-72B, DeepSeek-V3 (quantized)             |
| A100 80GB | 80GB | Full Qwen3-72B, larger MoE models              |
| H100 80GB | 80GB | All models, fastest training                   |

### By Budget

| Budget | Model                  | Est. Training Cost (1K steps) |
| ------ | ---------------------- | ----------------------------- |
| < $5   | DeepSeek-R1-Distill-7B | $2-3                          |
| $5-20  | Qwen3-32B, Yi-1.5-34B  | $8-15                         |
| $20-50 | Qwen3-72B              | $25-40                        |
| $50+   | DeepSeek-R1 (full)     | $50-100+                      |

---

## Fine-Tuning Recipes

### DeepSeek-R1 Reasoning Enhancement

```python
# Best for: Math, logic, step-by-step thinking
model_config = {
    "base_model": "unsloth/DeepSeek-R1-Distill-Qwen-32B",
    "lora_rank": 64,
    "lora_alpha": 64,
    "learning_rate": 2e-5,
    "max_seq_length": 4096,
    "load_in_4bit": True,
}
```

### Qwen3 General Purpose

```python
# Best for: Customer service, content generation
model_config = {
    "base_model": "unsloth/Qwen3-32B-bnb-4bit",
    "lora_rank": 32,
    "lora_alpha": 32,
    "learning_rate": 1e-4,
    "max_seq_length": 2048,
    "load_in_4bit": True,
}
```

### Baichuan Chinese Domain

```python
# Best for: Chinese law, finance, medicine
model_config = {
    "base_model": "baichuan-inc/Baichuan2-13B-Chat",
    "lora_rank": 16,
    "lora_alpha": 32,
    "learning_rate": 5e-5,
    "max_seq_length": 4096,
    "load_in_4bit": True,
}
```

---

## Competitive Advantage

### Why Fine-Tune Chinese Models With Us?

| Feature                 | Us         | Together AI | Predibase | OpenAI |
| ----------------------- | ---------- | ----------- | --------- | ------ |
| Chinese model expertise | ✅ Top 20  | ✅ Some     | ✅ Some   | ❌     |
| DeepSeek-R1 support     | ✅ Full    | ✅          | ❌        | ❌     |
| Qwen3 MoE support       | ✅ Full    | ✅          | ✅        | ❌     |
| 2x faster training      | ✅ Unsloth | ❌          | ❌        | ❌     |
| 80% less VRAM           | ✅ Unsloth | ❌          | ❌        | ❌     |
| Knowledge pipeline      | ✅         | ❌          | ❌        | ❌     |
| Chinese OCR→Training    | ✅         | ❌          | ❌        | ❌     |
| Cost tracking           | ✅         | ❌          | Partial   | ❌     |

### Cost Comparison

| Task                  | GPT-4 Fine-tune | Our Platform (Qwen3-32B) | Savings |
| --------------------- | --------------- | ------------------------ | ------- |
| 10K training samples  | ~$250           | ~$40                     | **84%** |
| 100K training samples | ~$2,500         | ~$300                    | **88%** |
| Inference (1M tokens) | $15-75          | $0.50-2                  | **96%** |

---

## Getting Started

### Quick Start Commands

```bash
# List available Chinese models
unsloth-mcp list_supported_models --filter="chinese"

# Load DeepSeek-R1 for fine-tuning
unsloth-mcp load_model --model="unsloth/DeepSeek-R1-Distill-Qwen-32B"

# Fine-tune on your dataset
unsloth-mcp finetune_model \
  --model="unsloth/Qwen3-32B-bnb-4bit" \
  --dataset="your-dataset.json" \
  --format="alpaca"
```

### Recommended Workflow

```
1. Choose model from Top 20 list
2. Prepare dataset (Alpaca/ShareGPT/ChatML)
3. Set budget limit (cost tracking)
4. Start training with checkpoints
5. Export to GGUF/Ollama/vLLM
6. Track in Model Registry
```

---

## Resources

### Documentation

- [Unsloth Qwen3 Guide](https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune)
- [DeepSeek-R1 Fine-tuning](https://unsloth.ai/blog/deepseek-r1)
- [DeepSeek-OCR Guide](https://docs.unsloth.ai/models/deepseek-ocr-how-to-run-and-fine-tune)

### Model Catalogs

- [Unsloth Models](https://docs.unsloth.ai/get-started/all-our-models)
- [Unsloth HuggingFace](https://huggingface.co/unsloth)

### Research

- [Chinese Open-Source LLMs Overview](https://intuitionlabs.ai/articles/chinese-open-source-llms-2025)
- [Ranking Chinese Model Builders](https://www.interconnects.ai/p/chinas-top-19-open-model-labs)

---

_Last updated: 2025-12-31_
