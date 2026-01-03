# Mac Setup Guide - Model Recommendations

This project is shared between two Macs via GitHub. Use different models based on your RAM.

## Hardware Configurations

| Mac | Chip | RAM | Use Case |
|-----|------|-----|----------|
| **Work MacBook** | M2 | 16GB | Training 7B models, production inference |
| **Personal MacBook** | M1 | 8GB | Learning, small model experiments |

## Model Recommendations by RAM

### 8GB RAM (M1 MacBook)
Use **small models** for learning:

```python
# Baby steps - always works
"unsloth/Qwen2.5-0.5B-Instruct"  # 0.5B params, ~2GB

# Slightly larger - good for learning
"unsloth/Qwen2.5-1.5B-Instruct"  # 1.5B params, ~3GB

# Maximum for 8GB
"unsloth/Qwen2.5-3B-Instruct"    # 3B params, ~5GB (tight fit!)
```

**Training settings for 8GB:**
```python
TrainingArguments(
    per_device_train_batch_size=1,      # Keep at 1!
    gradient_accumulation_steps=8,       # Simulates batch=8
    max_seq_length=512,                  # Shorter sequences
    fp16=False,                          # Let PyTorch decide
)
```

### 16GB RAM (M2 Work MacBook)
Can run **7B models** with 4-bit quantization:

```python
# Your production models
"unsloth/DeepSeek-R1-Distill-Qwen-7B"  # 7B, reasoning-focused, ~8GB
"unsloth/Qwen2.5-7B-Instruct"          # 7B, general purpose, ~8GB

# Also good
"unsloth/Llama-3.1-8B-bnb-4bit"        # 8B, popular choice
"unsloth/Mistral-7B-Instruct-v0.3-bnb-4bit"
```

**Training settings for 16GB:**
```python
TrainingArguments(
    per_device_train_batch_size=1,      # Start here
    gradient_accumulation_steps=4,       # Simulates batch=4
    max_seq_length=2048,                 # Full context
    optim="adamw_8bit",                  # Memory-efficient
)
```

## Quick Reference

```
8GB M1:  Qwen2.5-0.5B → Qwen2.5-1.5B → (Qwen2.5-3B if careful)
16GB M2: Qwen2.5-3B → DeepSeek-R1-7B / Qwen2.5-7B
```

## Git Workflow (Sharing Between Macs)

```bash
# On M2 (Work Mac) - after training
git add models/adapters/
git commit -m "feat: Add objection handler LoRA adapters"
git push origin main

# On M1 (Personal Mac) - to use the trained model
git pull origin main
# Models are in models/adapters/
```

**Note:** Large model files (GGUF, safetensors) are gitignored.
Only LoRA adapters (~50-200MB) are committed.

## Memory Tips

1. **Close other apps** before training
2. **Use 4-bit quantization** always: `load_in_4bit=True`
3. **Gradient checkpointing** saves memory: `use_gradient_checkpointing="unsloth"`
4. **Shorter sequences** = less memory: `max_seq_length=512`
5. **Batch size 1** with gradient accumulation

## Testing Your Setup

```bash
# Run on your Mac to check available memory
python -c "
import torch
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('MPS (Metal) available!')
    print(f'Device: {torch.backends.mps.is_built()}')
else:
    print('MPS not available - will use CPU')
"
```

## Learning Path

| Week | M1 (8GB) | M2 (16GB) |
|------|----------|-----------|
| 1 | Qwen2.5-0.5B + 20 math examples | Qwen2.5-0.5B + 20 math examples |
| 2 | Qwen2.5-1.5B + 100 objections | Qwen2.5-7B + 100 objections |
| 3 | Test & iterate | DeepSeek-R1-7B + full dataset |
| 4+ | Inference only | GRPO (if SFT works!) |
