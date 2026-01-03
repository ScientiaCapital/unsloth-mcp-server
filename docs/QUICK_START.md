# Quick Start Guide

## Your Learning Path (Baby Steps!)

```
┌─────────────────────────────────────────────────────────────────┐
│                    THE CORRECT LEARNING PATH                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  CRAWL (Week 1):  SFT on 100 examples → IT WORKS!               │
│      ↓                                                          │
│  WALK (Week 2):   SFT with more data, better prompts            │
│      ↓                                                          │
│  RUN (Week 3+):   ONLY NOW try GRPO on your working model       │
│                                                                  │
│  ⚠️  NEVER skip straight to GRPO - it won't work!               │
└─────────────────────────────────────────────────────────────────┘
```

## Choose Your Server

### Option A: Minimal 6-Tool Server (Recommended for Learning)
```bash
npm run start:minimal
```

**Tools included:**
1. `check_installation` - Verify setup
2. `list_supported_models` - See available models
3. `load_model` - Load a model
4. `finetune_model` - SFT training
5. `generate_text` - Test your model
6. `export_model` - Export to GGUF

### Option B: Full 34-Tool Server
```bash
npm run start
```

Includes all tools: RunPod, tokenizers, knowledge capture, GRPO, etc.

---

## Step 1: Check Your Setup

```bash
# In Claude Code, use the check_installation tool
# It will verify Python, Unsloth, and GPU availability
```

## Step 2: Prepare Your Data

Your training data should be in ChatML format:
```json
{
  "messages": [
    {"role": "user", "content": "How do I handle 'your price is too high'?"},
    {"role": "assistant", "content": "Great question! Here's how..."}
  ]
}
```

Use the prep script:
```bash
python src/scripts/prep_sales_data.py
```

## Step 3: Train (SFT First!)

Using Claude Code's finetune_model tool:
```json
{
  "tool": "finetune_model",
  "arguments": {
    "model_name": "unsloth/Qwen2.5-7B-Instruct",
    "dataset_path": "data/objection_training.jsonl",
    "output_dir": "models/adapters/objection_handler_v1",
    "num_epochs": 1,
    "batch_size": 2,
    "learning_rate": 2e-4
  }
}
```

## Step 4: Test Your Model

```json
{
  "tool": "generate_text",
  "arguments": {
    "prompt": "Customer says: 'I need to think about it.' How should I respond?",
    "max_tokens": 256
  }
}
```

## Step 5: Export for Production

```json
{
  "tool": "export_model",
  "arguments": {
    "output_format": "gguf",
    "quantization": "q4_k_m",
    "output_path": "models/exports/objection-handler.gguf"
  }
}
```

## Step 6: Deploy to Ollama

```bash
cd models/exports
ollama create objection-handler -f Modelfile
```

## Step 7: Use via curl!

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "objection-handler",
  "prompt": "Handle this objection: We already have a vendor.",
  "stream": false
}'
```

---

## Project Structure

```
unsloth-mcp-server/
├── src/
│   ├── minimal-server.ts     # 6-tool learning server
│   ├── index.ts              # Full 34-tool server
│   ├── scripts/              # Python scripts (extracted)
│   │   ├── list_models.py
│   │   ├── load_model.py
│   │   ├── finetune_model.py
│   │   ├── generate_text.py
│   │   ├── export_model.py
│   │   └── ...
│   └── tools/                # Tool modules
│       ├── core.ts           # 6 core tools
│       ├── training.ts       # Model info + GRPO
│       ├── tokenizer.ts      # SuperBPE
│       └── runpod.ts         # GPU cloud
├── models/
│   ├── adapters/             # Your trained LoRA adapters
│   ├── exports/              # GGUF files for Ollama
│   └── configs/grpo/         # GRPO training configs
├── examples/baby-steps/      # Learning scripts
└── docs/
    ├── QUICK_START.md        # This file
    ├── MAC_SETUP.md          # Mac hardware guide
    └── USING_YOUR_MODELS.md  # Deployment guide
```

---

## Mac Hardware Guide

| Mac | RAM | Best Models |
|-----|-----|-------------|
| M2 16GB | Work | Qwen2.5-7B, DeepSeek-R1-7B |
| M1 8GB | Personal | Qwen2.5-0.5B, Qwen2.5-1.5B |

See `docs/MAC_SETUP.md` for details.

---

## Common Issues

### "CUDA not available"
- Expected on Mac! Use Metal acceleration or train on Colab/RunPod

### "Out of memory"
- Reduce batch_size to 1
- Use smaller model (0.5B instead of 7B)
- Enable 4-bit quantization

### "Model generates gibberish"
- Train longer (more epochs)
- Check your training data format
- Make sure you're using ChatML format

---

## Next Steps

1. **Run examples/baby-steps/** - Start with the simplest scripts
2. **Train on 100 examples** - Don't try to scale too fast
3. **Test via generate_text** - Verify your model works
4. **Export to Ollama** - Make it portable
5. **Only then: GRPO** - Once SFT works, optimize preferences

Happy fine-tuning! 🚀
