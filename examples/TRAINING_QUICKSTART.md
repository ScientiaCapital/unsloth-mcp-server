# Coperniq SFT Training - Quick Start Guide

## 🎯 What You'll Get
After training, you'll have a sales agent AI that handles:
- Objection handling (20 examples from real Gong calls)
- Competitor positioning (ServiceTitan, Procore, BuildOps, etc.)
- MEP-specific scenarios (HVAC, Plumbing, Electrical, Solar)
- Discovery and closing techniques

## 📊 Training Data
- **83 examples** in ChatML format
- Sources: GRPO prompts, battle cards, competitive analysis
- Categories: objection_handling, competitor, closing, discovery, mep_specific

## 🚀 Option 1: Google Colab (FREE - Recommended)

### Step 1: Open Colab
Go to https://colab.research.google.com

### Step 2: Upload Files
Upload these two files:
- `train_coperniq_sft.py`
- `sft_training_data.jsonl`

### Step 3: Run Training
```python
# In a new cell
!python train_coperniq_sft.py
```

### Step 4: Download Your Model
After training completes (~10 minutes on T4):
- Download `coperniq-sales-sft/` folder
- Or push directly to HuggingFace with your token

## 🚀 Option 2: RunPod

### Step 1: Create Pod
- Template: PyTorch 2.1 + CUDA 12.1
- GPU: Any (RTX 3090, A100, etc.)
- Disk: 20GB

### Step 2: Upload & Run
```bash
# In terminal
wget [your training data URL]
python train_coperniq_sft.py
```

## ⚙️ Configuration

Edit `train_coperniq_sft.py` to customize:

```python
# Model - start small, scale up later
MODEL_NAME = "unsloth/Qwen2.5-0.5B-Instruct"  # 0.5B (T4-friendly)
# MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct"   # 3B (needs 16GB)
# MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct"   # 7B (needs 24GB)

# Training
MAX_STEPS = 60   # Increase for more epochs
BATCH_SIZE = 2   # Decrease if OOM
LEARNING_RATE = 2e-4  # Lower if training unstable
```

## 📁 Output Structure

After training:
```
coperniq-sales-sft/
├── adapter_config.json    # LoRA config
├── adapter_model.safetensors  # LoRA weights (~35MB)
├── merged_16bit/         # Full merged model (~1GB)
│   ├── model.safetensors
│   └── tokenizer files
├── gguf_q8_0/           # Ollama-ready
│   ├── coperniq-sales-q8_0.gguf
│   └── Modelfile
└── training_report.json  # Stats
```

## 🧪 Test Your Model

### With Ollama (local)
```bash
cd coperniq-sales-sft/gguf_q8_0/
ollama create coperniq-sales -f Modelfile
ollama run coperniq-sales
```

### With Python
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "./coperniq-sales-sft"
)
FastLanguageModel.for_inference(model)

# Generate
messages = [{"role": "user", "content": "How do I handle price objections?"}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
outputs = model.generate(inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))
```

## 🔄 Next Steps After SFT

### 1. Test Thoroughly
- Ask 20+ sales questions
- Check for formatting issues
- Verify domain knowledge

### 2. Add More Training Data
- Extract real emails from Close CRM (run `extract_close_emails.py`)
- Add more battle card scenarios
- Re-run training with `--data combined_data.jsonl`

### 3. GRPO Training (Preference Optimization)
```bash
python train_grpo.py --sft-adapter ./coperniq-sales-sft
```

### 4. Share with Team
```bash
export HF_TOKEN=your_token
python push_to_hub.py --model ./coperniq-sales-sft --repo ScientiaCapital/coperniq-sales-sft
```

## ❓ Troubleshooting

### Out of Memory (OOM)
```python
BATCH_SIZE = 1  # Reduce batch size
MAX_SEQ_LENGTH = 1024  # Reduce context length
```

### Slow Training
- Check GPU is detected: `torch.cuda.is_available()`
- Use T4 or better GPU
- Enable xformers: `pip install xformers`

### Bad Outputs
- Check training loss is decreasing
- Increase `MAX_STEPS` to 100+
- Add more diverse training examples

## 📚 Resources
- Unsloth docs: https://docs.unsloth.ai
- Training data generator: `data/generate_sft_dataset.py`
- GRPO training: `models/configs/grpo/train_grpo.py`
