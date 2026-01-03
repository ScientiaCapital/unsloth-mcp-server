# Sharing Models with Your Team via HuggingFace

## Quick Setup

### 1. Get Your HuggingFace Token

1. Go to https://huggingface.co/settings/tokens
2. Create a new token with **write** access
3. Save it:

```bash
# Add to your shell profile (~/.zshrc or ~/.bashrc)
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
```

Or login interactively:
```bash
pip install huggingface_hub
huggingface-cli login
```

---

## Three Ways to Share Models

### Option 1: LoRA Adapter (Recommended)

**Size:** ~35MB for a 7B model
**Best for:** Team members who have Unsloth installed

```bash
python src/scripts/push_to_hub.py \
  --adapter models/adapters/objection_handler_v1 \
  --repo ScientiaCapital/objection-handler-lora
```

**Your team uses it:**
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="ScientiaCapital/objection-handler-lora",
    load_in_4bit=True
)
```

---

### Option 2: Merged Model (Full Standalone)

**Size:** ~14GB for a 7B model
**Best for:** Team members who just want to use transformers

```bash
python src/scripts/push_to_hub.py \
  --adapter models/adapters/objection_handler_v1 \
  --repo ScientiaCapital/objection-handler-full \
  --merge
```

**Your team uses it:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("ScientiaCapital/objection-handler-full")
tokenizer = AutoTokenizer.from_pretrained("ScientiaCapital/objection-handler-full")
```

---

### Option 3: GGUF for Ollama (Easiest for Non-Technical Users)

**Size:** ~4-8GB depending on quantization
**Best for:** Anyone with Ollama installed (no Python needed!)

```bash
# First export to GGUF
# (use export_model tool or export script)

# Then push to HuggingFace
python src/scripts/push_to_hub.py \
  --gguf models/exports/objection-handler-q4_k_m.gguf \
  --repo ScientiaCapital/objection-handler-gguf
```

**Your team uses it:**
```bash
# One command - that's it!
ollama run hf.co/ScientiaCapital/objection-handler-gguf
```

---

## Recommended Workflow for Coperniq

```
┌─────────────────────────────────────────────────────────────────┐
│               COPERNIQ MODEL SHARING WORKFLOW                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  YOU (Tim):                                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 1. Train model locally or on RunPod/Colab               │    │
│  │ 2. Test with generate_text tool                          │    │
│  │ 3. Push to HuggingFace:                                  │    │
│  │    • LoRA for engineers (small, flexible)                │    │
│  │    • GGUF for sales team (Ollama, easy)                  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                      │
│                           ▼                                      │
│  HUGGINGFACE (ScientiaCapital org):                             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ ScientiaCapital/objection-handler-lora  (35MB)          │    │
│  │ ScientiaCapital/objection-handler-gguf  (4GB)           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                      │
│                           ▼                                      │
│  ENGINEERS:                          SALES TEAM:                 │
│  ┌────────────────────┐              ┌────────────────────┐     │
│  │ from unsloth import│              │ ollama run         │     │
│  │ FastLanguageModel  │              │ hf.co/Scientia...  │     │
│  │                    │              │                    │     │
│  │ # Fine-tune more   │              │ # Just ask         │     │
│  │ # or integrate     │              │ # questions!       │     │
│  └────────────────────┘              └────────────────────┘     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Private vs Public Models

### Make it Private (Team Only)

```bash
python src/scripts/push_to_hub.py \
  --adapter models/adapters/objection_handler_v1 \
  --repo ScientiaCapital/objection-handler-lora \
  --private
```

Then add team members to your HuggingFace organization:
1. Go to https://huggingface.co/ScientiaCapital
2. Settings → Members → Invite

### Make it Public (Open Source)

Just omit the `--private` flag. Anyone can download and use your model.

---

## Version Control Your Models

Use naming conventions to track versions:

```bash
# Version 1 - basic objection handling
ScientiaCapital/objection-handler-v1-lora

# Version 2 - added more training data
ScientiaCapital/objection-handler-v2-lora

# Version 3 - after GRPO optimization
ScientiaCapital/objection-handler-v3-grpo-lora
```

---

## Complete Example: Train → Share → Use

### Step 1: Train (You)
```bash
# Train on your 100 objection handling examples
python src/scripts/train_objection_handler.py
# Output: models/adapters/objection_handler_v1/
```

### Step 2: Test (You)
```bash
# Verify it works
python src/scripts/test_objection_handler.py
```

### Step 3: Export GGUF (You)
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/DeepSeek-R1-Distill-Qwen-7B",
    load_in_4bit=True
)
model.load_adapter("models/adapters/objection_handler_v1")

model.save_pretrained_gguf(
    "models/exports/objection-handler",
    tokenizer,
    quantization_method="q4_k_m"
)
```

### Step 4: Push to Hub (You)
```bash
# Push LoRA for engineers
python src/scripts/push_to_hub.py \
  --adapter models/adapters/objection_handler_v1 \
  --repo ScientiaCapital/objection-handler-lora \
  --private

# Push GGUF for sales team
python src/scripts/push_to_hub.py \
  --gguf models/exports/objection-handler-q4_k_m.gguf \
  --repo ScientiaCapital/objection-handler-gguf \
  --private
```

### Step 5: Share with Team (You)
```
Hey team! New objection handler model is ready:

ENGINEERS:
pip install unsloth
python -c "
from unsloth import FastLanguageModel
model, tok = FastLanguageModel.from_pretrained('ScientiaCapital/objection-handler-lora')
"

SALES TEAM (needs Ollama installed):
ollama run hf.co/ScientiaCapital/objection-handler-gguf

Try asking: "How do I handle 'your price is too high'?"
```

### Step 6: Team Uses It

**Engineer (Python):**
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "ScientiaCapital/objection-handler-lora",
    load_in_4bit=True
)

# Generate response
FastLanguageModel.for_inference(model)
inputs = tokenizer("Handle: We need to think about it.", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))
```

**Sales Person (Ollama):**
```bash
ollama run hf.co/ScientiaCapital/objection-handler-gguf

>>> How should I respond to "your price is too high"?
```

---

## Troubleshooting

### "Repository not found"
- Check if repo name is correct
- Make sure you have write access to the organization

### "Authentication failed"
```bash
# Re-login
huggingface-cli login

# Or check your token
echo $HF_TOKEN
```

### "File too large"
- HuggingFace has a 50GB limit per file
- For large merged models, use Git LFS (automatic)
- For GGUF, split if needed:
  ```bash
  split -b 10G model.gguf model.gguf.
  ```

---

## Summary

| Format | Size | Best For | Command |
|--------|------|----------|---------|
| LoRA | ~35MB | Engineers | `--adapter path --repo name` |
| Merged | ~14GB | Transformers users | `--adapter path --repo name --merge` |
| GGUF | ~4-8GB | Ollama/sales team | `--gguf path --repo name` |

Start with LoRA + GGUF. That covers both your engineers and sales team!
