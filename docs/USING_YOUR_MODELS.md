# Using Your Trained Models

## Two Deployment Paths

### Path 1: MCP Server (For Claude Code)

The MCP server lets Claude Code interact with your models through the 6 tools.

```
┌─────────────────────────────────────────────────────────────┐
│  Claude Code                                                 │
│     │                                                        │
│     ▼                                                        │
│  MCP Server (minimal-server.ts)                              │
│     │                                                        │
│     ├── load_model ──────▶ Load your trained adapter         │
│     ├── generate_text ───▶ Ask questions                     │
│     └── export_model ────▶ Save as GGUF                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Example MCP Tool Calls (via Claude Code):**

```json
// 1. Load your trained model
{
  "tool": "load_model",
  "arguments": {
    "model_name": "unsloth/DeepSeek-R1-Distill-Qwen-7B",
    "adapter_path": "models/adapters/my_objection_model",
    "load_in_4bit": true
  }
}

// 2. Ask a question
{
  "tool": "generate_text",
  "arguments": {
    "prompt": "Customer says: 'Your price is too high compared to competitors.' How should I respond?",
    "max_tokens": 256
  }
}

// 3. Export for Ollama
{
  "tool": "export_model",
  "arguments": {
    "output_format": "gguf",
    "quantization": "q4_k_m",
    "output_path": "models/exports/objection-handler-q4.gguf"
  }
}
```

---

### Path 2: Ollama (For curl/API/Apps)

After exporting to GGUF, you can serve your model via Ollama for direct API access.

```
┌─────────────────────────────────────────────────────────────┐
│  Your App / curl / Python                                    │
│     │                                                        │
│     ▼                                                        │
│  Ollama Server (http://localhost:11434)                      │
│     │                                                        │
│     └── Your GGUF Model                                      │
│         models/exports/objection-handler-q4.gguf             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Step-by-Step: From Training to curl

### Step 1: Train Your Model (Colab/RunPod)

```bash
# Run your training script
python src/scripts/train_objection_handler.py

# Output: models/adapters/objection_handler_v1/
#   ├── adapter_model.safetensors
#   ├── adapter_config.json
#   └── tokenizer files...
```

### Step 2: Export to GGUF

```python
# Using Unsloth's export function
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/DeepSeek-R1-Distill-Qwen-7B",
    adapter_path="models/adapters/objection_handler_v1",
    load_in_4bit=True
)

# Export to GGUF (quantized for CPU/Apple Silicon)
model.save_pretrained_gguf(
    "models/exports/objection-handler",
    tokenizer,
    quantization_method="q4_k_m"  # Good balance of size/quality
)
# Creates: models/exports/objection-handler-q4_k_m.gguf
```

### Step 3: Create Ollama Modelfile

```bash
# Create Modelfile for your custom model
cat > models/exports/Modelfile << 'EOF'
FROM ./objection-handler-q4_k_m.gguf

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""

SYSTEM """You are an expert sales representative trained on successful objection handling.
Respond with empathy, address concerns directly, and guide toward value."""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"
EOF
```

### Step 4: Register with Ollama

```bash
cd models/exports

# Create the model in Ollama
ollama create objection-handler -f Modelfile

# Verify it's registered
ollama list
# NAME                    SIZE
# objection-handler       4.1 GB
```

### Step 5: Use via curl!

```bash
# Simple generation
curl http://localhost:11434/api/generate -d '{
  "model": "objection-handler",
  "prompt": "Customer says: Your price is too high. How should I respond?",
  "stream": false
}'

# With system prompt
curl http://localhost:11434/api/generate -d '{
  "model": "objection-handler",
  "system": "You are helping a solar sales rep handle objections.",
  "prompt": "They say they need to think about it. What do I say?",
  "stream": false
}'

# Chat format (multi-turn)
curl http://localhost:11434/api/chat -d '{
  "model": "objection-handler",
  "messages": [
    {"role": "system", "content": "You are a sales objection expert."},
    {"role": "user", "content": "Customer: We already have a vendor. How do I respond?"}
  ],
  "stream": false
}'
```

---

## Python Integration

```python
import requests

def ask_objection_handler(question: str) -> str:
    """Query your trained objection handler model."""
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "objection-handler",
            "prompt": question,
            "stream": False
        }
    )
    return response.json()["response"]

# Example usage
answer = ask_objection_handler(
    "Customer says: 'I need to discuss this with my partner.' "
    "What's the best response?"
)
print(answer)
```

---

## Quick Reference: Model Locations

```
unsloth-mcp-server/
├── models/
│   ├── adapters/                    # Your trained LoRA adapters
│   │   ├── my_crm_model/            # From your Colab notebook
│   │   └── objection_handler_v1/    # Your sales objection model
│   │
│   ├── exports/                     # GGUF files for Ollama
│   │   ├── objection-handler-q4_k_m.gguf
│   │   └── Modelfile
│   │
│   └── configs/
│       └── grpo/                    # GRPO configs (advanced)
```

---

## Workflow Summary

```
┌────────────────────────────────────────────────────────────────┐
│                    YOUR AI ENGINEERING WORKFLOW                 │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  STEP 1: Prepare Data                                           │
│  └── python src/scripts/prep_sales_data.py                     │
│      └── Output: data/objection_training.jsonl (100 examples)  │
│                                                                 │
│  STEP 2: Train Model (Colab or M2 Mac)                          │
│  └── python src/scripts/train_objection_handler.py             │
│      └── Output: models/adapters/objection_handler_v1/         │
│                                                                 │
│  STEP 3: Test via MCP Server                                    │
│  └── npm run start:minimal                                      │
│      └── Claude Code uses generate_text tool                    │
│                                                                 │
│  STEP 4: Export for Production                                  │
│  └── export_model tool → GGUF file                              │
│      └── Output: models/exports/objection-handler.gguf         │
│                                                                 │
│  STEP 5: Deploy to Ollama                                       │
│  └── ollama create objection-handler -f Modelfile              │
│                                                                 │
│  STEP 6: Use via API                                            │
│  └── curl http://localhost:11434/api/generate                  │
│      └── Any app can now query your model!                     │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## Mac-Specific Notes

### M2 16GB (Tim's Work Mac)
- Can run: DeepSeek-R1-Distill-Qwen-7B, Qwen2.5-7B-Instruct
- GGUF quantization: q4_k_m or q5_k_m
- Ollama runs natively on Apple Silicon

### M1 8GB (Personal Mac)
- Can run: Qwen2.5-0.5B, Qwen2.5-1.5B, Qwen2.5-3B
- GGUF quantization: q4_0 or q4_k_s (smaller)
- May need to close other apps during inference

---

## Next Steps

1. **First**: Run `npm run start:minimal` to test the MCP server
2. **Then**: Train on your 100 objection examples
3. **Finally**: Export and deploy to Ollama for curl access

Questions? The 6-tool server gives you the foundation - build on it!
