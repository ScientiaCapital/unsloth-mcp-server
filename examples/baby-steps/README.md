# Baby Steps: Learn Unsloth Fine-Tuning

> **"Start like a baby crawling"** - Master the basics before running

## The Golden Rule

```
SFT BEFORE GRPO!
```

**SFT (Supervised Fine-Tuning)** teaches the model WHAT to do.
**GRPO (Group Relative Policy Optimization)** teaches HOW to do it better.

You can't optimize a model that doesn't know the task!

## Learning Path

```
CRAWL (Week 1): SFT on 20 math examples → IT WORKS!
   ↓
WALK (Week 2): SFT on YOUR sales/CRM data (swap the dataset)
   ↓
JOG (Week 3): SFT with more data, longer training
   ↓
RUN (Week 4+): ONLY NOW try GRPO
```

## Quick Start (10 minutes)

### Option A: Google Colab (Easiest)
1. Open [finetuning1.ipynb](https://colab.research.google.com/) in Colab
2. Set Runtime → T4 GPU
3. Run all cells
4. Download `my_model.zip`

### Option B: These Scripts
```bash
# In a GPU environment (Colab, RunPod, local)
python 00-install-unsloth.py  # Install dependencies
python 01-check-gpu.py        # Verify GPU
python 05-train-model.py      # Train (includes all steps)
python 08-inference.py        # Test your model
```

## The Scripts

| Script | What It Does | Time |
|--------|--------------|------|
| `00-install-unsloth.py` | Install Unsloth + deps | 2-3 min |
| `01-check-gpu.py` | Verify GPU is available | instant |
| `02-load-model.py` | Load Qwen2.5-0.5B | 1-2 min |
| `03-add-lora.py` | Add trainable adapters | instant |
| `04-create-dataset.py` | Create training data | instant |
| `05-train-model.py` | **Main training script** | 2-3 min |
| `06-test-model.py` | Test the model | 1 min |
| `07-save-model.py` | Save adapter (~50MB) | instant |
| `08-inference.py` | Use for inference | instant |

## Swap Your Data

The key insight: **Cell 5 / 04-create-dataset.py has hardcoded math examples.**

Once training works, swap with YOUR data:

### From coperniq-forge (Real Sales Emails)
```bash
# Link the training data
ln -s ~/projects/coperniq-forge/data/processed/training.jsonl data/training.jsonl

# Run training - it will auto-detect the file
python 05-train-model.py
```

### From Your Own Data
Create `data/training.jsonl` with one example per line:
```json
{"instruction": "Write a cold email to...", "output": "Hi [Name]..."}
{"instruction": "Extract CRM data from...", "output": "{\"company\": ...}"}
```

Or ChatML format:
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

## Expected Results

After 60 steps of training on 20 examples:
- **Loss**: Starts ~2.5, ends ~0.5-1.0
- **Training questions**: Should get correct
- **New questions**: May generalize (depends on training data)

## Troubleshooting

### "No GPU found"
- Colab: Runtime → Change runtime type → T4 GPU
- RunPod: Select a GPU template
- Local: Install NVIDIA CUDA drivers

### Loss stays high
- Check data format (should be ChatML)
- Increase training steps
- Check for data quality issues

### Model outputs garbage
- Data format is wrong
- Training didn't converge
- Try more examples (50-100)

## Next Steps After SFT Works

1. **Add more data** - 100-500 examples
2. **Train longer** - 200-500 steps
3. **Use your domain** - Sales emails, CRM extraction, etc.
4. **Export to GGUF** - For Ollama deployment
5. **ONLY THEN: GRPO** - Preference optimization

## Resources

- [Unsloth Docs](https://github.com/unslothai/unsloth)
- [coperniq-forge](https://github.com/ScientiaCapital/coperniq-forge) - Sales training data
- [Training Data Docs](../../training-data/) - GRPO configs (for later)
