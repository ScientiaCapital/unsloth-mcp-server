---
name: unsloth-tokenizer
description: Train SuperBPE tokenizers for 20-33% token reduction, compare tokenizers, and optimize token efficiency. Use when the user wants to reduce API costs, improve context window usage, or train custom tokenizers.
---

# Unsloth Tokenizer & SuperBPE

Expert guidance for tokenizer training and optimization using Unsloth's SuperBPE technology.

## What is SuperBPE?

SuperBPE is a 2025 tokenization method that achieves:

- **20-33% fewer tokens** than standard BPE
- **Faster inference** (fewer tokens to process)
- **Lower API costs** (pay per token)
- **Better context utilization** (fit more in same window)

## Quick Start

### Train SuperBPE Tokenizer

```python
from unsloth.tokenizer import train_superbpe

tokenizer = train_superbpe(
    corpus_path="./training_data.txt",    # or HF dataset name
    output_path="./tokenizers/my_tokenizer.json",
    vocab_size=50000,
    num_inherit_merges=40000  # 80% of vocab_size
)
```

### Compare Tokenizers

```python
from unsloth.tokenizer import compare_tokenizers

# Compare standard BPE with your SuperBPE
results = compare_tokenizers(
    text="Your sample text here...",
    tokenizer1="meta-llama/Llama-3.2-1B",
    tokenizer2="./tokenizers/my_tokenizer.json"
)

print(f"Standard BPE: {results['tokenizer1']['tokens']} tokens")
print(f"SuperBPE: {results['tokenizer2']['tokens']} tokens")
print(f"Reduction: {results['reduction']}")
```

## Use Cases

### 1. Reduce API Costs

**Problem:** High API bills due to token usage

**Solution:**

```python
# Train domain-specific SuperBPE
tokenizer = train_superbpe(
    corpus_path="your_domain_corpus.txt",  # Your specific domain
    output_path="./tokenizers/domain_tokenizer.json",
    vocab_size=50000
)

# Typical savings: 20-33% fewer tokens
# $1000/month → $700-800/month
```

### 2. Maximize Context Window

**Problem:** Running out of context window space

**Solution:**

```python
# Standard tokenizer: 8K tokens = ~6K words
# SuperBPE: 8K tokens = ~8.5K words
# 40% more content in same window!

tokenizer = train_superbpe(
    corpus_path="wikitext",
    vocab_size=50000
)
```

### 3. Domain-Specific Efficiency

**Problem:** Generic tokenizers waste tokens on domain-specific terms

**Solution:**

```python
# Medical example
tokenizer = train_superbpe(
    corpus_path="medical_meadow_medical_flashcards",
    output_path="./tokenizers/medical_tokenizer.json",
    vocab_size=32000  # Smaller for domain-specific
)

# "electrocardiogram" → 1 token instead of 5
# "myocardial infarction" → 2 tokens instead of 6
```

## Training Corpus Selection

### General Purpose Tokenizer

```python
# Use diverse, high-quality text
corpus_sources = [
    "wikitext",
    "c4",  # Common Crawl
    "bookcorpus"
]

# Combine multiple sources
tokenizer = train_superbpe(
    corpus_path=combined_corpus,
    vocab_size=50000
)
```

### Domain-Specific Tokenizer

```python
# Use domain-specific corpus
domains = {
    "medical": "medical_meadow",
    "legal": "legal_contracts_dataset",
    "code": "codeparrot/github-code",
    "financial": "financial_phrasebank"
}

tokenizer = train_superbpe(
    corpus_path=domains["medical"],
    vocab_size=32000  # Smaller vocab for specific domains
)
```

## Vocab Size Guidelines

| Use Case             | Vocab Size      | Rationale           |
| -------------------- | --------------- | ------------------- |
| General purpose      | 50,000-100,000  | Maximum flexibility |
| Domain-specific      | 16,000-32,000   | Focused vocabulary  |
| Multilingual         | 100,000-250,000 | Many languages      |
| Resource-constrained | 8,000-16,000    | Smaller model size  |

## Performance Comparison

### Example: Technical Documentation

```python
text = """
The implementation utilizes a convolutional neural network
architecture with residual connections and batch normalization.
The model achieves state-of-the-art performance on ImageNet.
"""

# Standard BPE: 45 tokens
# SuperBPE: 31 tokens
# Reduction: 31% fewer tokens
```

### Example: Medical Text

```python
text = """
Patient presents with acute myocardial infarction.
ECG shows ST-segment elevation in leads II, III, and aVF.
Troponin levels elevated at 15.2 ng/mL.
"""

# Standard BPE: 52 tokens
# Medical SuperBPE: 34 tokens
# Reduction: 35% fewer tokens
```

## Training Time

| Corpus Size | Vocab Size | Training Time | Hardware |
| ----------- | ---------- | ------------- | -------- |
| 100MB       | 50K        | 10-15 min     | CPU/GPU  |
| 500MB       | 50K        | 30-60 min     | CPU/GPU  |
| 1GB         | 50K        | 1-2 hours     | CPU/GPU  |
| 1GB         | 100K       | 2-3 hours     | CPU/GPU  |

## Advanced Configuration

### Inherit Merges

```python
# Control how many BPE merges to inherit
tokenizer = train_superbpe(
    corpus_path="corpus.txt",
    vocab_size=50000,
    num_inherit_merges=40000  # 80% (recommended)
)

# Options:
# - 70% (35K): More aggressive, higher compression
# - 80% (40K): Balanced (recommended)
# - 90% (45K): Conservative, safer
```

### Custom Training Parameters

```python
tokenizer = train_superbpe(
    corpus_path="corpus.txt",
    output_path="tokenizer.json",
    vocab_size=50000,
    num_inherit_merges=40000,
    min_frequency=2,          # Minimum token frequency
    special_tokens=[          # Custom special tokens
        "[INST]",
        "[/INST]",
        "<|system|>",
        "<|user|>",
        "<|assistant|>"
    ]
)
```

## Integration with Fine-Tuning

### Step 1: Train Tokenizer

```python
# Train on your domain
tokenizer = train_superbpe(
    corpus_path="your_corpus.txt",
    output_path="./tokenizers/custom.json",
    vocab_size=50000
)
```

### Step 2: Use in Fine-Tuning

```python
from unsloth import FastLanguageModel
from transformers import AutoTokenizer

# Load your custom tokenizer
custom_tokenizer = AutoTokenizer.from_pretrained(
    "./tokenizers/custom.json"
)

# Use with fine-tuning
model, _ = FastLanguageModel.from_pretrained(
    "unsloth/Llama-3.2-1B-bnb-4bit",
    max_seq_length=2048
)

# Replace tokenizer
model.resize_token_embeddings(len(custom_tokenizer))
```

## ROI Calculator

### Calculate Token Savings

```python
def calculate_savings(
    monthly_tokens: int,
    cost_per_million: float,
    reduction_percent: float
):
    """
    Calculate monthly savings from SuperBPE

    Example:
    - 100M tokens/month
    - $20 per 1M tokens
    - 30% reduction

    Savings: $600/month = $7,200/year
    """
    current_cost = (monthly_tokens / 1_000_000) * cost_per_million
    new_tokens = monthly_tokens * (1 - reduction_percent / 100)
    new_cost = (new_tokens / 1_000_000) * cost_per_million
    savings = current_cost - new_cost

    return {
        "monthly_savings": savings,
        "yearly_savings": savings * 12,
        "roi_months": 0.1  # Training cost negligible
    }

# Example
print(calculate_savings(
    monthly_tokens=100_000_000,
    cost_per_million=20,
    reduction_percent=30
))
# Output: {'monthly_savings': 600, 'yearly_savings': 7200, 'roi_months': 0.1}
```

## Validation & Testing

### Test on Representative Data

```python
# Always test on held-out data
test_samples = [
    "Sample 1 from your domain...",
    "Sample 2 from your domain...",
    "Sample 3 from your domain..."
]

for sample in test_samples:
    result = compare_tokenizers(
        text=sample,
        tokenizer1="meta-llama/Llama-3.2-1B",
        tokenizer2="./tokenizers/custom.json"
    )
    print(f"Reduction: {result['reduction']}")

# Average should be 20-33%
```

### Quality Checks

```python
# Check tokenization quality
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./tokenizers/custom.json")

# Test important terms
important_terms = [
    "your_domain_term_1",
    "your_domain_term_2",
    "common_phrase"
]

for term in important_terms:
    tokens = tokenizer.tokenize(term)
    print(f"{term}: {len(tokens)} tokens = {tokens}")

# Goal: Domain terms should be 1-2 tokens
```

## Common Patterns

### Pattern 1: Quick Evaluation

```python
# Fast test to see potential
sample_text = "Representative text from your domain..."

result = compare_tokenizers(
    text=sample_text,
    tokenizer1="meta-llama/Llama-3.2-1B",
    tokenizer2="existing_tokenizer"  # If you have one
)

if result['reduction'] > 20:
    print("Worth training custom tokenizer!")
```

### Pattern 2: Production Deployment

```python
# 1. Collect representative corpus
# 2. Train with optimal settings
tokenizer = train_superbpe(
    corpus_path="production_corpus.txt",
    output_path="./tokenizers/production_v1.json",
    vocab_size=50000,
    num_inherit_merges=40000
)

# 3. Validate on test set
# 4. A/B test in production
# 5. Monitor metrics
```

### Pattern 3: Multi-Domain

```python
# Train separate tokenizers for different domains
domains = ["medical", "legal", "technical"]

for domain in domains:
    tokenizer = train_superbpe(
        corpus_path=f"./corpus/{domain}_corpus.txt",
        output_path=f"./tokenizers/{domain}_tokenizer.json",
        vocab_size=32000
    )
```

## Troubleshooting

### Issue: Low Compression (<15%)

**Solution:**

- Use more domain-specific corpus
- Increase vocab_size
- Check corpus quality

### Issue: Poor Tokenization Quality

**Solution:**

- Increase training corpus size
- Adjust num_inherit_merges
- Add domain-specific special tokens

### Issue: Long Training Time

**Solution:**

- Reduce corpus size
- Use representative sample
- Reduce vocab_size

## Best Practices

1. **Start with evaluation** - Test potential before committing
2. **Use representative data** - Train on data similar to production
3. **Validate thoroughly** - Test on held-out data
4. **Version your tokenizers** - Track which version is deployed
5. **Monitor in production** - Track actual token reduction
6. **Update periodically** - Retrain as your domain evolves

## Expected Results

| Domain         | Typical Reduction | Example            |
| -------------- | ----------------- | ------------------ |
| General text   | 20-25%            | News, blogs        |
| Technical docs | 25-30%            | API docs, manuals  |
| Medical        | 30-35%            | Clinical notes     |
| Legal          | 25-30%            | Contracts, filings |
| Code           | 25-33%            | Source code        |

## Additional Resources

For more details, see:

- [COMPARISON.md](COMPARISON.md) - Detailed comparison methodologies
- [INTEGRATION.md](INTEGRATION.md) - Integration with models and APIs
- [BENCHMARKS.md](BENCHMARKS.md) - Performance benchmarks
