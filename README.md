# Unsloth MCP Server v2.0

An enhanced MCP server for [Unsloth](https://github.com/unslothai/unsloth) - a library that makes LLM fine-tuning 2x faster with 80% less memory.

## What's New in v2.0

- **SuperBPE Tokenizer Training**: Train state-of-the-art SuperBPE tokenizers for up to 33% token reduction
- **Model Information Tool**: Get detailed architecture and parameter information
- **Tokenizer Comparison**: Compare tokenization efficiency between different tokenizers
- **Performance Benchmarking**: Benchmark model inference speed and memory usage
- **Dataset Discovery**: Search and list Hugging Face datasets
- **Dataset Preparation**: Prepare and format datasets for fine-tuning

## What is Unsloth?

Unsloth is a library that dramatically improves the efficiency of fine-tuning large language models:

- **Speed**: 2x faster fine-tuning compared to standard methods
- **Memory**: 80% less VRAM usage, allowing fine-tuning of larger models on consumer GPUs
- **Context Length**: Up to 13x longer context lengths (e.g., 89K tokens for Llama 3.3 on 80GB GPUs)
- **Accuracy**: No loss in model quality or performance

Unsloth achieves these improvements through custom CUDA kernels written in OpenAI's Triton language, optimized backpropagation, and dynamic 4-bit quantization.

## Features

- Optimize fine-tuning for Llama, Mistral, Phi, Gemma, and other models
- 4-bit quantization for efficient training
- Extended context length support
- Simple API for model loading, fine-tuning, and inference
- Export to various formats (GGUF, Hugging Face, etc.)

## Quick Start

1. Install Unsloth: `pip install unsloth`
2. Install and build the server:
   ```bash
   cd unsloth-server
   npm install
   npm run build
   ```
3. Add to MCP settings:
   ```json
   {
     "mcpServers": {
       "unsloth-server": {
         "command": "node",
         "args": ["/path/to/unsloth-server/build/index.js"],
         "env": {
           "HUGGINGFACE_TOKEN": "your_token_here" // Optional
         },
         "disabled": false,
         "autoApprove": []
       }
     }
   }
   ```

## Available Tools

### check_installation

Verify if Unsloth is properly installed on your system.

**Parameters**: None

**Example**:
```javascript
const result = await use_mcp_tool({
  server_name: "unsloth-server",
  tool_name: "check_installation",
  arguments: {}
});
```

### list_supported_models

Get a list of all models supported by Unsloth, including Llama, Mistral, Phi, and Gemma variants.

**Parameters**: None

**Example**:
```javascript
const result = await use_mcp_tool({
  server_name: "unsloth-server",
  tool_name: "list_supported_models",
  arguments: {}
});
```

### load_model

Load a pretrained model with Unsloth optimizations for faster inference and fine-tuning.

**Parameters**:
- `model_name` (required): Name of the model to load (e.g., "unsloth/Llama-3.2-1B")
- `max_seq_length` (optional): Maximum sequence length for the model (default: 2048)
- `load_in_4bit` (optional): Whether to load the model in 4-bit quantization (default: true)
- `use_gradient_checkpointing` (optional): Whether to use gradient checkpointing to save memory (default: true)

**Example**:
```javascript
const result = await use_mcp_tool({
  server_name: "unsloth-server",
  tool_name: "load_model",
  arguments: {
    model_name: "unsloth/Llama-3.2-1B",
    max_seq_length: 4096,
    load_in_4bit: true
  }
});
```

### finetune_model

Fine-tune a model with Unsloth optimizations using LoRA/QLoRA techniques.

**Parameters**:
- `model_name` (required): Name of the model to fine-tune
- `dataset_name` (required): Name of the dataset to use for fine-tuning
- `output_dir` (required): Directory to save the fine-tuned model
- `max_seq_length` (optional): Maximum sequence length for training (default: 2048)
- `lora_rank` (optional): Rank for LoRA fine-tuning (default: 16)
- `lora_alpha` (optional): Alpha for LoRA fine-tuning (default: 16)
- `batch_size` (optional): Batch size for training (default: 2)
- `gradient_accumulation_steps` (optional): Number of gradient accumulation steps (default: 4)
- `learning_rate` (optional): Learning rate for training (default: 2e-4)
- `max_steps` (optional): Maximum number of training steps (default: 100)
- `dataset_text_field` (optional): Field in the dataset containing the text (default: 'text')
- `load_in_4bit` (optional): Whether to use 4-bit quantization (default: true)

**Example**:
```javascript
const result = await use_mcp_tool({
  server_name: "unsloth-server",
  tool_name: "finetune_model",
  arguments: {
    model_name: "unsloth/Llama-3.2-1B",
    dataset_name: "tatsu-lab/alpaca",
    output_dir: "./fine-tuned-model",
    max_steps: 100,
    batch_size: 2,
    learning_rate: 2e-4
  }
});
```

### generate_text

Generate text using a fine-tuned Unsloth model.

**Parameters**:
- `model_path` (required): Path to the fine-tuned model
- `prompt` (required): Prompt for text generation
- `max_new_tokens` (optional): Maximum number of tokens to generate (default: 256)
- `temperature` (optional): Temperature for text generation (default: 0.7)
- `top_p` (optional): Top-p for text generation (default: 0.9)

**Example**:
```javascript
const result = await use_mcp_tool({
  server_name: "unsloth-server",
  tool_name: "generate_text",
  arguments: {
    model_path: "./fine-tuned-model",
    prompt: "Write a short story about a robot learning to paint:",
    max_new_tokens: 512,
    temperature: 0.8
  }
});
```

### export_model

Export a fine-tuned Unsloth model to various formats for deployment.

**Parameters**:
- `model_path` (required): Path to the fine-tuned model
- `export_format` (required): Format to export to (gguf, ollama, vllm, huggingface)
- `output_path` (required): Path to save the exported model
- `quantization_bits` (optional): Bits for quantization (for GGUF export) (default: 4)

**Example**:
```javascript
const result = await use_mcp_tool({
  server_name: "unsloth-server",
  tool_name: "export_model",
  arguments: {
    model_path: "./fine-tuned-model",
    export_format: "gguf",
    output_path: "./exported-model.gguf",
    quantization_bits: 4
  }
});
```

### train_superbpe_tokenizer

Train a SuperBPE tokenizer for improved efficiency. SuperBPE tokenizers can encode text using up to 33% fewer tokens than standard BPE tokenizers, leading to faster inference and lower costs.

**Parameters**:
- `corpus_path` (required): Path to training corpus or dataset name from Hugging Face
- `output_path` (required): Path to save the trained tokenizer
- `vocab_size` (optional): Vocabulary size for the tokenizer (default: 50000)
- `num_inherit_merges` (optional): Number of merges to inherit from BPE stage (default: vocab_size * 0.8)

**Example**:
```javascript
const result = await use_mcp_tool({
  server_name: "unsloth-server",
  tool_name: "train_superbpe_tokenizer",
  arguments: {
    corpus_path: "wikitext",
    output_path: "./tokenizers/superbpe_tokenizer.json",
    vocab_size: 50000
  }
});
```

**Benefits**:
- 20-33% reduction in token count
- Faster inference times
- Lower API costs
- Better performance on downstream tasks

### get_model_info

Get detailed information about a model including architecture, parameters, and capabilities.

**Parameters**:
- `model_name` (required): Name or path of the model to inspect

**Example**:
```javascript
const result = await use_mcp_tool({
  server_name: "unsloth-server",
  tool_name: "get_model_info",
  arguments: {
    model_name: "unsloth/Llama-3.2-1B"
  }
});
```

**Returns**:
- Architecture type and details
- Number of parameters
- Hidden size and layers
- Vocabulary size
- Max sequence length
- Memory requirements

### compare_tokenizers

Compare tokenization efficiency between different tokenizers (e.g., standard BPE vs SuperBPE).

**Parameters**:
- `text` (required): Sample text to tokenize for comparison
- `tokenizer1_path` (required): Path to first tokenizer
- `tokenizer2_path` (required): Path to second tokenizer

**Example**:
```javascript
const result = await use_mcp_tool({
  server_name: "unsloth-server",
  tool_name: "compare_tokenizers",
  arguments: {
    text: "Your sample text here for comparison...",
    tokenizer1_path: "meta-llama/Llama-3.2-1B",
    tokenizer2_path: "./tokenizers/superbpe_tokenizer.json"
  }
});
```

**Returns**:
- Token counts for each tokenizer
- Efficiency gain percentage
- Winner determination

### benchmark_model

Benchmark model inference speed and memory usage with Unsloth optimizations.

**Parameters**:
- `model_name` (required): Name of the model to benchmark
- `prompt` (required): Sample prompt for benchmarking
- `num_iterations` (optional): Number of iterations to run (default: 10)
- `max_new_tokens` (optional): Tokens to generate per iteration (default: 128)

**Example**:
```javascript
const result = await use_mcp_tool({
  server_name: "unsloth-server",
  tool_name: "benchmark_model",
  arguments: {
    model_name: "unsloth/Llama-3.2-1B",
    prompt: "Write a story about AI:",
    num_iterations: 10,
    max_new_tokens: 128
  }
});
```

**Returns**:
- Average/min/max inference times
- Tokens per second
- Memory usage statistics

### list_datasets

List popular datasets available for fine-tuning from Hugging Face.

**Parameters**:
- `search_query` (optional): Search query to filter datasets
- `limit` (optional): Maximum number of datasets to return (default: 20)

**Example**:
```javascript
const result = await use_mcp_tool({
  server_name: "unsloth-server",
  tool_name: "list_datasets",
  arguments: {
    search_query: "instruction",
    limit: 20
  }
});
```

**Returns**:
- Dataset IDs and authors
- Download counts and likes
- Relevant tags

### prepare_dataset

Prepare and format a dataset for Unsloth fine-tuning.

**Parameters**:
- `dataset_name` (required): Name of the dataset to prepare
- `output_path` (required): Path to save the prepared dataset
- `text_field` (optional): Field containing the text data (default: "text")
- `format` (optional): Output format - json, jsonl, or csv (default: jsonl)

**Example**:
```javascript
const result = await use_mcp_tool({
  server_name: "unsloth-server",
  tool_name: "prepare_dataset",
  arguments: {
    dataset_name: "tatsu-lab/alpaca",
    output_path: "./datasets/alpaca_prepared.jsonl",
    text_field: "text",
    format: "jsonl"
  }
});
```

**Returns**:
- Number of examples processed
- Output file path
- Format confirmation

## Advanced Usage

### SuperBPE Workflow

Train and use SuperBPE tokenizers for maximum efficiency:

```javascript
// 1. Prepare your training dataset
const prepResult = await use_mcp_tool({
  server_name: "unsloth-server",
  tool_name: "prepare_dataset",
  arguments: {
    dataset_name: "wikitext",
    output_path: "./data/training_corpus.jsonl",
    format: "jsonl"
  }
});

// 2. Train a SuperBPE tokenizer
const tokenizer = await use_mcp_tool({
  server_name: "unsloth-server",
  tool_name: "train_superbpe_tokenizer",
  arguments: {
    corpus_path: "./data/training_corpus.jsonl",
    output_path: "./tokenizers/superbpe.json",
    vocab_size: 50000
  }
});

// 3. Compare with standard tokenizer
const comparison = await use_mcp_tool({
  server_name: "unsloth-server",
  tool_name: "compare_tokenizers",
  arguments: {
    text: "Sample text to compare tokenization efficiency...",
    tokenizer1_path: "meta-llama/Llama-3.2-1B",
    tokenizer2_path: "./tokenizers/superbpe.json"
  }
});

// Result: SuperBPE typically shows 20-33% reduction in tokens!
```

### Model Analysis and Benchmarking

Get comprehensive model information and performance metrics:

```javascript
// 1. Get detailed model information
const info = await use_mcp_tool({
  server_name: "unsloth-server",
  tool_name: "get_model_info",
  arguments: {
    model_name: "unsloth/Llama-3.2-1B"
  }
});

// 2. Benchmark the model
const benchmark = await use_mcp_tool({
  server_name: "unsloth-server",
  tool_name: "benchmark_model",
  arguments: {
    model_name: "unsloth/Llama-3.2-1B",
    prompt: "Write a short story:",
    num_iterations: 10,
    max_new_tokens: 128
  }
});

// Compare: tokens/sec, memory usage, inference time
```

### Dataset Discovery and Preparation

Find and prepare the perfect dataset for your fine-tuning needs:

```javascript
// 1. Search for relevant datasets
const datasets = await use_mcp_tool({
  server_name: "unsloth-server",
  tool_name: "list_datasets",
  arguments: {
    search_query: "instruction following",
    limit: 20
  }
});

// 2. Prepare selected dataset
const prepared = await use_mcp_tool({
  server_name: "unsloth-server",
  tool_name: "prepare_dataset",
  arguments: {
    dataset_name: "tatsu-lab/alpaca",
    output_path: "./datasets/alpaca.jsonl",
    text_field: "text",
    format: "jsonl"
  }
});

// 3. Fine-tune with prepared dataset
const result = await use_mcp_tool({
  server_name: "unsloth-server",
  tool_name: "finetune_model",
  arguments: {
    model_name: "unsloth/Llama-3.2-1B",
    dataset_name: "./datasets/alpaca.jsonl",
    output_dir: "./fine-tuned-model"
  }
});
```

### Custom Datasets

You can use custom datasets by formatting them properly and hosting them on Hugging Face or providing a local path:

```javascript
const result = await use_mcp_tool({
  server_name: "unsloth-server",
  tool_name: "finetune_model",
  arguments: {
    model_name: "unsloth/Llama-3.2-1B",
    dataset_name: "json",
    data_files: {"train": "path/to/your/data.json"},
    output_dir: "./fine-tuned-model"
  }
});
```

### Memory Optimization

For large models on limited hardware:
- Reduce batch size and increase gradient accumulation steps
- Use 4-bit quantization
- Enable gradient checkpointing
- Reduce sequence length if possible

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size, use 4-bit quantization, or try a smaller model
2. **Import Errors**: Ensure you have the correct versions of torch, transformers, and unsloth installed
3. **Model Not Found**: Check that you're using a supported model name or have access to private models

### Version Compatibility

- Python: 3.10, 3.11, or 3.12 (not 3.13)
- CUDA: 11.8 or 12.1+ recommended
- PyTorch: 2.0+ recommended

## Performance Benchmarks

| Model | VRAM | Unsloth Speed | VRAM Reduction | Context Length |
|-------|------|---------------|----------------|----------------|
| Llama 3.3 (70B) | 80GB | 2x faster | >75% | 13x longer |
| Llama 3.1 (8B) | 80GB | 2x faster | >70% | 12x longer |
| Mistral v0.3 (7B) | 80GB | 2.2x faster | 75% less | - |

## Requirements

- Python 3.10-3.12
- NVIDIA GPU with CUDA support (recommended)
- Node.js and npm

## License

Apache-2.0
