# Unsloth MCP Server v2.1

[![CI](https://github.com/ScientiaCapital/unsloth-mcp-server/workflows/CI/badge.svg)](https://github.com/ScientiaCapital/unsloth-mcp-server/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Node](https://img.shields.io/badge/node-%3E%3D18-brightgreen)](https://nodejs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.2-blue)](https://www.typescriptlang.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

An enhanced MCP server for [Unsloth](https://github.com/unslothai/unsloth) - a library that makes LLM fine-tuning 2x faster with 80% less memory.

## What's New in v2.0

### New Features
- **SuperBPE Tokenizer Training**: Train state-of-the-art SuperBPE tokenizers for up to 33% token reduction
- **Model Information Tool**: Get detailed architecture and parameter information
- **Tokenizer Comparison**: Compare tokenization efficiency between different tokenizers
- **Performance Benchmarking**: Benchmark model inference speed and memory usage
- **Dataset Discovery**: Search and list Hugging Face datasets
- **Dataset Preparation**: Prepare and format datasets for fine-tuning

### Production-Ready Infrastructure
- **Comprehensive Logging**: Winston-based structured logging with multiple transports
- **Input Validation**: Full validation for all tool inputs with helpful error messages
- **Security**: Input sanitization, path validation, resource limits, rate limiting
- **Error Handling**: Smart error detection with contextual suggestions
- **Performance Metrics**: Track tool usage, success rates, and execution times
- **Testing**: 43 comprehensive tests with high coverage
- **Type Safety**: Full TypeScript with strict mode enabled

[See PRODUCTION_GUIDE.md](PRODUCTION_GUIDE.md) for deployment and monitoring details.

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

## FAQ

### General Questions

**Q: What is MCP?**
A: Model Context Protocol (MCP) is Anthropic's protocol that allows Claude to interact with external tools and services. This server implements MCP to give Claude access to Unsloth's fine-tuning capabilities.

**Q: Do I need a GPU?**
A: Yes, for fine-tuning you need an NVIDIA GPU with CUDA support. Minimum 8GB VRAM recommended. Some tools like `list_supported_models` and `check_installation` work without GPU.

**Q: Which Python version is required?**
A: Python 3.10, 3.11, or 3.12. Python 3.13 is not yet supported by Unsloth.

**Q: Can I use this on CPU only?**
A: Fine-tuning on CPU is extremely slow and not recommended. However, you can use the dataset preparation and tokenizer tools on CPU.

### Installation Issues

**Q: "Module 'unsloth' not found"**
A: Install Unsloth: `pip install unsloth`

**Q: "CUDA out of memory" errors**
A: Reduce `batch_size` to 1, reduce `max_seq_length`, or use a smaller model (try Llama-1B instead of 7B).

**Q: Build fails with TypeScript errors**
A: Make sure you're using Node.js 18 or 20. Run `npm install` and `npm run build`.

### Usage Questions

**Q: How long does fine-tuning take?**
A: Depends on hardware and settings:
- RTX 3090: 10-30 minutes (100-500 steps)
- RTX 4090: 5-15 minutes (100-500 steps)
- A100: 3-10 minutes (100-500 steps)

**Q: How much does it cost on cloud GPUs?**
A: Approximately:
- AWS g5.xlarge (A10G): ~$1.50/hour
- Runpod RTX 4090: ~$1.50/hour
- Lambda Labs A100: ~$1.10/hour

Fine-tuning typically costs $0.50-$5 depending on model size and training time.

**Q: Can I fine-tune multiple models simultaneously?**
A: The server supports max 3 concurrent operations by default (configurable via `UNSLOTH_MAX_CONCURRENT_OPERATIONS`). Each operation requires significant GPU memory.

**Q: Where are my models saved?**
A: Models are saved to the `output_dir` you specify in the `finetune_model` call. By default, this is a local directory like `./my-finetuned-model/`.

### Performance Questions

**Q: Why is my training slower than expected?**
A: Check:
1. GPU is being used (`nvidia-smi`)
2. Not using too large `max_seq_length`
3. CUDA drivers are up to date
4. No thermal throttling

**Q: How can I improve model quality?**
A: 
1. Increase `max_steps` (try 500-1000)
2. Use higher quality training data
3. Experiment with `learning_rate` (1e-4 to 5e-4)
4. Use a larger model if resources allow
5. Train for multiple epochs

**Q: How do I reduce token costs with SuperBPE?**
A: Train a SuperBPE tokenizer on your domain-specific corpus. See `examples/02-superbpe-tokenizer.md` for a complete guide. You can typically achieve 20-33% token reduction.

### Docker Questions

**Q: How do I use the Docker version?**
A:
```bash
# Build the image
docker-compose build

# Run the server
docker-compose up unsloth-mcp

# For development with hot reload
docker-compose --profile dev up unsloth-mcp-dev
```

**Q: Do I need NVIDIA Container Toolkit?**
A: Yes, for GPU access in Docker. Install from: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

### Troubleshooting

**Q: Tests are failing**
A: Run `npm install` again, then `npm test`. If specific tests fail, check the error messages and ensure all dependencies are installed.

**Q: "Permission denied" errors**
A: The server has security restrictions on file paths. Check `UNSLOTH_ALLOWED_PATHS` and `UNSLOTH_BLOCKED_PATHS` in your configuration.

**Q: Logs show "Script execution timed out"**
A: Increase `UNSLOTH_MAX_EXECUTION_TIME` (default 10 minutes) or reduce the workload size.

**Q: Cache not working**
A: Check `UNSLOTH_CACHE_ENABLED=true` and ensure the `.cache` directory is writable.

### Contributing

**Q: How can I contribute?**
A: See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. We welcome:
- Bug reports and fixes
- New features
- Documentation improvements
- Example scripts
- Performance optimizations

**Q: How do I report a bug?**
A: Open an issue on GitHub with:
1. Clear description of the problem
2. Steps to reproduce
3. Expected vs actual behavior
4. Environment details (OS, Node version, GPU, etc.)
5. Error logs

## Support

- **Documentation**: See [examples/](examples/) for detailed guides
- **Issues**: [GitHub Issues](https://github.com/ScientiaCapital/unsloth-mcp-server/issues)
- **Unsloth**: [Unsloth GitHub](https://github.com/unslothai/unsloth)

## Related Projects

- [Unsloth](https://github.com/unslothai/unsloth) - Fast LLM fine-tuning library
- [Model Context Protocol](https://modelcontextprotocol.io/) - MCP specification
- [Claude Desktop](https://claude.ai/) - Claude AI assistant

