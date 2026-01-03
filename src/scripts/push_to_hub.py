#!/usr/bin/env python3
"""
Push trained models to HuggingFace Hub for team sharing.

Usage:
    # Push LoRA adapter only (smaller, faster)
    python push_to_hub.py --adapter models/adapters/my_model --repo ScientiaCapital/objection-handler-lora

    # Push merged model (full model, larger but standalone)
    python push_to_hub.py --adapter models/adapters/my_model --repo ScientiaCapital/objection-handler --merge

    # Push GGUF for Ollama users
    python push_to_hub.py --gguf models/exports/model.gguf --repo ScientiaCapital/objection-handler-gguf
"""

import argparse
import os
from pathlib import Path

def setup_auth():
    """Ensure HuggingFace authentication is set up."""
    from huggingface_hub import HfApi, login

    # Check for token in environment
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    if not token:
        print("=" * 60)
        print("HuggingFace Authentication Required")
        print("=" * 60)
        print("\nOption 1: Set environment variable")
        print("  export HF_TOKEN=hf_xxxxx")
        print("\nOption 2: Login interactively")
        print("  huggingface-cli login")
        print("\nGet your token at: https://huggingface.co/settings/tokens")
        print("=" * 60)

        # Try interactive login
        try:
            login()
            return True
        except Exception as e:
            print(f"Login failed: {e}")
            return False

    # Verify token works
    try:
        api = HfApi(token=token)
        user = api.whoami()
        print(f"✓ Authenticated as: {user['name']}")
        return True
    except Exception as e:
        print(f"Token validation failed: {e}")
        return False


def push_lora_adapter(adapter_path: str, repo_id: str, private: bool = False):
    """
    Push LoRA adapter to HuggingFace Hub.

    Advantages:
    - Small size (~35MB for 7B model)
    - Fast upload
    - Users load base model + your adapter
    """
    from huggingface_hub import HfApi, upload_folder

    adapter_path = Path(adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")

    print(f"\n📤 Pushing LoRA adapter to: {repo_id}")
    print(f"   Source: {adapter_path}")

    api = HfApi()

    # Create repo if needed
    try:
        api.create_repo(repo_id, private=private, exist_ok=True)
        print(f"   Repository: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"   Note: {e}")

    # Upload adapter files
    upload_folder(
        folder_path=str(adapter_path),
        repo_id=repo_id,
        commit_message="Upload LoRA adapter trained with Unsloth",
    )

    print(f"\n✅ Success! Your adapter is at:")
    print(f"   https://huggingface.co/{repo_id}")
    print(f"\n📝 Usage for your team:")
    print(f"""
from unsloth import FastLanguageModel

# Load base model + your adapter
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="{repo_id}",  # Your adapter
    load_in_4bit=True
)
""")


def push_merged_model(adapter_path: str, repo_id: str, base_model: str = None, private: bool = False):
    """
    Merge adapter into base model and push full model.

    Advantages:
    - Standalone (no need to specify base model)
    - Works with any HuggingFace-compatible tool

    Disadvantages:
    - Large size (~14GB for 7B model)
    - Slower upload
    """
    from unsloth import FastLanguageModel
    from huggingface_hub import HfApi

    adapter_path = Path(adapter_path)

    # Detect base model from adapter config
    if not base_model:
        import json
        config_path = adapter_path / "adapter_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            base_model = config.get("base_model_name_or_path")

    if not base_model:
        raise ValueError("Could not detect base model. Please specify --base-model")

    print(f"\n📤 Merging and pushing to: {repo_id}")
    print(f"   Base model: {base_model}")
    print(f"   Adapter: {adapter_path}")

    # Load model with adapter
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        load_in_4bit=True,
    )

    # Load adapter weights
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(adapter_path))

    # Merge adapter into base model
    print("   Merging adapter into base model...")
    model = model.merge_and_unload()

    # Push to hub
    print("   Uploading merged model (this may take a while)...")
    model.push_to_hub(repo_id, private=private)
    tokenizer.push_to_hub(repo_id, private=private)

    print(f"\n✅ Success! Your merged model is at:")
    print(f"   https://huggingface.co/{repo_id}")
    print(f"\n📝 Usage for your team:")
    print(f"""
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
""")


def push_gguf(gguf_path: str, repo_id: str, private: bool = False):
    """
    Push GGUF file to HuggingFace for Ollama users.

    Advantages:
    - Works with Ollama, llama.cpp, LM Studio
    - Quantized (smaller size)
    - CPU-friendly
    """
    from huggingface_hub import HfApi, upload_file

    gguf_path = Path(gguf_path)
    if not gguf_path.exists():
        raise FileNotFoundError(f"GGUF not found: {gguf_path}")

    print(f"\n📤 Pushing GGUF to: {repo_id}")
    print(f"   Source: {gguf_path}")

    api = HfApi()

    # Create repo
    api.create_repo(repo_id, private=private, exist_ok=True)

    # Upload GGUF
    upload_file(
        path_or_fileobj=str(gguf_path),
        path_in_repo=gguf_path.name,
        repo_id=repo_id,
        commit_message="Upload GGUF quantized model",
    )

    # Create model card
    model_card = f"""---
license: apache-2.0
tags:
- gguf
- ollama
- unsloth
---

# {repo_id.split('/')[-1]}

GGUF quantized model for use with Ollama, llama.cpp, or LM Studio.

## Usage with Ollama

```bash
# Download and run
ollama run hf.co/{repo_id}

# Or create a custom Modelfile
cat > Modelfile << 'EOF'
FROM hf.co/{repo_id}/{gguf_path.name}

TEMPLATE \"\"\"{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
\"\"\"

PARAMETER temperature 0.7
PARAMETER stop "<|im_end|>"
EOF

ollama create my-model -f Modelfile
```

## Usage with llama.cpp

```bash
./main -m {gguf_path.name} -p "Your prompt here"
```

Trained with [Unsloth](https://github.com/unslothai/unsloth)
"""

    upload_file(
        path_or_fileobj=model_card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        commit_message="Add model card",
    )

    print(f"\n✅ Success! Your GGUF is at:")
    print(f"   https://huggingface.co/{repo_id}")
    print(f"\n📝 Usage for your team:")
    print(f"""
# With Ollama (easiest)
ollama run hf.co/{repo_id}

# Or download manually
huggingface-cli download {repo_id} {gguf_path.name}
""")


def main():
    parser = argparse.ArgumentParser(
        description="Push trained models to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Push LoRA adapter (recommended - small and fast)
  python push_to_hub.py --adapter models/adapters/my_model --repo ScientiaCapital/objection-handler-lora

  # Push merged model (standalone, but large)
  python push_to_hub.py --adapter models/adapters/my_model --repo ScientiaCapital/objection-handler --merge

  # Push GGUF for Ollama
  python push_to_hub.py --gguf models/exports/model.gguf --repo ScientiaCapital/objection-handler-gguf

  # Make it private (team only)
  python push_to_hub.py --adapter models/adapters/my_model --repo ScientiaCapital/objection-handler-lora --private
        """
    )

    parser.add_argument("--adapter", help="Path to LoRA adapter directory")
    parser.add_argument("--gguf", help="Path to GGUF file")
    parser.add_argument("--repo", required=True, help="HuggingFace repo (e.g., ScientiaCapital/my-model)")
    parser.add_argument("--merge", action="store_true", help="Merge adapter into base model before pushing")
    parser.add_argument("--base-model", help="Base model for merging (auto-detected if not specified)")
    parser.add_argument("--private", action="store_true", help="Make repository private")

    args = parser.parse_args()

    # Validate arguments
    if not args.adapter and not args.gguf:
        parser.error("Must specify either --adapter or --gguf")

    # Setup authentication
    if not setup_auth():
        print("\n❌ Authentication failed. Please set HF_TOKEN or run 'huggingface-cli login'")
        return 1

    try:
        if args.gguf:
            push_gguf(args.gguf, args.repo, args.private)
        elif args.merge:
            push_merged_model(args.adapter, args.repo, args.base_model, args.private)
        else:
            push_lora_adapter(args.adapter, args.repo, args.private)

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
