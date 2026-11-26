# Server Model Discovery & Inventory Guide

This document provides exact, dedicated commands to discover, inspect, and identify all AI models stored on your server disks (Text LLMs, Vision-Language VLMs, Embeddings, and Evaluation Judges).

---

## 1. Master Model Scanner: Discover All Models & Locations

Run this single command on your server terminal to scan all disks (`/data1`, `/data2`, `/home`, `/opt`) and report every AI model found with its exact path, model type (Text, Vision, or Embedding), and disk size:

```bash
python -c "
import os
from pathlib import Path

search_roots = ['/data1', '/data2', '/home/aimlsvcadmin', '/opt']
model_indicators = ['config.json', 'model.safetensors.index.json', 'pytorch_model.bin.index.json', 'tokenizer_config.json', 'preprocessor_config.json']

print('=' * 80)
print(' SCANNING SERVER DISKS FOR LOCAL AI MODELS...')
print('=' * 80)

found_models = []

for root_dir in search_roots:
    if not os.path.exists(root_dir):
        continue
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip hidden git or cache debris
        if '/.git/' in dirpath or '/.cache/' in dirpath:
            continue
        
        has_config = 'config.json' in filenames
        has_weights = any(f.endswith('.safetensors') or f.endswith('.bin') or f.endswith('.gguf') for f in filenames)
        
        if has_config and has_weights:
            path_obj = Path(dirpath)
            # Estimate folder size
            try:
                total_bytes = sum(f.stat().st_size for f in path_obj.glob('*') if f.is_file())
                size_gb = round(total_bytes / (1024**3), 2)
            except Exception:
                size_gb = 0.0

            # Detect model architecture type
            model_type = 'Text LLM'
            if any('vision' in f.lower() or 'processor' in f.lower() for f in filenames) or 'vl' in dirpath.lower() or 'vision' in dirpath.lower():
                model_type = 'Vision-Language (VLM)'
            elif 'bge' in dirpath.lower() or 'embed' in dirpath.lower() or 'sentence' in dirpath.lower():
                model_type = 'Embedding Model'

            found_models.append((dirpath, model_type, size_gb))

print(f'\nTotal Models Found: {len(found_models)}\n')
for idx, (path, mtype, sz) in enumerate(found_models, 1):
    print(f'[{idx:02d}] Type: {mtype:<25} | Size: {sz:>6.2f} GB')
    print(f'     Path: {path}\n')
print('=' * 80)
"
```

---

## 2. Inspect Found Model Directories in Detail

Quickly inspect what files exist inside the model folders under `/data1/aimlDevTeam/`:

```bash
ls -lh /data1/aimlDevTeam/QWEN /data1/aimlDevTeam/Qwen2_vlm /data1/aimlDevTeam/vlm /data1/aimlDevTeam/Qwen_API 2>/dev/null
```

Or view the folder tree structure up to 3 levels deep:

```bash
find /data1/aimlDevTeam/ -maxdepth 3 -type d 2>/dev/null
```

---

## 3. Verify Model Loading Capabilities

Test whether your local Qwen models load cleanly into PyTorch and vLLM without downloading anything from the internet:

### 3.1 Test Text Qwen Model Loading (Offline)
Replace `/PATH/TO/YOUR/TEXT_QWEN` with the path discovered from Section 1:

```bash
python -c "
from transformers import AutoTokenizer, AutoConfig
model_path = '/data1/aimlDevTeam/QWEN'  # Update if inside a subfolder
try:
    cfg = AutoConfig.from_pretrained(model_path, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    print(f'SUCCESS: Text Qwen Loaded! Architecture: {cfg.architectures}, Vocab: {tok.vocab_size}')
except Exception as e:
    print(f'Text Model Check Error: {e}')
"
```

### 3.2 Test Vision Qwen Model (VLM) Loading (Offline)
Replace `/PATH/TO/YOUR/VISION_QWEN` with the path discovered from Section 1:

```bash
python -c "
from transformers import AutoProcessor, AutoConfig
vl_path = '/data1/aimlDevTeam/Qwen2_vlm'  # Update if inside a subfolder
try:
    cfg = AutoConfig.from_pretrained(vl_path, local_files_only=True)
    proc = AutoProcessor.from_pretrained(vl_path, local_files_only=True)
    print(f'SUCCESS: Vision Qwen Loaded! Architecture: {cfg.architectures}')
except Exception as e:
    print(f'Vision Model Check Error: {e}')
"
```

---

## 4. How to Set the Paths in `.env`

Once you have identified the model paths from the scan above, update your `.env` file on your server:

```env
# Primary Text Model (For Task 7 Facts, Tasks 9-13 Q&A Generation, Task 15 Judge Evaluation)
VLLM_MODEL_NAME_OR_PATH=/path/to/your/discovered/text/qwen

# Vision-Language Model (For Task 4 Chart, Table, and Image Visual Understanding)
VLLM_VL_MODEL_NAME_OR_PATH=/path/to/your/discovered/vision/qwen

# Embedding Model (For Task 6 Topic Shifts & Task 8 Entity Linking)
EMBEDDING_MODEL_NAME_OR_PATH=/path/to/your/discovered/bge/model
```
