# Fast Targeted Model Discovery Guide

This document provides **ultra-fast targeted commands** (completing in 2–5 seconds) to locate **Qwen models**, **Llama models**, and **Vision/Embedding models** on your server without slow full-disk crawling.

---

## 1. Quick Qwen Models Search (Takes 2 Seconds)

Search specifically for all **Qwen text and vision models** inside `/data1/aimlDevTeam` and `/data1`:

```bash
python -c "
import os
from pathlib import Path

target_dir = '/data1'
print('=' * 70)
print(' SCANNING FOR QWEN MODELS (FAST TARGETED SEARCH)...')
print('=' * 70)

found = []
for root, dirs, files in os.walk(target_dir):
    # Only check directories with qwen in their name
    if 'qwen' in root.lower():
        if any(f.endswith('.safetensors') or f.endswith('.bin') or f == 'config.json' for f in files):
            try:
                sz = round(sum(os.path.getsize(os.path.join(root, f)) for f in files if os.path.isfile(os.path.join(root, f))) / (1024**3), 2)
            except Exception:
                sz = 0.0
            
            mtype = 'Vision VLM' if ('vl' in root.lower() or 'vision' in root.lower()) else 'Text LLM'
            found.append((root, mtype, sz))

if not found:
    print('No Qwen model weights found in', target_dir)
else:
    for idx, (path, mtype, sz) in enumerate(found, 1):
        print(f'[{idx:02d}] Type: {mtype:<15} | Size: {sz:>6.2f} GB')
        print(f'     Path: {path}\n')
print('=' * 70)
"
```

---

## 2. Quick Llama Models Search (Takes 2 Seconds)

Search specifically for all **Llama models** (Llama-2, Llama-3, Llama-3.1, Llama-3.2, etc.):

```bash
python -c "
import os
from pathlib import Path

target_dir = '/data1'
print('=' * 70)
print(' SCANNING FOR LLAMA MODELS (FAST TARGETED SEARCH)...')
print('=' * 70)

found = []
for root, dirs, files in os.walk(target_dir):
    # Only check directories with llama in their name
    if 'llama' in root.lower():
        if any(f.endswith('.safetensors') or f.endswith('.bin') or f.endswith('.gguf') or f == 'config.json' for f in files):
            try:
                sz = round(sum(os.path.getsize(os.path.join(root, f)) for f in files if os.path.isfile(os.path.join(root, f))) / (1024**3), 2)
            except Exception:
                sz = 0.0
            
            mtype = 'Vision VLM' if ('vision' in root.lower() or 'vl' in root.lower()) else 'Text LLM'
            found.append((root, mtype, sz))

if not found:
    print('No Llama model weights found in', target_dir)
else:
    for idx, (path, mtype, sz) in enumerate(found, 1):
        print(f'[{idx:02d}] Type: {mtype:<15} | Size: {sz:>6.2f} GB')
        print(f'     Path: {path}\n')
print('=' * 70)
"
```

---

## 3. Instant Linux Shell Check (Bash `ls` & `find`)

Run these two instant terminal commands to see what is stored in `/data1/aimlDevTeam`:

### A. List Qwen folders:
```bash
find /data1/aimlDevTeam -maxdepth 3 -iname "*qwen*" 2>/dev/null
```

### B. List Llama folders:
```bash
find /data1/aimlDevTeam -maxdepth 3 -iname "*llama*" 2>/dev/null
```

---

## 4. How to Put the Discovered Model in `.env`

Once you get the output path, copy it into `.env`:

```env
# Text Model Path (For Q&A generation & evaluation)
VLLM_MODEL_NAME_OR_PATH=/data1/aimlDevTeam/YOUR_QWEN_OR_LLAMA_PATH

# Vision Model Path (For Charts & Images)
VLLM_VL_MODEL_NAME_OR_PATH=/data1/aimlDevTeam/YOUR_VISION_QWEN_PATH
```
