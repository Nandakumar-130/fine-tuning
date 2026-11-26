# Comprehensive Environment & Pipeline Check Guide

This document contains all individual test and verification commands to test whether every package, tool, GPU component, and model is installed and working on your Linux server.

---

## 1. One-Shot Master Verification (All 20 Packages)

Run this single command to check all packages from `requirements.txt` at once with their installed versions:

```bash
python -c "
import importlib

packages = [
    ('PyMuPDF (fitz)', 'fitz'),
    ('pypdf', 'pypdf'),
    ('opencv-python', 'cv2'),
    ('Pillow', 'PIL'),
    ('numpy', 'numpy'),
    ('paddleocr', 'paddleocr'),
    ('paddlepaddle', 'paddle'),
    ('symspellpy', 'symspellpy'),
    ('language-tool-python', 'language_tool_python'),
    ('python-dateutil', 'dateutil'),
    ('spacy', 'spacy'),
    ('sentence-transformers', 'sentence_transformers'),
    ('pydantic', 'pydantic'),
    ('pydantic-settings', 'pydantic_settings'),
    ('python-dotenv', 'dotenv'),
    ('vllm', 'vllm'),
    ('transformers', 'transformers'),
    ('accelerate', 'accelerate'),
    ('torch', 'torch'),
    ('tqdm', 'tqdm')
]

installed = []
not_installed = []

for name, mod in packages:
    try:
        m = importlib.import_module(mod)
        version = getattr(m, '__version__', 'installed')
        installed.append((name, version))
    except ImportError:
        not_installed.append(name)

print('=' * 60)
print(' INSTALLED PACKAGES:')
print('=' * 60)
for name, ver in installed:
    print(f'  [✓] {name:<28} (version: {ver})')

print('\n' + '=' * 60)
print(' NOT INSTALLED / MISSING PACKAGES:')
print('=' * 60)
if not not_installed:
    print('  All 20 packages are successfully installed!')
else:
    for name in not_installed:
        print(f'  [✗] {name}')
print('=' * 60)
"
```

---

## 2. Individual Package Functional Tests

Test each package individually to verify that it not only imports, but functions properly:

### 2.1 PDF & Document Processing

#### PyMuPDF (`fitz`) - Layout & Geometry
```bash
python -c "import fitz; doc = fitz.open('data_set_use.pdf'); print('PyMuPDF OK: Total Pages =', len(doc))"
```

#### pypdf - PDF Tools
```bash
python -c "import pypdf; print('pypdf OK: version =', pypdf.__version__)"
```

#### OpenCV (`cv2`) - Image Preprocessing
```bash
python -c "import cv2, numpy as np; img = np.zeros((100, 100, 3), dtype=np.uint8); gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); print('OpenCV OK: version =', cv2.__version__)"
```

#### Pillow (`PIL`) - Image Handling
```bash
python -c "from PIL import Image; img = Image.new('RGB', (50, 50), color='white'); print('Pillow OK: version =', Image.__version__)"
```

#### NumPy - Numerical Arrays
```bash
python -c "import numpy as np; a = np.array([1, 2, 3]); print('NumPy OK: version =', np.__version__)"
```

---

### 2.2 OCR & Scanned Text Processing

#### PaddlePaddle - Deep Learning Backend
```bash
python -c "import paddle; print('PaddlePaddle OK: version =', paddle.__version__, '| CUDA compiled =', paddle.is_compiled_with_cuda())"
```

#### PaddleOCR - Text Detection & Recognition
```bash
python -c "from paddleocr import PaddleOCR; print('PaddleOCR import OK')"
```

#### PP-Structure - Table Layout Analysis
```bash
python -c "from paddleocr import PPStructure; print('PP-Structure import OK')"
```

---

### 2.3 Quality Verification & Linguistic Tools

#### SymSpell - Spelling Validation
```bash
python -c "import symspellpy; print('SymSpell OK: version =', symspellpy.__version__)"
```

#### LanguageTool - Grammar Checker (Offline Check)
```bash
python -c "import language_tool_python; print('LanguageTool module OK: version =', language_tool_python.__version__)"
```

#### Python DateUtil - Calendar & Date Parsing
```bash
python -c "from dateutil import parser; d = parser.parse('2025-03-31'); print('DateUtil OK: parsed =', d)"
```

#### Decimal & Regex - Number & Monetary Validation
```bash
python -c "from decimal import Decimal; import re; val = Decimal('25000.50'); print('Decimal & Regex OK: val =', val)"
```

---

### 2.4 Text Structuring & Embeddings

#### spaCy - Sentence & Entity Parsing
```bash
python -c "import spacy; print('spaCy OK: version =', spacy.__version__)"
```

#### spaCy English Model (`en_core_web_sm`)
```bash
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); doc = nlp('Larsen and Toubro is located in Chennai.'); print('spaCy en_core_web_sm OK: entities =', [(ent.text, ent.label_) for ent in doc.ents])"
```

#### Sentence-Transformers - Embeddings Module
```bash
python -c "import sentence_transformers; print('SentenceTransformers OK: version =', sentence_transformers.__version__)"
```

#### Local BGE Embedding Model Test
```bash
python -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('BAAI/bge-small-en-v1.5', local_files_only=True); print('BGE Embedding Model OK (loaded from local cache)')"
```

---

### 2.5 Configuration & Settings

#### Pydantic & Pydantic-Settings
```bash
python -c "import pydantic, pydantic_settings; print('Pydantic OK: version =', pydantic.__version__)"
```

#### Python-Dotenv - Environment Loader
```bash
python -c "import dotenv; from src.config import settings; print('Config OK: PDF =', settings.PDF_FILE_PATH, '| Output =', settings.OUTPUT_DIR)"
```

---

### 2.6 GPU, Inference & Deep Learning Models

#### NVIDIA GPU Driver & VRAM (`nvidia-smi`)
```bash
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv
```

#### PyTorch & CUDA Functionality
```bash
python -c "import torch; print('PyTorch OK: version =', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'); print('VRAM Total (GB):', round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if torch.cuda.is_available() else 0)"
```

#### HuggingFace Transformers & Accelerate
```bash
python -c "import transformers, accelerate; print('Transformers OK: version =', transformers.__version__, '| Accelerate:', accelerate.__version__)"
```

#### vLLM - High-Throughput Inference Engine
```bash
python -c "import vllm; print('vLLM OK: version =', vllm.__version__)"
```

#### Qwen Model Local Cache Check
```bash
python -c "from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct', local_files_only=True); print('Qwen Tokenizer OK (loaded from offline local cache)')"
```

---

## 3. Server Disk & Model Storage Search

Find all Qwen and BGE model folders stored across server disks:

```bash
find /data1 /data2 /home -maxdepth 4 -type d \( -iname "*qwen*" -o -iname "*bge*" \) 2>/dev/null ; du -sh ~/.cache/huggingface/hub/*
```

---

## 4. Full Pipeline Offline Integration Test Suite

Run the master 34-case automated integration suite verifying every single stage:

```bash
python test_34_integration.py
```
