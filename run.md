# Execution & Run Commands Guide (Linux & Windows)

This document provides tested, production-ready commands to set up the environment, install dependencies, configure models, and run the pipeline on:
1. **Red Hat Enterprise Linux (RHEL) / CentOS / Rocky Linux / AlmaLinux / Fedora**
2. **Ubuntu / Debian / WSL2**
3. **Windows 10 / 11 (PowerShell)**

---

## Part 1: Red Hat Enterprise Linux (RHEL / CentOS / Rocky / AlmaLinux)

> **Corporate / Enterprise Network Friendly:** Uses standard system tools (`dnf`/`yum`) and Python `pip` with zero external shell curl scripts.

### 1. Install System Prerequisites

On Red Hat systems, use `dnf` (or `yum`):

```bash
sudo dnf install -y \
    curl \
    git \
    gcc \
    gcc-c++ \
    make \
    mesa-libGL \
    glib2 \
    glib2-devel \
    java-11-openjdk-headless \
    python3-devel
```
*(If your system uses `yum` instead of `dnf`, run `sudo yum install -y ...` with the same package list above).*

---

### 2. Create Virtual Environment & Upgrade Build Tools

```bash
# 1. Clone the repository (or pull latest changes)
git clone https://github.com/Nandakumar-130/fine-tuning.git
cd fine-tuning

# 2. Create the virtual environment
python3 -m venv .venv

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. CRITICAL: Upgrade pip, setuptools, and wheel
# (This ensures pre-compiled binary wheels are matched for Python 3.9)
pip install --upgrade pip setuptools wheel
```

---

### 3. Install Python Dependencies (Standard `pip`)

Run:

```bash
pip install -r requirements.txt
```

> **Why this works seamlessly:**  
> - Uses only standard PyPI (no blocked installer scripts).
> - `spacy` is pinned to `>=3.7.0,<3.8.0` (`spacy 3.7.5` and `thinc 8.2.5`), ensuring 100% pre-built binary wheels for Python 3.9 without source compilation or `thinc` errors.

*(Optional: If you want to use `uv`, install it via pip: `pip install uv`)*

---

### 4. Configure Environment (`.env`)

Check your configuration:

```bash
nano .env
```

Ensure parameters match your server hardware:

```ini
PDF_FILE_PATH=data_set_use.pdf
OUTPUT_DIR=PDF_QA_PIPELINE

VLLM_MODEL_NAME_OR_PATH=Qwen/Qwen2.5-3B-Instruct
VLLM_VL_MODEL_NAME_OR_PATH=Qwen/Qwen2.5-VL-3B-Instruct
EMBEDDING_MODEL_NAME_OR_PATH=BAAI/bge-small-en-v1.5

# Set USE_VLLM=True if CUDA GPU is available; False for CPU
USE_VLLM=True
VLLM_GPU_MEMORY_UTILIZATION=0.85
VLLM_MAX_MODEL_LEN=4096
TENSOR_PARALLEL_SIZE=1

CHUNK_SIZE=500
CHUNK_OVERLAP=50
MAX_QA_PER_PERSPECTIVE=10
TRAIN_VAL_SPLIT_RATIO=0.85
```

---

### 5. Run the Pipeline on Linux

#### A. Run Pilot Run (Sample Pages 7–8):
```bash
python run_pilot.py --start-page 7 --end-page 8 --output-dir pilot_sample
```

#### B. Run 40-Case Integration Test Suite:
```bash
python test_34_integration.py
```

#### C. Run Full Pipeline:
```bash
python main.py
```

#### D. Run in Background (Survives Terminal Close / SSH Disconnect):
```bash
nohup python main.py > pipeline_run.log 2>&1 &
tail -f pipeline_run.log
```

---

## Part 2: Ubuntu / Debian / WSL2

```bash
# 1. System packages
sudo apt-get update && sudo apt-get install -y curl git build-essential libgl1-mesa-glx libglib2.0-0 default-jre python3-dev

# 2. Virtual environment & packages
cd fine-tuning
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# 3. Run pilot or full pipeline
python run_pilot.py --start-page 7 --end-page 8 --output-dir pilot_sample
python main.py
```

---

## Part 3: Windows 10 / 11 (PowerShell)

```powershell
# 1. Clone & enter repository
git clone https://github.com/Nandakumar-130/fine-tuning.git
cd fine-tuning

# 2. Virtual environment & packages
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# 3. Run pilot test or integration test
python run_pilot.py --start-page 7 --end-page 8 --output-dir pilot_sample
python test_34_integration.py

# 4. Run full pipeline
python main.py
```

---

## Part 4: Inspecting Generated Datasets & Outputs

### On Linux:
```bash
# Extracted multimodal components
ls -lh pilot_sample/extracted_data/

# Processed chunks & SFT datasets
ls -lh pilot_sample/processed_data/

# Check line counts of final datasets
wc -l pilot_sample/final/approved_qa.jsonl
wc -l pilot_sample/final/train.jsonl
wc -l pilot_sample/final/validation.jsonl

# View Quality Audit Summary
cat pilot_sample/quality/quality_audit_report.json
```

### On Windows (PowerShell):
```powershell
Get-ChildItem pilot_sample\processed_data, pilot_sample\extracted_data
(Get-Content pilot_sample\final\approved_qa.jsonl).Count
```

---

## Part 5: Troubleshooting Guide

| Issue | Cause | Fix |
| :--- | :--- | :--- |
| `sudo: apt-get: command not found` | Red Hat/CentOS uses `dnf` or `yum`. | Use `sudo dnf install -y ...` (or `yum`). |
| `curl: (22) 403 Forbidden` on astral.sh | Corporate network blocks installer script. | Use standard `pip install -r requirements.txt` (or `pip install uv`). |
| `bash: ~/.local/bin/env: No such file` | Failed curl script never created the file. | Activate `.venv` directly with `source .venv/bin/activate`. |
| `Failed to build spacy` / `thinc<8.4.0` | Python 3.9 on unpinned `spacy 3.8+`. | Run `pip install -U pip setuptools wheel` then `pip install -r requirements.txt`. |
| Process killed upon SSH disconnect | Foreground process terminated. | Run with `nohup python main.py > pipeline_run.log 2>&1 &`. |
