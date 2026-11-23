# Execution & Run Commands Guide (Red Hat / Linux & Windows)

This guide provides tested, step-by-step commands to set up the environment, install dependencies, configure models, and execute the PDF-to-Q&A pipeline on:
- **Red Hat Enterprise Linux (RHEL) / CentOS / Rocky Linux / AlmaLinux / Fedora**
- **Ubuntu / Debian / WSL2**
- **Windows 10 / 11 (PowerShell)**

---

## Part 1: Red Hat Enterprise Linux (RHEL / CentOS / Rocky / AlmaLinux)

### 1. System Package Prerequisites (RHEL / CentOS / Rocky)

On Red Hat systems, use `dnf` (or `yum` on older versions). Do **not** use `apt-get`.

```bash
# Update and install system build libraries, OpenGL, GLib, and Java (for LanguageTool)
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

# (If your system uses yum instead of dnf, run):
# sudo yum install -y curl git gcc gcc-c++ make mesa-libGL glib2 glib2-devel java-11-openjdk-headless python3-devel
```

*(Note for Ubuntu / Debian users only: `sudo apt-get update && sudo apt-get install -y curl git build-essential libgl1-mesa-glx libglib2.0-0 default-jre`)*

---

### 2. Python Virtual Environment Setup

Always create a clean virtual environment and upgrade `pip`, `setuptools`, and `wheel` first. This prevents the `Failed to build spacy / thinc` wheel resolution error on Python 3.9.

```bash
# 1. Clone the repository
git clone https://github.com/Nandakumar-130/fine-tuning.git
cd fine-tuning

# 2. Create and activate a clean virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. CRITICAL: Upgrade pip, setuptools, and wheel
pip install --upgrade pip setuptools wheel
```

---

### 3. Install Python Dependencies

#### Method A: Using `pip install -r requirements.txt` (Most Reliable & Recommended)

This avoids network 403 Forbidden issues from external installer scripts and uses pinned pre-built binary wheels:

```bash
pip install -r requirements.txt
```

> **Why did `spacy` fail earlier?**  
> On Python 3.9 (`cp39`), unconstrained `pip install spacy` attempted to build `spacy>=3.8` and `thinc>=8.3.12` from source. `requirements.txt` is pinned to `spacy>=3.7.0,<3.8.0`, which provides pre-compiled `manylinux` wheels for Python 3.9 instantly with zero source compilation.

#### Method B: Using `uv` Package Manager

If you prefer `uv`, install it safely via `pip` (bypasses `curl https://astral.sh/uv/install.sh` 403 blocks):

```bash
pip install uv
uv sync
```

---

### 4. Configure Environment (`.env`)

Verify your `.env` configuration file:

```bash
nano .env
```

Ensure the key parameters match your hardware:

```ini
PDF_FILE_PATH=data_set_use.pdf
OUTPUT_DIR=PDF_QA_PIPELINE

VLLM_MODEL_NAME_OR_PATH=Qwen/Qwen2.5-3B-Instruct
VLLM_VL_MODEL_NAME_OR_PATH=Qwen/Qwen2.5-VL-3B-Instruct
EMBEDDING_MODEL_NAME_OR_PATH=BAAI/bge-small-en-v1.5

# Set USE_VLLM=True if CUDA GPU is available; False for CPU fallback
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

#### A. Run the Two-Page Pilot Test First (Safe Verification):

Runs extraction, vision checks, 5 chunking strategies, Q&A cascades, deduplication, and SFT export on pages 7–8:

```bash
python run_pilot.py --start-page 7 --end-page 8 --output-dir pilot_sample
# Or if using uv:
# uv run python run_pilot.py --start-page 7 --end-page 8 --output-dir pilot_sample
```

#### B. Run the 40-Case Integration Test Suite:

```bash
python test_34_integration.py
```

#### C. Run the Full Production Pipeline:

```bash
python main.py
# Or if using uv:
# uv run python main.py
```

#### D. Run in Background (Survives SSH Disconnect):

```bash
nohup python main.py > pipeline_run.log 2>&1 &
tail -f pipeline_run.log
```

---

## Part 2: Windows 10 / 11 (PowerShell)

### 1. Clone & Set Up Environment (PowerShell)

```powershell
git clone https://github.com/Nandakumar-130/fine-tuning.git
cd fine-tuning

python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 2. Run Commands on Windows

```powershell
# 1. Run Integration Test Suite
python test_34_integration.py

# 2. Run Pilot Test on pages 7-8
python run_pilot.py --start-page 7 --end-page 8 --output-dir pilot_sample

# 3. Run Full Pipeline
python main.py
```

---

## Part 3: Inspecting Outputs & Datasets

### Check Generated Datasets

#### On Linux:
```bash
# 1. Extracted Multimodal Data
ls -lh pilot_sample/extracted_data/

# 2. Processed Chunks & SFT Datasets
ls -lh pilot_sample/processed_data/

# 3. View Line Counts of Approved Q&A and SFT splits
wc -l pilot_sample/final/approved_qa.jsonl
wc -l pilot_sample/final/train.jsonl
wc -l pilot_sample/final/validation.jsonl

# 4. View Quality Audit Summary
cat pilot_sample/quality/quality_audit_report.json
```

#### On Windows (PowerShell):
```powershell
# Check generated files
Get-ChildItem pilot_sample\processed_data, pilot_sample\extracted_data

# Check approved Q&A count
(Get-Content pilot_sample\final\approved_qa.jsonl).Count
```

---

## Part 4: Troubleshooting Common Issues

| Error | Root Cause | Solution |
| :--- | :--- | :--- |
| `sudo: apt-get: command not found` | System is RHEL / CentOS / Rocky / Fedora (not Ubuntu/Debian). | Use `sudo dnf install -y ...` or `sudo yum install -y ...`. |
| `curl: (22) error: 403` on `astral.sh` | Network or firewall blocks the installer script. | Install uv directly via pip: `pip install uv` or use `pip install -r requirements.txt`. |
| `Could not find version for thinc<8.4.0,>=8.3.12` / `Failed to build spacy` | Python 3.9 with unpinned `spacy 3.8+` attempting source build. | Run `pip install --upgrade pip setuptools wheel` then `pip install -r requirements.txt` (uses `spacy>=3.7.0,<3.8.0`). |
| `bash: ~/.local/bin/env: No such file or directory` | Failed curl command never created the environment script. | Use `source .venv/bin/activate` inside your standard virtual environment. |
