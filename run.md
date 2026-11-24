# Execution & Run Commands Guide (Red Hat Linux)

This guide provides the exact commands to set up, install dependencies, resolve common server repository/network issues, and run the PDF-to-Q&A dataset preparation pipeline on your **Red Hat Enterprise Linux (RHEL / CentOS / Rocky Linux / AlmaLinux)** server.

---

## 1. Clean Git State & Pull Latest Code

If you encounter **"Applying autostash resulted in conflicts"** or local branch divergence, reset cleanly to the remote branch before pulling:

```bash
# 1. Reset any local changes/conflicts
git reset --hard HEAD
git stash drop 2>/dev/null || true

# 2. Pull the latest clean code from GitHub
git pull origin main
```

---

## 2. System Package Prerequisites (Fix for Docker 403 Error)

If your corporate firewall/proxy blocks `download.docker.com` (giving `Status code: 403 for repomd.xml - Failed to download metadata for repo 'docker-ce-stable'`), disable that repository during install using `--disablerepo=docker-ce-stable`:

```bash
# Install core build libraries, OpenGL, GLib, and Java (disabling blocked corporate/docker repo)
sudo dnf install -y \
    --disablerepo=docker-ce-stable \
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

*(Note: If your system uses `yum` instead of `dnf`, run):*
```bash
sudo yum install -y --disablerepo=docker-ce-stable curl git gcc gcc-c++ make mesa-libGL glib2 glib2-devel java-11-openjdk-headless python3-devel
```

*(Optional: Permanently disable the blocked Docker repository so it never interrupts dnf again):*
```bash
sudo dnf config-manager --set-disabled docker-ce-stable 2>/dev/null || true
```

---

## 3. Virtual Environment Setup

Always create a dedicated virtual environment and upgrade the packaging tools first:

```bash
# 1. Create the virtual environment (if not already created)
python3 -m venv .venv

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. CRITICAL: Upgrade pip, setuptools, and wheel
# (Required on Python 3.9 so pre-built binary wheels match properly without source compilation)
pip install --upgrade pip setuptools wheel
```

---

## 4. Install Python Dependencies

Install all dependencies directly from PyPI using standard `pip`:

```bash
pip install -r requirements.txt
```

> **Why this works seamlessly:**
> - Completely avoids external script/curl 403 errors (no `astral.sh` or Docker needed).
> - `spacy` is pinned to `>=3.7.0,<3.8.0`, which provides pre-compiled `manylinux` binary wheels for Python 3.9 with zero C++ compilation or `thinc` build errors.

---

## 5. Verify Configuration (`.env`)

Check your `.env` configuration file:

```bash
cat .env
```

Ensure the key settings match your environment:

```ini
PDF_FILE_PATH=data_set_use.pdf
OUTPUT_DIR=PDF_QA_PIPELINE

VLLM_MODEL_NAME_OR_PATH=Qwen/Qwen2.5-3B-Instruct
VLLM_VL_MODEL_NAME_OR_PATH=Qwen/Qwen2.5-VL-3B-Instruct
EMBEDDING_MODEL_NAME_OR_PATH=BAAI/bge-small-en-v1.5

# Set to True if NVIDIA CUDA GPU is available; False if CPU only
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

## 6. Execution Commands

Make sure your virtual environment is active before running any command:
```bash
source .venv/bin/activate
```

### A. Run Two-Page Pilot Test (Safe Verification on Pages 7–8)
Verifies text, KPI cards, charts, 5 chunking strategies, Q&A cascades, deduplication, and SFT export:

```bash
python run_pilot.py --start-page 7 --end-page 8 --output-dir pilot_sample
```

### B. Run the 40-Case Integration Test Suite
Verifies all modules, schemas, Qwen-as-a-Judge evaluation, and file outputs:

```bash
python test_34_integration.py
```

### C. Run the Full Production Pipeline
Processes the entire document with automatic checkpoint saving and recovery:

```bash
python main.py
```

### D. Run Full Pipeline in Background (Survives SSH Disconnect)
To let the pipeline run in the background if your terminal closes:

```bash
nohup python main.py > pipeline_run.log 2>&1 &
```

To monitor live progress:
```bash
tail -f pipeline_run.log
```

---

## 7. Inspecting Generated Outputs

After running the pilot or full pipeline, check the outputs:

```bash
# 1. Extracted Multimodal Data (paragraphs, tables, charts, KPI boxes, images, relationships)
ls -lh pilot_sample/extracted_data/

# 2. Processed Chunks & ChatML SFT Datasets
ls -lh pilot_sample/processed_data/

# 3. View record counts in final datasets
wc -l pilot_sample/final/approved_qa.jsonl
wc -l pilot_sample/final/train.jsonl
wc -l pilot_sample/final/validation.jsonl

# 4. View the Quality Audit Report
cat pilot_sample/quality/quality_audit_report.json
```

---

## 8. Troubleshooting Quick Reference

| Issue | Cause | Fix |
| :--- | :--- | :--- |
| `Status code: 403 ... docker-ce-stable` | Corporate firewall blocks Docker repo metadata. | Use `sudo dnf install -y --disablerepo=docker-ce-stable ...`. |
| `Applying autostash resulted in conflicts` | Local changes collided with remote updates during pull. | Run `git reset --hard HEAD` and `git stash drop`, then `git pull origin main`. |
| `sudo: apt-get: command not found` | Red Hat uses `dnf` or `yum`. | Use `sudo dnf install -y ...` (or `yum`). |
| `curl: (22) error: 403` on astral.sh | Firewall/proxy blocked the script. | Do not use curl; use standard `pip install -r requirements.txt`. |
| `bash: ~/.local/bin/env: No such file` | The curl script never ran. | Activate the venv directly with `source .venv/bin/activate`. |
| `Failed to build spacy` / `thinc` | Python 3.9 on unpinned `spacy`. | Run `pip install -U pip setuptools wheel` then `pip install -r requirements.txt`. |
| Terminal closes during long run | SSH session timed out. | Run with `nohup python main.py > pipeline_run.log 2>&1 &`. |
