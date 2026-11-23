# Execution & Run Commands Guide (Linux & Windows)

This document provides clear, step-by-step instructions to install all dependencies, configure models, run the pilot or production pipeline, and inspect outputs on both **Linux (Ubuntu / Debian / RHEL / WSL2)** and **Windows (PowerShell / Command Prompt)**.

---

## Part 1: Linux / CUDA Server (Recommended for vLLM & GPU)

### 1. System Package Prerequisites (Linux)

Open terminal and run:

```bash
sudo apt-get update && sudo apt-get install -y \
    curl \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    default-jre
```

### 2. Install `uv` Package Manager (Fastest & Recommended)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

### 3. Clone Repository & Install Python Packages

```bash
git clone https://github.com/Nandakumar-130/fine-tuning.git
cd fine-tuning
uv sync
```

*(Alternative using standard Python venv and pip):*

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pypdf pymupdf opencv-python-headless pillow numpy paddleocr paddlepaddle symspellpy python-dateutil language-tool-python spacy sentence-transformers pydantic pydantic-settings python-dotenv vllm transformers accelerate torch tqdm
```

### 4. Configure Environment (`.env`)

Verify or adjust your `.env` settings:

```bash
nano .env
```

Ensure models are set to:

```ini
PDF_FILE_PATH=data_set_use.pdf
OUTPUT_DIR=PDF_QA_PIPELINE

VLLM_MODEL_NAME_OR_PATH=Qwen/Qwen2.5-3B-Instruct
VLLM_VL_MODEL_NAME_OR_PATH=Qwen/Qwen2.5-VL-3B-Instruct
EMBEDDING_MODEL_NAME_OR_PATH=BAAI/bge-small-en-v1.5

USE_VLLM=True
VLLM_GPU_MEMORY_UTILIZATION=0.85
VLLM_MAX_MODEL_LEN=4096
TENSOR_PARALLEL_SIZE=1

CHUNK_SIZE=500
CHUNK_OVERLAP=50
MAX_QA_PER_PERSPECTIVE=10
TRAIN_VAL_SPLIT_RATIO=0.85
```

### 5. Run the Pipeline on Linux

#### A. Run Two-Page Pilot Test (Pages 1 & 2 only):

Tests extraction, Qwen2.5-VL vision descriptions, text Q&A cascades, and evaluations safely in `PILOT_QA_PIPELINE/`:

```bash
uv run python run_pilot.py
# Or with standard venv:
python run_pilot.py
```

#### B. Run Full Production Pipeline (All 738 Pages):

```bash
uv run python main.py
# Or with standard venv:
python main.py
```

#### C. Run in Background (Survives SSH Disconnect):

```bash
nohup uv run python main.py > pipeline_run.log 2>&1 &
tail -f pipeline_run.log
```

---

## Part 2: Windows 10 / 11 (PowerShell & CPU Fallback)

> **Note:** `vLLM` is designed natively for Linux with CUDA. On Windows, the pipeline automatically detects the absence of CUDA/vLLM and switches safely to the Hugging Face Transformers CPU pipeline.

### 1. Clone Repository (PowerShell)

```powershell
git clone https://github.com/Nandakumar-130/fine-tuning.git
cd fine-tuning
```

### 2. Create Virtual Environment & Install Packages

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install pypdf pymupdf opencv-python-headless pillow numpy symspellpy python-dateutil spacy sentence-transformers pydantic pydantic-settings python-dotenv transformers accelerate torch tqdm
```

### 3. Run on Windows

#### A. Run Two-Page Pilot Test:

```powershell
python run_pilot.py
```

#### B. Run Full Pipeline:

```powershell
python main.py
```

---

## Part 3: Inspecting Output Datasets & Logs

### Check Counts of Generated Records

- **On Linux:**

  ```bash
  # Total approved Q&A records
  wc -l PDF_QA_PIPELINE/final/approved_qa.jsonl

  # Train / Validation split counts
  wc -l PDF_QA_PIPELINE/final/train.jsonl
  wc -l PDF_QA_PIPELINE/final/validation.jsonl

  # Live event logs
  tail -n 25 PDF_QA_PIPELINE/logs/processing_logs.jsonl
  ```
- **On Windows (PowerShell):**

  ```powershell
  # Total approved Q&A records
  (Get-Content PDF_QA_PIPELINE\final\approved_qa.jsonl).Count

  # Train / Validation split counts
  (Get-Content PDF_QA_PIPELINE\final\train.jsonl).Count
  (Get-Content PDF_QA_PIPELINE\final\validation.jsonl).Count

  # Live event logs
  Get-Content PDF_QA_PIPELINE\logs\processing_logs.jsonl -Tail 25
  ```

---

## Part 4: Resuming Interrupted Runs

If a run is stopped or interrupted (e.g. system reboot, timeout, Ctrl+C):

- State is preserved in `logs/checkpoints.json`.
- Output records are flushed immediately to disk as they are generated.
- Simply run the command again:
  ```bash
  # Linux
  uv run python main.py

  # Windows
  python main.py
  ```

The pipeline automatically skips all completed stages and chunks, resuming from the exact point of interruption without duplicate processing.

---

## Part 5: Git Commands to Update and Push Changes

Whenever you modify configuration, prompts, or code:

```bash
git add .
git commit -m "Update pipeline configuration and run parameters"
git push origin main
```
