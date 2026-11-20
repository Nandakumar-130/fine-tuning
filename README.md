# PDF-to-Q&A Pipeline — Linux & CUDA Production Guide

A production-grade, 100% offline data engineering and dataset preparation pipeline implementing all specifications of the **PDF-to-Q&A Pipeline Guide**.

Optimized for **Linux (Ubuntu / Debian / RHEL) with CUDA and vLLM**. Managed 100% natively using **`uv`**.

---

## 🐧 Linux Quick Setup & Run Instructions

### 1. Prerequisites on Linux (System Packages)

Update your Linux system packages and ensure essential build and multimedia libraries are installed:

```bash
sudo apt-get update && sudo apt-get install -y \
    curl \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    default-jre
```
*(Note: `default-jre` is recommended for offline LanguageTool grammar audits, and `libgl1-mesa-glx` supports OpenCV/PaddleOCR image operations on headless servers).*

---

### 2. Install `uv` on Linux

If `uv` is not already installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

Verify installation:
```bash
uv --version
```

---

### 3. Clone Repository & Install Dependencies

```bash
git clone https://github.com/Nandakumar-130/fine-tuning.git
cd fine-tuning
```

Install all dependencies in one command using `uv sync`:

```bash
uv sync
```

**What this does on Linux:**
- Automatically creates an isolated virtual environment (`.venv`).
- Installs `vLLM` with high-throughput PagedAttention for NVIDIA GPUs.
- Installs `opencv-python-headless` (optimized for Linux servers without display servers).
- Installs `paddleocr`, `paddlepaddle`, `symspellpy`, `python-dateutil`, `sentence-transformers`, `transformers`, `torch`, and `tqdm`.

---

### 4. Configuration (`.env`)

Check or edit your `.env` file:

```ini
# Path to your target PDF
PDF_FILE_PATH=L&T_AR25.pdf

# Output directory for all generated records
OUTPUT_DIR=PDF_QA_PIPELINE

# Local Model Path or Hugging Face ID (Cached locally, 100% offline)
VLLM_MODEL_NAME_OR_PATH=Qwen/Qwen2.5-7B-Instruct
VLLM_VL_MODEL_NAME_OR_PATH=Qwen/Qwen2-VL-7B-Instruct
EMBEDDING_MODEL_NAME_OR_PATH=BAAI/bge-small-en-v1.5

# vLLM Engine Configuration on Linux/CUDA
USE_VLLM=True
VLLM_GPU_MEMORY_UTILIZATION=0.85
VLLM_MAX_MODEL_LEN=4096
TENSOR_PARALLEL_SIZE=1

# Chunking & Overlap settings
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# OCR & Quality Thresholds
OCR_LANG=en
USE_GPU_OCR=True
OCR_CONFIDENCE_ACCEPT=0.95
OCR_CONFIDENCE_REVIEW=0.80

# Q&A Limits per stage
MAX_QA_PER_PERSPECTIVE=10
BATCH_SIZE=4
TRAIN_VAL_SPLIT_RATIO=0.85
```

---

### 5. Run the Pipeline on Linux

Execute the pipeline using `uv run`:

```bash
uv run python main.py
```

#### Running as a Background Daemon on Linux (Optional):
For large documents where you want the pipeline to run in the background (surviving SSH disconnects):

```bash
nohup uv run python main.py > pipeline_run.log 2>&1 &
```

To monitor live progress:
```bash
tail -f pipeline_run.log
```

---

## 📺 Live Progress & Console Output

When running, the console displays real-time progress indicators:

```text
[Live Q&A Stage 1: Direct]: 100%|██████████| 12/12 [00:45<00:00, 3.8s/chunk]
  [LIVE Q&A #0001] [Direct] chunk_001 (Pages [1]): "Which division does L&T operate in Chennai?..."
  [LIVE Q&A #0002] [Direct] chunk_001 (Pages [1]): "What equipment does L&T manufacture for energy projects?..."
  [LIVE Q&A #0003] [Direct] chunk_001 (Pages [1]): "When was the facility established in Chennai?..."
```

Every generated question is streamed and saved **immediately** to disk as it is created.

---

## 🔄 Crash Resilience & Stopping/Resuming

- If you press `Ctrl+C` or the Linux server restarts during Q&A generation, **no progress is lost**.
- Progress is tracked in `PDF_QA_PIPELINE/logs/checkpoints.json`.
- When you run `uv run python main.py` again, the pipeline automatically detects completed chunks and **resumes from the exact chunk where it was interrupted**.

---

## 📋 Complete 17-Task Implementation Checklist

| # | Task from PDF | Output File Location | Status |
|---|---|---|---|
| **1** | Inventory PDF content types | `PDF_QA_PIPELINE/original/content_inventory.jsonl` | ✅ Completed |
| **2** | Extract text, OCR, tables, and images separately | `PDF_QA_PIPELINE/extracted/text_records.jsonl`<br>`PDF_QA_PIPELINE/extracted/image_records.jsonl`<br>`PDF_QA_PIPELINE/extracted/table_records.jsonl`<br>`PDF_QA_PIPELINE/extracted/images/`<br>`PDF_QA_PIPELINE/extracted/crops/` | ✅ Completed |
| **3** | Verify OCR, spelling, numbers, dates, order and tables | `PDF_QA_PIPELINE/verified/verified_records.jsonl`<br>*(SymSpell, Decimal, dateutil, coordinate retention)* | ✅ Completed |
| **4** | Describe images and verify text association | `PDF_QA_PIPELINE/extracted/image_records.jsonl`<br>*(Status: `verified`, `uncertain`, `not_related`)* | ✅ Completed |
| **5** | Create page-level records | `PDF_QA_PIPELINE/pages/page_records.jsonl` | ✅ Completed |
| **6** | Chunk and compare 400/600/800 token settings | `PDF_QA_PIPELINE/chunks/chunks.jsonl`<br>`PDF_QA_PIPELINE/chunks/chunk_experiments_comparison.json` | ✅ Completed |
| **7** | Extract and verify facts/entities | `PDF_QA_PIPELINE/facts/facts.jsonl` | ✅ Completed |
| **8** | Build entity index and cross-page groups | `PDF_QA_PIPELINE/relationships/entity_index.jsonl`<br>`PDF_QA_PIPELINE/relationships/related_chunk_groups.jsonl` | ✅ Completed |
| **9** | Generate Direct Q&A (Prompt A) | `PDF_QA_PIPELINE/qa/direct_qa.jsonl` | ✅ Completed |
| **10** | Generate Indirect from ALL Direct Q&A (Prompt B) | `PDF_QA_PIPELINE/qa/indirect_qa.jsonl` | ✅ Completed |
| **11** | Generate Attribute from ALL Indirect Q&A (Prompt C) | `PDF_QA_PIPELINE/qa/attribute_qa.jsonl` | ✅ Completed |
| **12** | Generate Combined from ALL Attribute Q&A (Prompt D) | `PDF_QA_PIPELINE/qa/combined_qa.jsonl` | ✅ Completed |
| **13** | Generate Cross-Page Q&A from verified groups (Prompt E) | `PDF_QA_PIPELINE/qa/cross_page_qa.jsonl` | ✅ Completed |
| **14** | Remove exact/normalized duplicates only | `PDF_QA_PIPELINE/quality/exact_duplicates.jsonl`<br>*(Conservative normalization: retains semantic variety)* | ✅ Completed |
| **15** | Evaluate against original evidence (Prompt F) | `PDF_QA_PIPELINE/quality/evaluation_results.jsonl`<br>*(Decisions: `PASS`, `REJECT`, `REVIEW`)* | ✅ Completed |
| **16** | Recover failures and audit samples (Prompt G) | `PDF_QA_PIPELINE/quality/rejected_qa.jsonl`<br>`PDF_QA_PIPELINE/logs/processing_logs.jsonl` | ✅ Completed |
| **17** | Export approved SFT data and split safely | `PDF_QA_PIPELINE/final/train.jsonl`<br>`PDF_QA_PIPELINE/final/validation.jsonl`<br>*(Grouped by fact to eliminate data leakage)* | ✅ Completed |

---

## 📂 Output Folder Structure

```
PDF_QA_PIPELINE/
├── original/
│   ├── source.pdf                      # Original input PDF
│   └── content_inventory.jsonl         # Task 1: Inventory of pages, blocks, drawings
├── extracted/
│   ├── text_records.jsonl              # Task 2: Native text blocks with coordinates
│   ├── image_records.jsonl             # Task 2 & 4: Image IDs, paths, captions & status
│   ├── table_records.jsonl             # Task 2: Tables with headers, rows & Markdown
│   ├── images/                         # Extracted embedded images
│   └── crops/                          # Low-confidence OCR crops for inspection
├── verified/
│   └── verified_records.jsonl          # Task 3: Audited text records (SymSpell & Decimal)
├── pages/
│   └── page_records.jsonl              # Task 5: Page-level unified records
├── chunks/
│   ├── chunks.jsonl                    # Task 6: Structure-aware chunks with overlap
│   └── chunk_experiments_comparison.json # Task 6: 400 vs 600 vs 800 token evaluation
├── facts/
│   └── facts.jsonl                     # Task 7: Atomic verified factual propositions
├── relationships/
│   ├── entity_index.jsonl              # Task 8: Entity references across pages
│   └── related_chunk_groups.jsonl      # Task 8: Cross-page multi-chunk evidence groups
├── qa/
│   ├── direct_qa.jsonl                 # Task 9: Stage 1 Direct Q&A (Prompt A)
│   ├── indirect_qa.jsonl               # Task 10: Stage 2 Indirect Q&A (Prompt B)
│   ├── attribute_qa.jsonl              # Task 11: Stage 3 Attribute Q&A (Prompt C)
│   ├── combined_qa.jsonl               # Task 12: Stage 4 Combined Multi-Fact Q&A (Prompt D)
│   └── cross_page_qa.jsonl             # Task 13: Stage 5 Cross-Page Q&A (Prompt E)
├── quality/
│   ├── exact_duplicates.jsonl          # Task 14: Deduplication log
│   ├── evaluation_results.jsonl        # Task 15: Independent Prompt F evaluation verdicts
│   ├── rejected_qa.jsonl               # Task 16: Rejected items & reason audit
│   └── quality_audit_report.json       # Comprehensive quality audit metrics
├── final/
│   ├── approved_qa.jsonl               # Full rich records for all approved Q&A
│   ├── train.jsonl                     # Task 17: Leak-free SFT train split
│   └── validation.jsonl                # Task 17: Leak-free SFT validation split
└── logs/
    ├── pipeline.log                    # Standard runtime console log
    ├── processing_logs.jsonl           # Live structured JSONL event log
    └── checkpoints.json                # Crash recovery checkpoint tracker
```
