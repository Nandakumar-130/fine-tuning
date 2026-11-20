# PDF-to-Q&A Pipeline — Complete Qwen-Based Dataset Preparation and Fine-Tuning Guide

A complete, production-grade, 100% offline data engineering pipeline implementing all specifications of the **PDF-to-Q&A Pipeline Guide**. 

It handles content inventory, extraction (native text, OCR, tables, images), quality checks (SymSpell, LanguageTool, Decimal, Dateutil), hybrid chunking, fact/entity extraction, cross-page relationship detection, 5-stage sequential perspective Q&A generation (with live console progress and immediate streaming to disk), independent evaluation, and leak-free SFT export.

The project is managed 100% natively using **`uv`**.

---

## 🚀 Quick Start Guide (Install & Run)

### Prerequisites
- Python 3.10, 3.11, or 3.12
- `uv` installed ([Install uv guide](https://docs.astral.sh/uv/getting-started/installation/))
  - On Windows (PowerShell):
    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```
  - On Linux / macOS:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

---

### Step 1: Install All Dependencies with `uv`

Run a single command in your terminal / PowerShell:

```powershell
uv sync
```

**What this does automatically:**
- Creates a local virtual environment (`.venv`).
- Resolves all 330 dependencies locked in `uv.lock`.
- Installs PyMuPDF, OpenCV, PaddleOCR, SymSpell, LanguageTool, dateutil, SentenceTransformers, vLLM / Transformers, PyTorch, Pydantic, and tqdm.
- No manual `pip` or virtual environment creation needed.

---

### Step 2: Configure Environment Variables (`.env`)

The project is pre-configured with [.env](file:///c:/Users/Soundar/Documents/data/.env). You can adjust any parameter:

```ini
# 1. Target PDF file path
PDF_FILE_PATH=PDF-to-Q&A Pipeline — Complete Qwen-Based Dataset Preparation and Fine-Tuning Guide.pdf

# 2. Output folder name or full path where all generated files will be stored
OUTPUT_DIR=PDF_QA_PIPELINE

# 3. Model identifier or path to local downloaded weights
VLLM_MODEL_NAME_OR_PATH=Qwen/Qwen2.5-7B-Instruct
VLLM_VL_MODEL_NAME_OR_PATH=Qwen/Qwen2-VL-7B-Instruct
EMBEDDING_MODEL_NAME_OR_PATH=BAAI/bge-small-en-v1.5

# 4. Inference Engine Settings
USE_VLLM=True
VLLM_GPU_MEMORY_UTILIZATION=0.85
VLLM_MAX_MODEL_LEN=4096
TENSOR_PARALLEL_SIZE=1

# 5. Chunking & Overlap
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# 6. OCR Settings
OCR_LANG=en
USE_GPU_OCR=False
OCR_CONFIDENCE_ACCEPT=0.95
OCR_CONFIDENCE_REVIEW=0.80

# 7. Q&A Generation Settings
MAX_QA_PER_PERSPECTIVE=10
BATCH_SIZE=4
TRAIN_VAL_SPLIT_RATIO=0.85
```

---

### Step 3: Run the Complete Pipeline

Run the pipeline using:

```powershell
uv run python main.py
```

---

## 📺 Live Progress & Console Output

While the code runs, you will see a live progress bar and real-time question generation in your console:

```text
[Live Q&A Stage 1: Direct]: 100%|██████████| 12/12 [00:45<00:00, 3.8s/chunk]
  [LIVE Q&A #0001] [Direct] chunk_001 (Pages [1]): "Which division does L&T operate in Chennai?..."
  [LIVE Q&A #0002] [Direct] chunk_001 (Pages [1]): "What equipment does L&T manufacture for energy projects?..."
  [LIVE Q&A #0003] [Direct] chunk_001 (Pages [1]): "When was the facility established in Chennai?..."
```

Every single question is streamed and saved **immediately** to disk as it is generated.

---

## 🔄 Crash Resilience & Stopping/Resuming

- If you press `Ctrl+C` or turn off your computer during Q&A generation, **you will not lose any progress**.
- The checkpoint tracker saves the state to `PDF_QA_PIPELINE/logs/checkpoints.json`.
- When you run `uv run python main.py` again, the pipeline automatically detects completed chunks and **resumes from the exact chunk where it was interrupted**.

---

## 📋 Complete 17-Task Implementation Checklist

All 17 tasks from the PDF guide have been fully implemented and verified:

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

## 📂 Detailed Output Folder Structure

Every stage stores its data into its own dedicated file inside the `OUTPUT_DIR`:

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

---

## 🎯 Fine-Tuning Format Output (`final/train.jsonl` & `final/validation.jsonl`)

The dataset is formatted for direct fine-tuning with Qwen models:

```json
{
  "id": "qa_dir_chunk_001_00",
  "messages": [
    {
      "role": "user",
      "content": "Which division does L&T operate in Chennai?"
    },
    {
      "role": "assistant",
      "content": "L&T operates a heavy engineering division in Chennai."
    }
  ],
  "metadata": {
    "perspective": "direct",
    "source_chunk_ids": ["chunk_001"],
    "source_pages": [1],
    "fact_ids": ["fact_chunk_001_00"]
  }
}
```
