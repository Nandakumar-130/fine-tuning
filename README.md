# PDF-to-Q&A Generation & SFT Dataset Preparation Pipeline

A production-grade, 100% offline data engineering and dataset preparation pipeline implementing all specifications of the **PDF-to-Q&A Pipeline Guide**.

Designed to process corporate annual reports and technical documentation (e.g., **Larsen & Toubro Annual Review 2025**, 738 pages) and synthesize a leak-free, verified, multi-perspective instruction dataset for fine-tuning **`HuggingFaceTB/SmolLM2-135M-Instruct`** using **`Qwen/Qwen2.5-3B-Instruct`** and **`Qwen/Qwen2.5-VL-3B-Instruct`** on **Linux / CUDA with vLLM**.

---

## 🚀 Model Architecture & Roles

The pipeline integrates two specialized models using a resource-conscious sequential loading architecture:

### 1. Primary Text LLM: `Qwen/Qwen2.5-3B-Instruct`
- **Execution Backend:** `vLLM` (with high-throughput PagedAttention and structured JSON generation).
- **Initialization Lifecycle:** Loaded **ONCE** into memory when inference begins (Task 7) and reused across all tasks without reloading weights.
- **Responsibilities:**
  - **Task 7:** Atomic fact & entity proposition extraction (subject-relation-object triples).
  - **Task 8:** Cross-page entity indexing and relationship detection.
  - **Tasks 9–13:** Five sequential Q&A perspectives (Direct, Indirect, Attribute, Combined, Cross-Page).
  - **Task 15:** Independent Qwen-as-a-Judge evidence evaluation (Prompt F).
  - **Task 16:** Rejection feedback & evidence-grounded answer regeneration.
  - **Task 16:** JSON schema & syntax repair (Prompt G).

### 2. Vision-Language Model: `Qwen/Qwen2.5-VL-3B-Instruct`
- **Execution Backend:** `vLLM` / Hugging Face Transformers.
- **Initialization Lifecycle:** Loaded **only** during Task 4 (Image Understanding) and unloaded immediately (`unload()` + `torch.cuda.empty_cache()`) before the text model loads.
- **Responsibilities:**
  - **Task 4:** Visual comprehension of extracted PDF charts, financial diagrams, equipment photographs, and figure labels.
  - Extracting quantitative values, axis titles, and metric legends from images.
  - Verifying image-to-text association with surrounding paragraphs.

### 3. Target Fine-Tuning Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- The pipeline formats approved Q&A records into standard ChatML conversation format:
  ```json
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
  ```
- Splits datasets 85% train / 15% validation clustered by source fact and chunk ID to eliminate data leakage.

---

## 🛠️ Complete 17-Task Pipeline Workflow

```
[Task 1] Content Inventory Scan ──────► Page types, text density, OCR necessity
         │
[Task 2] PDF Extraction ──────────────► Native text, pdfplumber tables, image extraction
         │
[Task 3] Quality Verification ────────► SymSpell, Decimal & dateutil validation
         │
[Task 4] Vision Understanding ────────► Qwen2.5-VL-3B-Instruct describes charts/diagrams (then unloads)
         │
[Task 5] Page Organization ───────────► Unified PageRecord assembly
         │
[Task 6] Tokenizer & Chunking ────────► Qwen2.5-3B offline tokenizer, density audit, 400/600/800 experiments
         │
[Init]   Text LLM Load Once ──────────► Qwen2.5-3B-Instruct loaded via vLLM
         │
[Task 7] Fact Extraction ─────────────► Subject-relation-object propositions
         │
[Task 8] Entity Linking ──────────────► Cross-page multi-chunk evidence groups
         │
[Tasks 9-13] Five Q&A Perspectives ───► Direct -> Indirect -> Attribute -> Combined -> Cross-Page
         │
[Task 14] Deduplication ──────────────► NFKC normalization (preserves valid paraphrases)
         │
[Task 15] Qwen-as-a-Judge ─────────────► Prompt F strict evidence verification (PASS/REJECT/REVIEW)
         │
[Task 16] Rejection & Repair ─────────► Prompt F feedback regeneration & Prompt G syntax repair
         │
[Task 17] Leak-Free SFT Export ───────► ChatML format split 85/15 by fact cluster
```

---

## ⚙️ Hardware Requirements & Environment Setup

- **Operating System:** Linux (Ubuntu 22.04+ / Debian 12 / RHEL 9) or WSL2 on Windows.
- **GPU Requirements:** NVIDIA GPU with CUDA support (Compute Capability $\ge 8.0$ recommended).
- **VRAM Requirements:**
  - `Qwen2.5-3B-Instruct` (BF16): ~6 GB VRAM.
  - `Qwen2.5-VL-3B-Instruct` (BF16): ~6 GB VRAM.
  - Context & KV Cache Buffer: ~2–4 GB.
  - **Recommended GPU:** 1x NVIDIA RTX 4080 (16 GB), RTX 3090/4090 (24 GB), or A10G/A100.
- **Pre-Flight Hardware Check:** The pipeline automatically inspects CUDA availability and available VRAM via `check_gpu_resources()`, logging warnings if free memory is constrained.

---

## 📦 Linux / CUDA Installation with `uv`

### 1. System Dependencies
```bash
sudo apt-get update && sudo apt-get install -y \
    curl git build-essential libgl1-mesa-glx libglib2.0-0 default-jre
```

### 2. Install `uv` & Clone Repository
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

git clone https://github.com/Nandakumar-130/fine-tuning.git
cd fine-tuning
uv sync
```

### 3. Environment Configuration (`.env`)
```ini
PDF_FILE_PATH=L&T_AR25.pdf
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

---

## 🧪 Separate Testing and Production Execution

### Option A: Run the Two-Page Pilot (`run_pilot.py`)
To verify extraction, image understanding, Q&A perspective cascades, evaluation, and SFT export before running the full 738-page document:

```bash
uv run python run_pilot.py
```
- **Scope:** Processes only pages 1 and 2 of `L&T_AR25.pdf`.
- **Safety:** Saves all records to `PILOT_QA_PIPELINE/`. Never touches or overwrites production data.

### Option B: Run the Complete Production Pipeline (`main.py`)
To process the full 738-page document:

```bash
uv run python main.py
```

#### Running as a Background Daemon (surviving SSH disconnects):
```bash
nohup uv run python main.py > pipeline_run.log 2>&1 &
tail -f pipeline_run.log
```

---

## 🔄 Checkpointing & Crash Resumption

- All live generated Q&A records, evaluation decisions, and processing events are flushed immediately to disk (`f.flush()`).
- Stage-level and chunk-level completion states are tracked in `logs/checkpoints.json`.
- If execution is interrupted (e.g., server reboot or manual stop), rerun `uv run python main.py`. The pipeline automatically skips completed stages and chunks, resuming seamlessly from the exact interrupted item.

---

## 📂 Output Folder Structure

```
PDF_QA_PIPELINE/  (or PILOT_QA_PIPELINE/ for pilot)
├── original/
│   └── content_inventory.jsonl         # Task 1: Document layout inventory
├── extracted/
│   ├── text_records.jsonl              # Task 2: Native text blocks with bounding boxes
│   ├── table_records.jsonl             # Task 2: Financial tables with Markdown format
│   ├── image_records.jsonl             # Task 2 & 4: Image paths, descriptions & status
│   └── images/                         # Extracted image files
├── verified/
│   └── verified_records.jsonl          # Task 3: Spell-checked & verified text
├── pages/
│   └── page_records.jsonl              # Task 5: Page-level unified records
├── chunks/
│   ├── chunks.jsonl                    # Task 6: Structure-aware chunks
│   ├── preflight_density_audit.json    # Task 6: Token density & financial page audit
│   └── chunk_experiments_comparison.json # Task 6: 400 vs 600 vs 800 token evaluation
├── facts/
│   └── facts.jsonl                     # Task 7: Verified atomic propositions
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
│   ├── evaluation_results.jsonl        # Task 15: Prompt F judge verdicts (PASS/REJECT/REVIEW)
│   ├── rejected_qa.jsonl               # Task 16: Rejected items & failure reasons
│   └── quality_audit_report.json       # Pipeline quality metrics
└── final/
    ├── approved_qa.jsonl               # Full rich records for all approved Q&A
    ├── train.jsonl                     # Task 17: Leak-free SFT train split (85%)
    └── validation.jsonl                # Task 17: Leak-free SFT validation split (15%)
```
