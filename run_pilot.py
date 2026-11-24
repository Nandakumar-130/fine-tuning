"""
Two-Page Pilot Pipeline Execution Script
Executes the identical extraction, chunking, Qwen2.5-VL-3B-Instruct (vision),
Qwen2.5-3B-Instruct (text inference), and evaluation workflow on exactly 2 pages.
Saves all outputs to PILOT_QA_PIPELINE without altering production outputs or checkpoints.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

# Load .env variables
load_dotenv(ROOT_DIR / ".env")

from src.config import settings
from src.schemas import (
    TextBlockRecord, TableRecord, ImageRecord, VerifiedRecord,
    PageRecord, ChunkRecord, FactRecord, EntityIndexRecord,
    RelatedChunkGroup, RichQARecord, KPIBoxRecord, ChartRecord, ContentGroupRecord
)
from src.io_utils import ensure_directory, read_jsonl, write_jsonl, CheckpointTracker, StructuredLogger
from src.inventory_scanner import ContentInventoryScanner
from src.pdf_extractor import PDFExtractor
from src.quality_verifier import ExtractionQualityVerifier
from src.image_association import ImageAssociationManager
from src.relationship_mapper import RelationshipMapper
from src.page_organizer import PageOrganizer
from src.chunker import HybridChunker
from src.vllm_engine import VLLMClient, VLLMVisionClient, get_offline_qwen_tokenizer, check_gpu_resources
from src.fact_extractor import FactEntityExtractor
from src.qa_generator import SequentialPerspectiveQAGenerator
from src.deduplicator import QADeduplicator
from src.evaluator import IndependentEvaluator
from src.exporter import FineTuningExporter
from src.auditor import QualityAuditor

def setup_pilot_logger(output_dir: Path):
    logs_dir = ensure_directory(output_dir / "logs")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(logs_dir / "pilot_pipeline.log", encoding="utf-8")
        ]
    )

def run_pilot(page_start: int = 1, page_end: int = 2, output_dir: Optional[str] = None):
    pilot_dir = Path(output_dir) if output_dir else ROOT_DIR / "PILOT_QA_PIPELINE"
    ensure_directory(pilot_dir)
    setup_pilot_logger(pilot_dir)
    logger = logging.getLogger("PilotRunner")
    logger.info(f"=== STARTING TWO-PAGE PILOT RUN (Pages {page_start} to {page_end}) ===")
    logger.info(f"Pilot Output Directory: {pilot_dir}")

    # Inspect hardware
    hw_info = check_gpu_resources(required_vram_gb=8.0)
    logger.info(f"Pre-Flight Hardware Check: {hw_info}")

    pdf_path = ROOT_DIR / settings.PDF_FILE_PATH
    if not pdf_path.exists():
        logger.error(f"Target PDF not found: {pdf_path}")
        return

    tracker = CheckpointTracker(pilot_dir / "logs" / "pilot_checkpoints.json")
    slogger = StructuredLogger(pilot_dir / "logs")

    # 1. Content Inventory Scanner
    logger.info(">>> Pilot Task 1: Inventory Scan...")
    scanner = ContentInventoryScanner(pdf_path, output_dir=pilot_dir)
    inventory = scanner.scan_document(page_range=(page_start, page_end))

    # 2. PDF Extraction (Filtered to pilot pages)
    logger.info(f">>> Pilot Task 2: Extracting Native Text, Tables, and Images from pages {page_start}-{page_end}...")
    extractor = PDFExtractor(pdf_path, output_dir=pilot_dir)
    
    # Extract only pilot pages
    raw_text, images, tables = extractor.extract_all(page_range=(page_start, page_end))
    text_records = [t for t in raw_text if page_start <= t.page_number <= page_end]
    table_records = [tb for tb in tables if page_start <= tb.page_number <= page_end]
    image_records = [im for im in images if page_start <= im.page_number <= page_end]
    logger.info(f"Extracted {len(text_records)} text blocks, {len(table_records)} tables, {len(image_records)} images for pilot.")

    # 3. Quality Verification
    logger.info(">>> Pilot Task 3: Quality Verification...")
    verifier = ExtractionQualityVerifier(output_dir=pilot_dir)
    verified_records = verifier.verify_all(text_records)

    # 4. Image Understanding via Qwen2.5-VL-3B-Instruct
    logger.info(">>> Pilot Task 4: Vision Model Association...")
    image_manager = ImageAssociationManager(output_dir=pilot_dir)
    vision_client = None
    if image_records and settings.VLLM_VL_MODEL_NAME_OR_PATH:
        try:
            logger.info("Initializing Qwen2.5-VL-3B-Instruct for pilot visual understanding...")
            vision_client = VLLMVisionClient(
                model_name_or_path=settings.VLLM_VL_MODEL_NAME_OR_PATH,
                use_vllm=settings.USE_VLLM
            )
        except Exception as e:
            logger.warning(f"Could not load Vision model for pilot ({e}). Using captions fallback.")
            vision_client = None

    image_records = image_manager.associate_images_with_text(image_records, text_records, vllm_client=vision_client)
    if vision_client:
        vision_client.unload()
        vision_client = None

    # 5. Page Organization
    logger.info(">>> Pilot Task 5: Page Organization...")
    organizer = PageOrganizer(output_dir=pilot_dir)
    page_records = organizer.assemble_pages(
        text_records=text_records,
        tables=table_records,
        images=image_records,
        verified_records=verified_records
    )

    # 5.5 Relationship Mapping & Content Groups (Phase 5)
    logger.info(">>> Pilot Task 5.5: Relationship Mapping & Content Groups...")
    mapper = RelationshipMapper(output_dir=pilot_dir)
    relationships, content_groups = mapper.map_relationships(
        paragraphs=getattr(extractor, "all_paragraph_records", []),
        tables=table_records,
        charts=getattr(extractor, "all_chart_records", []),
        kpis=getattr(extractor, "all_kpi_records", []),
        images=image_records
    )

    # 6. Structure-Aware Chunking & Tokenizer (5 Strategies)
    logger.info(">>> Pilot Task 6: Tokenizer & 5-Strategy Hybrid Chunking...")
    exact_qwen_tokenizer = get_offline_qwen_tokenizer(settings.VLLM_MODEL_NAME_OR_PATH)
    chunker = HybridChunker(
        output_dir=pilot_dir,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        tokenizer=exact_qwen_tokenizer,
        embedding_model_name=settings.EMBEDDING_MODEL_NAME_OR_PATH,
        use_embeddings=False
    )
    chunks = chunker.chunk_document(
        page_records,
        table_records,
        image_records,
        kpi_boxes=getattr(extractor, "all_kpi_records", []),
        charts=getattr(extractor, "all_chart_records", []),
        content_groups=content_groups
    )
    logger.info(f"Formed {len(chunks)} chunks for pilot.")

    # 7. Initialize Qwen2.5-3B-Instruct text LLM (LOAD ONCE)
    logger.info(">>> Initializing Qwen2.5-3B-Instruct for Pilot Inference Stages...")
    vllm_client = VLLMClient(
        model_name_or_path=settings.VLLM_MODEL_NAME_OR_PATH,
        use_vllm=settings.USE_VLLM,
        gpu_memory_utilization=settings.VLLM_GPU_MEMORY_UTILIZATION,
        max_model_len=settings.VLLM_MAX_MODEL_LEN,
        tensor_parallel_size=settings.TENSOR_PARALLEL_SIZE
    )

    # 8. Fact Extraction & Entity Linking
    logger.info(">>> Pilot Task 7: Fact & Entity Extraction...")
    fact_extractor = FactEntityExtractor(output_dir=pilot_dir, vllm_client=vllm_client)
    facts = fact_extractor.run_fact_extraction(chunks)
    entity_index, related_groups = fact_extractor.build_cross_page_relationships(facts, chunks)

    # 9. Q&A Generation (All 5 Perspectives)
    logger.info(">>> Pilot Tasks 9-13: Five-Perspective Q&A Generation...")
    qa_gen = SequentialPerspectiveQAGenerator(
        output_dir=pilot_dir,
        vllm_client=vllm_client,
        max_qa_per_perspective=settings.MAX_QA_PER_PERSPECTIVE,
        checkpoint_tracker=tracker,
        structured_logger=slogger
    )

    direct_qa = qa_gen.generate_stage1_direct(chunks)
    indirect_qa = qa_gen.generate_stage2_indirect(chunks, direct_qa)
    attribute_qa = qa_gen.generate_stage3_attribute(chunks, indirect_qa)
    combined_qa = qa_gen.generate_stage4_combined(chunks, attribute_qa)
    all_cross_groups = related_groups + content_groups
    cross_page_qa = qa_gen.generate_stage5_cross_page(all_cross_groups, chunks=chunks)

    all_raw_qa = direct_qa + indirect_qa + attribute_qa + combined_qa + cross_page_qa
    logger.info(f"Total raw Q&A generated in pilot: {len(all_raw_qa)}")
    write_jsonl(pilot_dir / "processed_data" / "generated_qa.jsonl", all_raw_qa)

    # 10. Deduplication
    logger.info(">>> Pilot Task 14: Deduplication...")
    deduper = QADeduplicator(output_dir=pilot_dir)
    unique_qa, duplicates = deduper.deduplicate(all_raw_qa)
    logger.info(f"Unique Q&A: {len(unique_qa)} | Duplicates removed: {len(duplicates)}")

    # 11. Independent Evaluation (Prompt F + Feedback)
    logger.info(">>> Pilot Task 15-16: Independent Qwen-as-a-Judge Evaluation & Feedback...")
    evaluator = IndependentEvaluator(output_dir=pilot_dir, vllm_client=vllm_client)
    passed_qa, rejected_qa, reviewed_qa = evaluator.evaluate_all(unique_qa, chunks)

    # 12. Fine-Tuning Export
    logger.info(">>> Pilot Task 17: Exporting to ChatML SFT Format...")
    exporter = FineTuningExporter(output_dir=pilot_dir, train_ratio=settings.TRAIN_VAL_SPLIT_RATIO)
    train_sft, val_sft = exporter.export_and_split(passed_qa)

    # 13. Quality Audit Report
    auditor = QualityAuditor(output_dir=pilot_dir)
    auditor.generate_quality_report()

    logger.info("=== TWO-PAGE PILOT EXECUTION COMPLETED SUCCESSFULLY ===")
    logger.info(f"Approved Q&A pairs: {len(passed_qa)}")
    logger.info(f"Train conversations: {len(train_sft)} -> {pilot_dir / 'final' / 'train.jsonl'}")
    logger.info(f"Validation conversations: {len(val_sft)} -> {pilot_dir / 'final' / 'validation.jsonl'}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pilot PDF-to-Q&A Pipeline Run")
    parser.add_argument("--start-page", type=int, default=1, help="Start page number (1-indexed)")
    parser.add_argument("--end-page", type=int, default=2, help="End page number (1-indexed)")
    parser.add_argument("--output-dir", type=str, default="PILOT_QA_PIPELINE", help="Pilot output directory")
    args = parser.parse_args()

    run_pilot(page_start=args.start_page, page_end=args.end_page, output_dir=args.output_dir)
