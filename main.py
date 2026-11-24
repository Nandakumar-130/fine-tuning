import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

# Load .env variables
load_dotenv(ROOT_DIR / ".env")

from src.config import settings
from src.schemas import (
    TextBlockRecord, TableRecord, ImageRecord, VerifiedRecord,
    PageRecord, ChunkRecord, FactRecord, EntityIndexRecord,
    RelatedChunkGroup, RichQARecord, ContentGroupRecord, RelationshipRecord
)
from src.io_utils import ensure_directory, read_jsonl, write_jsonl, CheckpointTracker, StructuredLogger
from src.inventory_scanner import ContentInventoryScanner
from src.pdf_extractor import PDFExtractor
from src.quality_verifier import ExtractionQualityVerifier
from src.image_association import ImageAssociationManager
from src.relationship_mapper import RelationshipMapper
from src.page_organizer import PageOrganizer
from src.chunker import HybridChunker
from src.vllm_engine import VLLMClient, VLLMVisionClient, get_offline_qwen_tokenizer
from src.fact_extractor import FactEntityExtractor
from src.qa_generator import SequentialPerspectiveQAGenerator
from src.deduplicator import QADeduplicator
from src.evaluator import IndependentEvaluator
from src.exporter import FineTuningExporter
from src.auditor import QualityAuditor

def setup_logger(output_dir: Path):
    logs_dir = ensure_directory(output_dir / "logs")
    log_file = logs_dir / "pipeline.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("MainPipeline")

def main():
    pdf_path = Path(settings.PDF_FILE_PATH)
    output_dir = Path(settings.OUTPUT_DIR)
    ensure_directory(output_dir)

    logger = setup_logger(output_dir)
    slogger = StructuredLogger(output_dir / "logs")

    logger.info("=" * 80)
    logger.info("PRODUCTION PDF-to-Q&A DATASET PREPARATION PIPELINE")
    logger.info(f"Target PDF File: {pdf_path.resolve()}")
    logger.info(f"Output Directory: {output_dir.resolve()}")
    logger.info(f"Inference Model: {settings.VLLM_MODEL_NAME_OR_PATH} (vLLM={settings.USE_VLLM})")
    logger.info("=" * 80)

    slogger.log_event("INFO", "init", f"Pipeline started for {pdf_path.name}")

    # Checkpoint tracker for crash resilience (resume from where it stopped)
    checkpoint_file = output_dir / "logs" / "checkpoints.json"
    tracker = CheckpointTracker(checkpoint_file)

    try:
        # -----------------------------------------------------------------------
        # Task 1: Survey & Inventory PDF Content Types (Checklist Item 1)
        # -----------------------------------------------------------------------
        inventory_scanner = ContentInventoryScanner(pdf_path=pdf_path, output_dir=output_dir)
        inventory_file = output_dir / "original" / "content_inventory.jsonl"
        if tracker.is_stage_done("inventory") and inventory_file.exists():
            logger.info("Task 1 (Inventory) already completed. Loading existing inventory.")
            inventory = read_jsonl(inventory_file)
        else:
            logger.info(">>> Task 1: Surveying PDF Content Types...")
            slogger.log_event("INFO", "inventory", "Starting Task 1 content survey")
            inventory = inventory_scanner.scan_inventory()
            tracker.mark_stage_done("inventory")
            slogger.log_event("INFO", "inventory", f"Scanned {len(inventory)} pages")

        # -----------------------------------------------------------------------
        # Task 2: Extract Text, OCR (OpenCV Preprocessing), Tables, Images
        # -----------------------------------------------------------------------
        extractor = PDFExtractor(
            pdf_path=pdf_path,
            output_dir=output_dir,
            ocr_lang=settings.OCR_LANG,
            use_gpu_ocr=settings.USE_GPU_OCR
        )
        text_records_file = output_dir / "extracted" / "text_records.jsonl"
        if tracker.is_stage_done("extraction") and text_records_file.exists():
            logger.info("Task 2 (Extraction) already completed. Loading existing records.")
            text_records = read_jsonl(text_records_file, TextBlockRecord)
            image_records = read_jsonl(output_dir / "extracted" / "image_records.jsonl", ImageRecord)
            table_records = read_jsonl(output_dir / "extracted" / "table_records.jsonl", TableRecord)
        else:
            logger.info(">>> Task 2: Extracting Native Text, Tables, Images, and OCR with OpenCV...")
            slogger.log_event("INFO", "extraction", "Starting extraction with OpenCV and PaddleOCR")
            text_records, image_records, table_records = extractor.extract_all()
            tracker.mark_stage_done("extraction")
            slogger.log_event("INFO", "extraction", f"Extracted {len(text_records)} text blocks, {len(image_records)} images, {len(table_records)} tables")

        # -----------------------------------------------------------------------
        # Task 3: Extraction Quality Verification (SymSpell, LanguageTool, Decimal, Dateutil)
        # -----------------------------------------------------------------------
        verifier = ExtractionQualityVerifier(
            output_dir=output_dir,
            ocr_accept=settings.OCR_CONFIDENCE_ACCEPT,
            ocr_review=settings.OCR_CONFIDENCE_REVIEW
        )
        verified_records_file = output_dir / "verified" / "verified_records.jsonl"
        if tracker.is_stage_done("quality_verification") and verified_records_file.exists():
            logger.info("Task 3 (Quality Verification) already completed. Loading verified records.")
            verified_records = read_jsonl(verified_records_file, VerifiedRecord)
        else:
            logger.info(">>> Task 3: Running Extraction Quality Verification (SymSpell, Decimal & dateutil)...")
            slogger.log_event("INFO", "verification", "Running SymSpell, Decimal and dateutil checks")
            verified_records = verifier.verify_all(text_records)
            tracker.mark_stage_done("quality_verification")
            slogger.log_event("INFO", "verification", f"Verified {len(verified_records)} text records")

        # -----------------------------------------------------------------------
        # Task 4: Describe Images and Verify Text Association
        # -----------------------------------------------------------------------
        logger.info(">>> Task 4: Associating Images with Captions & Surrounding Text...")
        image_manager = ImageAssociationManager(output_dir=output_dir)
        
        # Load Vision Model ONLY if image records exist and need visual understanding
        vision_client = None
        images_needing_desc = [img for img in image_records if not img.description]
        if images_needing_desc and settings.VLLM_VL_MODEL_NAME_OR_PATH:
            try:
                logger.info(f"Initializing Qwen2.5-VL-3B-Instruct for visual image understanding ({len(images_needing_desc)} images)...")
                vision_client = VLLMVisionClient(
                    model_name_or_path=settings.VLLM_VL_MODEL_NAME_OR_PATH,
                    use_vllm=settings.USE_VLLM
                )
            except Exception as e:
                logger.warning(f"Could not initialize Vision Client ({e}). Continuing with rule-based captions.")
                vision_client = None

        image_records = image_manager.associate_images_with_text(image_records, text_records, vllm_client=vision_client)

        # Unload Vision model immediately to free GPU memory for text model
        if vision_client is not None:
            vision_client.unload()
            vision_client = None

        # -----------------------------------------------------------------------
        # Task 5: Assemble Unified Page-Level Records
        # -----------------------------------------------------------------------
        organizer = PageOrganizer(output_dir=output_dir)
        page_records_file = output_dir / "pages" / "page_records.jsonl"
        if tracker.is_stage_done("page_organization") and page_records_file.exists():
            logger.info("Task 5 (Page Organization) already completed. Loading page records.")
            page_records = read_jsonl(page_records_file, PageRecord)
        else:
            logger.info(">>> Task 5: Assembling Page-Level Records with Verified Text Propagation...")
            page_records = organizer.assemble_pages(
                text_records=text_records,
                tables=table_records,
                images=image_records,
                verified_records=verified_records
            )
            tracker.mark_stage_done("page_organization")

        # -----------------------------------------------------------------------
        # Task 5.5: Relationship Mapping & Content Groups (Phase 5)
        # -----------------------------------------------------------------------
        logger.info(">>> Task 5.5: Mapping Content Relationships & Creating Content Groups...")
        mapper = RelationshipMapper(output_dir=output_dir)
        relationships, content_groups = mapper.map_relationships(
            paragraphs=getattr(extractor, "all_paragraph_records", []),
            tables=table_records,
            charts=getattr(extractor, "all_chart_records", []),
            kpis=getattr(extractor, "all_kpi_records", []),
            images=image_records
        )
        slogger.log_event("INFO", "relationships", f"Mapped {len(relationships)} relationships into {len(content_groups)} content groups")

        # -----------------------------------------------------------------------
        # Independent Qwen Tokenizer Loading (Offline Only - ZERO Weight Loading)
        # -----------------------------------------------------------------------
        logger.info("Attempting independent offline load of exact Qwen tokenizer...")
        exact_qwen_tokenizer = get_offline_qwen_tokenizer(settings.VLLM_MODEL_NAME_OR_PATH)
        if exact_qwen_tokenizer is not None:
            logger.info("Exact Qwen tokenizer successfully attached for structure-aware chunking.")
        else:
            logger.warning(
                "Exact Qwen tokenizer not found in local cache; continuing with documented conservative character-based estimation fallback. "
                "No external tokenizer will be silently substituted."
            )

        # -----------------------------------------------------------------------
        # Task 6: Structure-Aware Hybrid Chunking & Experiments (400, 600, 800 tokens)
        # -----------------------------------------------------------------------
        chunker = HybridChunker(
            output_dir=output_dir,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            tokenizer=exact_qwen_tokenizer,
            embedding_model_name=settings.EMBEDDING_MODEL_NAME_OR_PATH
        )
        chunks_file = output_dir / "chunks" / "chunks.jsonl"
        if tracker.is_stage_done("chunking") and chunks_file.exists():
            logger.info("Task 6 (Chunking) already completed. Loading chunks from disk.")
            chunks = read_jsonl(chunks_file, ChunkRecord)
        else:
            logger.info(">>> Task 6: Structure-Aware Chunking with SentenceTransformers Topic Boundary...")
            slogger.log_event("INFO", "chunking", "Starting hybrid chunking and 400/600/800 token experiments")
            chunks = chunker.chunk_document(
                page_records,
                table_records,
                image_records,
                kpi_boxes=getattr(extractor, "all_kpi_records", []),
                charts=getattr(extractor, "all_chart_records", []),
                content_groups=content_groups
            )
            tracker.mark_stage_done("chunking")
            slogger.log_event("INFO", "chunking", f"Formed {len(chunks)} coherent chunks")

        # -----------------------------------------------------------------------
        # Initialize Local Inference Engine (LOAD ONCE WHEN INFERENCE IS ABOUT TO BEGIN)
        # -----------------------------------------------------------------------
        logger.info("Initializing Local Inference Engine (LOAD ONCE for inference stages)...")
        vllm_client = VLLMClient(
            model_name_or_path=settings.VLLM_MODEL_NAME_OR_PATH,
            use_vllm=settings.USE_VLLM,
            gpu_memory_utilization=settings.VLLM_GPU_MEMORY_UTILIZATION,
            max_model_len=settings.VLLM_MAX_MODEL_LEN,
            tensor_parallel_size=settings.TENSOR_PARALLEL_SIZE
        )

        # -----------------------------------------------------------------------
        # Task 7: Fact & Entity Extraction
        # -----------------------------------------------------------------------
        fact_extractor = FactEntityExtractor(output_dir=output_dir, vllm_client=vllm_client)
        facts_file = output_dir / "facts" / "facts.jsonl"
        if tracker.is_stage_done("fact_extraction") and facts_file.exists():
            logger.info("Task 7 (Fact Extraction) already completed. Loading facts from disk.")
            facts = read_jsonl(facts_file, FactRecord)
        else:
            logger.info(">>> Task 7: Proposing & Validating Atomic Facts from Chunks...")
            slogger.log_event("INFO", "fact_extraction", "Extracting atomic factual propositions")
            facts = fact_extractor.run_fact_extraction(chunks)
            tracker.mark_stage_done("fact_extraction")
            slogger.log_event("INFO", "fact_extraction", f"Extracted {len(facts)} verified facts")

        # -----------------------------------------------------------------------
        # Task 8: Entity Index & Cross-Page Relationship Groups
        # -----------------------------------------------------------------------
        groups_file = output_dir / "relationships" / "related_chunk_groups.jsonl"
        if tracker.is_stage_done("entity_linking") and groups_file.exists():
            logger.info("Task 8 (Entity Linking) already completed. Loading cross-page groups.")
            entity_index = read_jsonl(output_dir / "relationships" / "entity_index.jsonl", EntityIndexRecord)
            related_groups = read_jsonl(groups_file, RelatedChunkGroup)
        else:
            logger.info(">>> Task 8: Building Entity Index & Cross-Page Groups...")
            slogger.log_event("INFO", "entity_linking", "Building cross-page multi-chunk evidence groups")
            entity_index, related_groups = fact_extractor.build_cross_page_relationships(facts, chunks)
            tracker.mark_stage_done("entity_linking")
            slogger.log_event("INFO", "entity_linking", f"Linked {len(entity_index)} entities across pages")

        # -----------------------------------------------------------------------
        # Sequential Perspective Q&A Generation (At least 10 Q&A per chunk per perspective)
        # Order: Direct -> Indirect -> Attribute -> Combined -> Cross-page
        # -----------------------------------------------------------------------
        qa_gen = SequentialPerspectiveQAGenerator(
            output_dir=output_dir,
            vllm_client=vllm_client,
            max_qa_per_perspective=settings.MAX_QA_PER_PERSPECTIVE,
            checkpoint_tracker=tracker,
            structured_logger=slogger
        )

        # Stage 1: Direct Q&A (Prompt A)
        direct_qa_file = output_dir / "qa" / "direct_qa.jsonl"
        if tracker.is_stage_done("direct_qa") and direct_qa_file.exists():
            logger.info("Task 9 (Direct Q&A) already completed. Loading direct Q&A.")
            direct_qa = read_jsonl(direct_qa_file, RichQARecord)
        else:
            logger.info(">>> Task 9: Generating Stage 1 Direct Q&A (Prompt A)...")
            direct_qa = qa_gen.generate_stage1_direct(chunks)
            tracker.mark_stage_done("direct_qa")

        # Stage 2: Indirect Q&A (Prompt B) from ALL Direct Q&A
        indirect_qa_file = output_dir / "qa" / "indirect_qa.jsonl"
        if tracker.is_stage_done("indirect_qa") and indirect_qa_file.exists():
            logger.info("Task 10 (Indirect Q&A) already completed. Loading indirect Q&A.")
            indirect_qa = read_jsonl(indirect_qa_file, RichQARecord)
        else:
            logger.info(">>> Task 10: Generating Stage 2 Indirect Q&A (Prompt B)...")
            indirect_qa = qa_gen.generate_stage2_indirect(chunks, direct_qa)
            tracker.mark_stage_done("indirect_qa")

        # Stage 3: Attribute Q&A (Prompt C) from ALL Indirect Q&A
        attribute_qa_file = output_dir / "qa" / "attribute_qa.jsonl"
        if tracker.is_stage_done("attribute_qa") and attribute_qa_file.exists():
            logger.info("Task 11 (Attribute Q&A) already completed. Loading attribute Q&A.")
            attribute_qa = read_jsonl(attribute_qa_file, RichQARecord)
        else:
            logger.info(">>> Task 11: Generating Stage 3 Attribute Q&A (Prompt C)...")
            attribute_qa = qa_gen.generate_stage3_attribute(chunks, indirect_qa)
            tracker.mark_stage_done("attribute_qa")

        # Stage 4: Combined Q&A (Prompt D) from ALL Attribute Q&A
        combined_qa_file = output_dir / "qa" / "combined_qa.jsonl"
        if tracker.is_stage_done("combined_qa") and combined_qa_file.exists():
            logger.info("Task 12 (Combined Q&A) already completed. Loading combined Q&A.")
            combined_qa = read_jsonl(combined_qa_file, RichQARecord)
        else:
            logger.info(">>> Task 12: Generating Stage 4 Combined Multi-Fact Q&A (Prompt D)...")
            combined_qa = qa_gen.generate_stage4_combined(chunks, attribute_qa)
            tracker.mark_stage_done("combined_qa")

        # Stage 5: Cross-Page Q&A (Prompt E)
        cross_page_qa_file = output_dir / "qa" / "cross_page_qa.jsonl"
        if tracker.is_stage_done("cross_page_qa") and cross_page_qa_file.exists():
            logger.info("Task 13 (Cross-Page Q&A) already completed. Loading cross-page Q&A.")
            cross_page_qa = read_jsonl(cross_page_qa_file, RichQARecord)
        else:
            logger.info(">>> Task 13: Generating Stage 5 Cross-Page Q&A (Prompt E)...")
            all_cross_groups = related_groups + content_groups
            cross_page_qa = qa_gen.generate_stage5_cross_page(all_cross_groups, chunks=chunks)
            tracker.mark_stage_done("cross_page_qa")

        all_generated_qa: List[RichQARecord] = direct_qa + indirect_qa + attribute_qa + combined_qa + cross_page_qa
        logger.info(f"Total raw Q&A generated across 5 perspectives: {len(all_generated_qa)}")
        slogger.log_event("INFO", "qa_generation", f"Generated total {len(all_generated_qa)} Q&A pairs")
        write_jsonl(output_dir / "processed_data" / "generated_qa.jsonl", all_generated_qa)

        # -----------------------------------------------------------------------
        # Task 14: Deduplication (Exact & Normalized duplicates only; keeps semantic variety)
        # -----------------------------------------------------------------------
        logger.info(">>> Task 14: Deduplicating Exact & Normalized Questions...")
        deduplicator = QADeduplicator(output_dir=output_dir)
        unique_qa, duplicates = deduplicator.deduplicate(all_generated_qa)
        slogger.log_event("INFO", "deduplication", f"Retained {len(unique_qa)} unique Q&A; removed {len(duplicates)} exact duplicates")

        # -----------------------------------------------------------------------
        # Task 15 & 16: Independent Evidence Evaluation & Error Recovery (Prompt F & G)
        # -----------------------------------------------------------------------
        logger.info(">>> Tasks 15 & 16: Independent Evaluation (Prompt F) & Error Recovery...")
        evaluator = IndependentEvaluator(output_dir=output_dir, vllm_client=vllm_client)
        approved_qa, rejected_qa, review_qa = evaluator.evaluate_all(unique_qa, chunks)
        slogger.log_event("INFO", "evaluation", f"Evaluated: {len(approved_qa)} PASS, {len(rejected_qa)} REJECT, {len(review_qa)} REVIEW")

        # -----------------------------------------------------------------------
        # Task 17: Fine-Tuning SFT Export (Leak-Free Split by Fact/Chunk)
        # -----------------------------------------------------------------------
        logger.info(">>> Task 17: Exporting SFT Training & Validation Datasets...")
        exporter = FineTuningExporter(output_dir=output_dir, train_ratio=settings.TRAIN_VAL_SPLIT_RATIO)
        train_sft, val_sft = exporter.export_and_split(approved_qa)
        slogger.log_event("INFO", "export", f"Exported {len(train_sft)} train and {len(val_sft)} val samples")

        # -----------------------------------------------------------------------
        # Quality Audit Report
        # -----------------------------------------------------------------------
        auditor = QualityAuditor(output_dir=output_dir)
        report = auditor.generate_report()
        slogger.log_event("INFO", "completion", "Pipeline completed successfully", report)

        logger.info("=" * 80)
        logger.info("ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!")
        logger.info(f"Total Pages Processed: {report['pdf_pages_processed']}")
        logger.info(f"Total Chunks Created: {report['total_chunks_created']}")
        logger.info(f"Approved Q&A Total: {report['approved_qa_total']}")
        logger.info(f"Train Dataset Samples: {report['fine_tuning_export_splits']['train_samples']}")
        logger.info(f"Validation Dataset Samples: {report['fine_tuning_export_splits']['validation_samples']}")
        logger.info(f"Structured logs written to: {output_dir / 'logs' / 'processing_logs.jsonl'}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Fatal pipeline error: {e}", exc_info=True)
        slogger.log_event("FATAL", "pipeline", f"Unhandled pipeline error: {e}")
        raise

if __name__ == "__main__":
    main()
