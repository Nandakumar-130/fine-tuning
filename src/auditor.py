import json
import logging
from pathlib import Path
from typing import Dict, Any, List

from src.io_utils import ensure_directory, read_jsonl

logger = logging.getLogger("QualityReporter")

class QualityAuditor:
    """
    Step 13: Quality Report before Fine-Tuning
    PDF Requirements (Page 10 & 12):
    Report:
    - PDF pages processed and pages represented in Q&A.
    - OCR warnings, corrected values and unresolved items.
    - Total chunks, failed chunks and verified facts.
    - Q&A counts by Direct, Indirect, Attribute, Combined and Cross-page.
    - Exact/normalized duplicates removed.
    - PASS/REVIEW/REJECT counts.
    - Q&A with valid source chunk and page references.
    - Approved examples and train/validation counts.
    """
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.quality_dir = self.output_dir / "quality"
        self.logs_dir = self.output_dir / "logs"
        ensure_directory(self.quality_dir)
        ensure_directory(self.logs_dir)

    def generate_report(self) -> Dict[str, Any]:
        logger.info("Compiling comprehensive quality audit report...")

        # Read counts from artifacts
        pages_recs = read_jsonl(self.output_dir / "pages" / "page_records.jsonl")
        verified_recs = read_jsonl(self.output_dir / "verified" / "verified_records.jsonl")
        chunks = read_jsonl(self.output_dir / "chunks" / "chunks.jsonl")
        facts = read_jsonl(self.output_dir / "facts" / "facts.jsonl")

        direct_qa = read_jsonl(self.output_dir / "qa" / "direct_qa.jsonl")
        indirect_qa = read_jsonl(self.output_dir / "qa" / "indirect_qa.jsonl")
        attribute_qa = read_jsonl(self.output_dir / "qa" / "attribute_qa.jsonl")
        combined_qa = read_jsonl(self.output_dir / "qa" / "combined_qa.jsonl")
        cross_qa = read_jsonl(self.output_dir / "qa" / "cross_page_qa.jsonl")

        duplicates = read_jsonl(self.quality_dir / "exact_duplicates.jsonl")
        eval_results = read_jsonl(self.quality_dir / "evaluation_results.jsonl")
        approved_qa = read_jsonl(self.output_dir / "final" / "approved_qa.jsonl")
        rejected_qa = read_jsonl(self.quality_dir / "rejected_qa.jsonl")

        train_sft = read_jsonl(self.output_dir / "final" / "train.jsonl")
        val_sft = read_jsonl(self.output_dir / "final" / "validation.jsonl")

        # Calculations
        ocr_warnings = sum(len(r.get("warnings", [])) for r in verified_recs)
        corrected_values = sum(1 for r in verified_recs if r.get("correction_reason"))

        pass_count = sum(1 for r in eval_results if r.get("decision") == "PASS")
        reject_count = sum(1 for r in eval_results if r.get("decision") == "REJECT")
        review_count = sum(1 for r in eval_results if r.get("decision") == "REVIEW")

        represented_pages = set()
        valid_provenance_count = 0
        for q in approved_qa:
            pgs = q.get("source_pages", [])
            represented_pages.update(pgs)
            if q.get("source_chunk_ids") and pgs:
                valid_provenance_count += 1

        report = {
            "pdf_pages_processed": len(pages_recs),
            "pages_represented_in_qa": sorted(list(represented_pages)),
            "ocr_warnings_detected": ocr_warnings,
            "values_normalized_or_corrected": corrected_values,
            "total_chunks_created": len(chunks),
            "total_facts_verified": len(facts),
            "qa_counts_by_perspective": {
                "direct": len(direct_qa),
                "indirect": len(indirect_qa),
                "attribute": len(attribute_qa),
                "combined": len(combined_qa),
                "cross_page": len(cross_qa),
                "total_generated": (len(direct_qa) + len(indirect_qa) + len(attribute_qa) + len(combined_qa) + len(cross_qa))
            },
            "exact_normalized_duplicates_removed": len(duplicates),
            "evaluation_breakdown": {
                "PASS": pass_count,
                "REJECT": reject_count,
                "REVIEW": review_count
            },
            "qa_with_valid_source_chunk_and_pages": valid_provenance_count,
            "approved_qa_total": len(approved_qa),
            "fine_tuning_export_splits": {
                "train_samples": len(train_sft),
                "validation_samples": len(val_sft),
                "total_sft": len(train_sft) + len(val_sft)
            }
        }

        # Write to quality audit log
        report_file = self.quality_dir / "quality_audit_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Quality audit report successfully saved to: {report_file}")
        return report
