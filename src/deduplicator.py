from __future__ import annotations
import re
import unicodedata
from typing import List, Tuple, Set, Dict, Any
from pathlib import Path
import logging

from src.schemas import RichQARecord
from src.io_utils import ensure_directory, write_jsonl

logger = logging.getLogger("Deduplicator")

class QADeduplicator:
    """
    Step 9: Exact & Normalized Duplicate Removal
    PDF Specific Guidelines:
    - Retain different phrasings! Semantically similar questions teach the fine-tuned model
      varied user wording.
    - Do NOT use embedding similarity as an automatic deletion rule!
    - Remove only exact duplicates or normalization-level duplicates:
      1. Normalize Unicode (NFKC)
      2. Convert to lowercase
      3. Trim leading/trailing spaces
      4. Collapse repeated whitespace
      5. Apply conservative punctuation normalization
    - Log removed duplicates and the ID of the retained record.
    - Do not remove an example merely because its answer matches another answer.
    - Save duplicates log to quality/exact_duplicates.jsonl.
    """
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.quality_dir = self.output_dir / "quality"
        ensure_directory(self.quality_dir)

    @staticmethod
    def normalize_question(q: str) -> str:
        # 1. Unicode normalization
        text = unicodedata.normalize("NFKC", q)
        # 2. Lowercase
        text = text.lower()
        # 3. Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # 4. Conservative punctuation normalization (strip trailing ?, ., !)
        text = re.sub(r'[?!.,;:\'"`]+$', '', text).strip()
        return text

    def deduplicate(self, qa_records: List[RichQARecord]) -> Tuple[List[RichQARecord], List[Dict[str, Any]]]:
        logger.info(f"Running exact/normalized duplicate check on {len(qa_records)} Q&A pairs...")

        seen_normalized: Dict[str, str] = {}  # normalized_str -> retained_qa_id
        unique_records: List[RichQARecord] = []
        duplicate_logs: List[Dict[str, Any]] = []

        for record in qa_records:
            norm_q = self.normalize_question(record.question)
            record.normalized_question = norm_q

            if norm_q in seen_normalized:
                retained_id = seen_normalized[norm_q]
                duplicate_logs.append({
                    "removed_qa_id": record.qa_id,
                    "retained_qa_id": retained_id,
                    "perspective": record.perspective,
                    "removed_question": record.question,
                    "normalized_form": norm_q,
                    "reason": "Exact or punctuation/case/whitespace duplicate"
                })
            else:
                seen_normalized[norm_q] = record.qa_id
                unique_records.append(record)

        # Log deduplication records
        dup_file = self.quality_dir / "exact_duplicates.jsonl"
        write_jsonl(dup_file, duplicate_logs)

        logger.info(
            f"Deduplication complete. Retained: {len(unique_records)} pairs, "
            f"Removed duplicates: {len(duplicate_logs)}. Logged to: {dup_file}"
        )
        return unique_records, duplicate_logs
