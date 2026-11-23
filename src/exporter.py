import random
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

from src.schemas import RichQARecord, SFTConversationRecord, ChatMessage
from src.io_utils import ensure_directory, write_jsonl

logger = logging.getLogger("FineTuningExporter")

class FineTuningExporter:
    """
    Step 11: Final Fine-Tuning SFT Export & Train/Validation Splitting
    PDF Requirements:
    - Keep rich audit records (qa_id, question, answer, perspective, chunk IDs, source pages).
    - Convert approved examples to Qwen chat SFT format:
      {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    - Critical Split Rule:
      Split by source fact / entity group to avoid data leakage!
      Never put one paraphrase in train and another paraphrase of the same fact in validation.
    - Saves:
      final/train.jsonl
      final/validation.jsonl
    """
    def __init__(self, output_dir: str | Path, train_ratio: float = 0.85):
        self.output_dir = Path(output_dir)
        self.final_dir = self.output_dir / "final"
        self.processed_dir = self.output_dir / "processed_data"
        ensure_directory(self.final_dir)
        ensure_directory(self.processed_dir)
        self.train_ratio = train_ratio

    def export_and_split(
        self,
        approved_records: List[RichQARecord]
    ) -> Tuple[List[SFTConversationRecord], List[SFTConversationRecord]]:
        logger.info(f"Exporting {len(approved_records)} approved Q&A records to SFT format...")

        # Group records by underlying fact or primary source chunk to prevent leakage
        groups: Dict[str, List[RichQARecord]] = {}

        for rec in approved_records:
            # Group key: first fact_id if available, else first source_chunk_id
            if rec.fact_ids:
                key = rec.fact_ids[0]
            elif rec.source_chunk_ids:
                key = rec.source_chunk_ids[0]
            else:
                key = "unknown_cluster"

            groups.setdefault(key, []).append(rec)

        group_keys = list(groups.keys())
        # Deterministic shuffle for reproducibility
        random.seed(42)
        random.shuffle(group_keys)

        split_idx = int(len(group_keys) * self.train_ratio)
        train_keys = set(group_keys[:split_idx])
        val_keys = set(group_keys[split_idx:])

        train_sft: List[SFTConversationRecord] = []
        val_sft: List[SFTConversationRecord] = []

        for key, recs in groups.items():
            target_list = train_sft if key in train_keys else val_sft
            for r in recs:
                sft_item = SFTConversationRecord(
                    id=r.qa_id,
                    messages=[
                        ChatMessage(role="user", content=r.question),
                        ChatMessage(role="assistant", content=r.answer)
                    ],
                    metadata={
                        "perspective": r.perspective,
                        "source_chunk_ids": r.source_chunk_ids,
                        "source_pages": r.source_pages,
                        "fact_ids": r.fact_ids
                    }
                )
                target_list.append(sft_item)

        # Write final train & validation files
        train_file = self.final_dir / "train.jsonl"
        val_file = self.final_dir / "validation.jsonl"
        proc_train_file = self.processed_dir / "sft_train.jsonl"
        proc_val_file = self.processed_dir / "sft_val.jsonl"

        write_jsonl(train_file, train_sft)
        write_jsonl(val_file, val_sft)
        write_jsonl(proc_train_file, train_sft)
        write_jsonl(proc_val_file, val_sft)

        logger.info(
            f"SFT Dataset Export complete:\n"
            f"  - Train records: {len(train_sft)} -> {train_file}\n"
            f"  - Validation records: {len(val_sft)} -> {val_file}\n"
            f"  - Group clusters preserved without leakage: {len(group_keys)}"
        )
        return train_sft, val_sft
