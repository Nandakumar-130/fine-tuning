import json
import re
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from src.schemas import (
    ChunkRecord, FactRecord, RelatedChunkGroup, RichQARecord, QAResponseSchema, QAPairItem
)
from src.io_utils import ensure_directory, append_jsonl_record, CheckpointTracker, StructuredLogger
from src.vllm_engine import VLLMClient

logger = logging.getLogger("QAGenerator")

class SequentialPerspectiveQAGenerator:
    """
    Step 8: Sequential Perspective-Based Q&A Generation
    Strict sequential order from PDF:
      Stage 1: Direct Q&A (Prompt A)
      Stage 2: Indirect Q&A (Prompt B) from ALL approved Direct Q&A + original evidence
      Stage 3: Attribute Q&A (Prompt C) from ALL approved Indirect Q&A + original evidence
      Stage 4: Combined Q&A (Prompt D) from ALL approved Attribute Q&A + original evidence
      Stage 5: Cross-page Q&A (Prompt E) from verified cross-page multi-chunk groups

    Key Live Features:
      - Live console progress bar (tqdm) showing exactly which chunk is processing.
      - Live real-time counter: prints [LIVE Q&A #Count] for every single question generated.
      - Live disk streaming (flushed immediately to its separate JSONL file).
      - Checkpoint tracker: if stopped with Ctrl+C, resumes seamlessly from the exact chunk.
    """
    def __init__(
        self,
        output_dir: str | Path,
        vllm_client: VLLMClient,
        max_qa_per_perspective: int = 10,
        checkpoint_tracker: Optional[CheckpointTracker] = None,
        structured_logger: Optional[StructuredLogger] = None
    ):
        self.output_dir = Path(output_dir)
        self.qa_dir = self.output_dir / "qa"
        ensure_directory(self.qa_dir)
        self.llm = vllm_client
        self.max_qa = max(10, max_qa_per_perspective)
        self.tracker = checkpoint_tracker
        self.slogger = structured_logger
        self.global_live_counter = 0

    def _extract_qa_items(self, response_text: str, perspective: str, fallback_chunk: ChunkRecord) -> List[RichQARecord]:
        items: List[RichQARecord] = []
        try:
            clean_json = self._extract_json_block(response_text)
            data = json.loads(clean_json)
            raw_items = data.get("items", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])

            for idx, item in enumerate(raw_items):
                q = str(item.get("question", "")).strip()
                a = str(item.get("answer", "")).strip()
                if q and a:
                    chunk_ids = item.get("source_chunk_ids") or [fallback_chunk.chunk_id]
                    pages = item.get("source_pages") or fallback_chunk.page_numbers
                    f_ids = item.get("fact_ids") or []
                    p_id = item.get("parent_qa_id")

                    items.append(
                        RichQARecord(
                            qa_id=f"qa_{perspective[:3]}_{fallback_chunk.chunk_id}_{idx:02d}",
                            question=q,
                            answer=a,
                            perspective=perspective,
                            parent_qa_id=p_id,
                            fact_ids=f_ids,
                            source_chunk_ids=chunk_ids,
                            source_pages=pages,
                            evaluation_status="PENDING"
                        )
                    )
        except Exception as e:
            if self.slogger:
                self.slogger.log_event("WARNING", f"{perspective}_qa", f"JSON parsing failed for {fallback_chunk.chunk_id}: {e}")
            logger.warning(f"Failed parsing {perspective} Q&A items: {e}. Executing Prompt G format repair.")
            try:
                schema_hint = '{"items":[{"question":"...","answer":"...","fact_ids":[],"source_chunk_ids":[],"source_pages":[]}]}'
                repaired = self.llm.repair_json_format(schema_hint, response_text, str(e))
                clean_json = self._extract_json_block(repaired)
                data = json.loads(clean_json)
                raw_items = data.get("items", []) if isinstance(data, dict) else []
                for idx, item in enumerate(raw_items):
                    q = str(item.get("question", "")).strip()
                    a = str(item.get("answer", "")).strip()
                    if q and a:
                        items.append(
                            RichQARecord(
                                qa_id=f"qa_{perspective[:3]}_{fallback_chunk.chunk_id}_{idx:02d}",
                                question=q,
                                answer=a,
                                perspective=perspective,
                                parent_qa_id=item.get("parent_qa_id"),
                                fact_ids=item.get("fact_ids", []),
                                source_chunk_ids=item.get("source_chunk_ids", [fallback_chunk.chunk_id]),
                                source_pages=item.get("source_pages", fallback_chunk.page_numbers),
                                evaluation_status="PENDING"
                            )
                        )
            except Exception as repair_err:
                if self.slogger:
                    self.slogger.log_event("ERROR", f"{perspective}_qa", f"Prompt G repair failed for {fallback_chunk.chunk_id}: {repair_err}")
                logger.error(f"Format repair failed for {perspective} on {fallback_chunk.chunk_id}: {repair_err}")

        return items

    # -----------------------------------------------------------------------
    # STAGE 1: Direct Q&A (Prompt A)
    # -----------------------------------------------------------------------
    def generate_stage1_direct(self, chunks: List[ChunkRecord]) -> List[RichQARecord]:
        stage_name = "direct_qa"
        out_file = self.qa_dir / "direct_qa.jsonl"
        all_stage_records: List[RichQARecord] = []

        system_prompt = (
            "You create evidence-grounded Q&A. Use only supplied evidence. "
            "Do not use outside knowledge or invent facts. Return JSON only."
        )

        pbar = tqdm(chunks, desc="[Live Q&A Stage 1: Direct]", unit="chunk", file=sys.stdout)
        for chunk in pbar:
            if self.tracker and self.tracker.is_chunk_done(stage_name, chunk.chunk_id):
                continue

            user_prompt = (
                f"SOURCE_EVIDENCE:\n{chunk.content}\n\n"
                f"SOURCE_METADATA:\nChunk IDs: [{chunk.chunk_id}]\nPage numbers: {chunk.page_numbers}\n\n"
                f"TASK: Generate at least 10 distinct, evidence-grounded DIRECT question-answer pairs covering facts, "
                "data values, definitions, and actions stated in this chunk. Cover distinct facts where possible. "
                "Answers must be clear and supported. Do not add facts absent from the evidence. Return:\n"
                '{"items":[{"question":"...","answer":"...", "fact_ids":[], "source_chunk_ids":["' + chunk.chunk_id + '"], "source_pages":' + str(chunk.page_numbers) + ', "perspective":"direct"}]}'
            )

            resp = self.llm.generate(system_prompt, user_prompt, temperature=0.2)
            items = self._extract_qa_items(resp, "direct", chunk)

            # Live file streaming and live reporting
            for it in items:
                self.global_live_counter += 1
                append_jsonl_record(out_file, it)
                all_stage_records.append(it)
                tqdm.write(f"  [LIVE Q&A #{self.global_live_counter:04d}] [Direct] {chunk.chunk_id} (Pages {chunk.page_numbers}): \"{it.question[:75]}...\"")

            if self.tracker:
                self.tracker.mark_chunk_done(stage_name, chunk.chunk_id)
            if self.slogger:
                self.slogger.log_event("INFO", stage_name, f"Generated {len(items)} Direct Q&A for {chunk.chunk_id}")

            pbar.set_postfix({"Stage1_Q&A_Generated": len(all_stage_records), "Total_Live_Q&A": self.global_live_counter})

        return all_stage_records

    # -----------------------------------------------------------------------
    # STAGE 2: Indirect Q&A (Prompt B)
    # -----------------------------------------------------------------------
    def generate_stage2_indirect(
        self,
        chunks: List[ChunkRecord],
        approved_direct_qa: List[RichQARecord]
    ) -> List[RichQARecord]:
        stage_name = "indirect_qa"
        out_file = self.qa_dir / "indirect_qa.jsonl"
        all_stage_records: List[RichQARecord] = []

        direct_by_chunk: Dict[str, List[RichQARecord]] = {}
        for d in approved_direct_qa:
            for cid in d.source_chunk_ids:
                direct_by_chunk.setdefault(cid, []).append(d)

        system_prompt = (
            "Generate new question formulations from approved Q&A. "
            "Original evidence is the authority for factual correctness. Return valid JSON only."
        )

        pbar = tqdm(chunks, desc="[Live Q&A Stage 2: Indirect]", unit="chunk", file=sys.stdout)
        for chunk in pbar:
            if self.tracker and self.tracker.is_chunk_done(stage_name, chunk.chunk_id):
                continue

            chunk_direct = direct_by_chunk.get(chunk.chunk_id, approved_direct_qa[:10])
            batch_summary = "\n".join([f"- Direct Q ({q.qa_id}): {q.question} | Ans: {q.answer}" for q in chunk_direct[:10]])

            user_prompt = (
                f"ORIGINAL_SOURCE_EVIDENCE:\n{chunk.content}\n\n"
                f"ALL_APPROVED_DIRECT_QA:\n{batch_summary}\n\n"
                f"TASK: Review every Direct Q&A item. Generate at least 10 distinct INDIRECT or relationship-focused Q&A pairs. "
                'Explore new framing such as "How is X connected to Y?" or "Under what conditions does X occur?" '
                "Do not simply copy Direct questions. Do not invent relationships. Verify answers against original evidence.\n"
                'Return items with question, answer, fact_ids, source_chunk_ids, source_pages and perspective="indirect".'
            )

            resp = self.llm.generate(system_prompt, user_prompt, temperature=0.3)
            items = self._extract_qa_items(resp, "indirect", chunk)

            for it in items:
                self.global_live_counter += 1
                append_jsonl_record(out_file, it)
                all_stage_records.append(it)
                tqdm.write(f"  [LIVE Q&A #{self.global_live_counter:04d}] [Indirect] {chunk.chunk_id} (Pages {chunk.page_numbers}): \"{it.question[:75]}...\"")

            if self.tracker:
                self.tracker.mark_chunk_done(stage_name, chunk.chunk_id)
            if self.slogger:
                self.slogger.log_event("INFO", stage_name, f"Generated {len(items)} Indirect Q&A for {chunk.chunk_id}")

            pbar.set_postfix({"Stage2_Q&A_Generated": len(all_stage_records), "Total_Live_Q&A": self.global_live_counter})

        return all_stage_records

    # -----------------------------------------------------------------------
    # STAGE 3: Attribute Q&A (Prompt C)
    # -----------------------------------------------------------------------
    def generate_stage3_attribute(
        self,
        chunks: List[ChunkRecord],
        approved_indirect_qa: List[RichQARecord]
    ) -> List[RichQARecord]:
        stage_name = "attribute_qa"
        out_file = self.qa_dir / "attribute_qa.jsonl"
        all_stage_records: List[RichQARecord] = []

        indirect_by_chunk: Dict[str, List[RichQARecord]] = {}
        for ind in approved_indirect_qa:
            for cid in ind.source_chunk_ids:
                indirect_by_chunk.setdefault(cid, []).append(ind)

        system_prompt = (
            "Create evidence-grounded attribute questions. "
            "Do not infer attributes that are not stated. Return JSON only."
        )

        pbar = tqdm(chunks, desc="[Live Q&A Stage 3: Attribute]", unit="chunk", file=sys.stdout)
        for chunk in pbar:
            if self.tracker and self.tracker.is_chunk_done(stage_name, chunk.chunk_id):
                continue

            chunk_indirect = indirect_by_chunk.get(chunk.chunk_id, approved_indirect_qa[:10])
            batch_summary = "\n".join([f"- Indirect Q ({q.qa_id}): {q.question} | Ans: {q.answer}" for q in chunk_indirect[:10]])

            user_prompt = (
                f"ORIGINAL SOURCE EVIDENCE:\n{chunk.content}\n\n"
                f"ALL_APPROVED_INDIRECT_QA:\n{batch_summary}\n\n"
                f"TASK: Review all supplied Indirect Q&A. Identify attributes present in the original evidence: "
                "location, equipment, date, capacity, amount, role, category, metrics, units, etc. "
                f"Generate at least 10 ATTRIBUTE-focused Q&A pairs. Do not invent missing attributes. Verify answers against evidence.\n"
                'Return items with question, answer, fact_ids, source_chunk_ids, source_pages and perspective="attribute".'
            )

            resp = self.llm.generate(system_prompt, user_prompt, temperature=0.2)
            items = self._extract_qa_items(resp, "attribute", chunk)

            for it in items:
                self.global_live_counter += 1
                append_jsonl_record(out_file, it)
                all_stage_records.append(it)
                tqdm.write(f"  [LIVE Q&A #{self.global_live_counter:04d}] [Attribute] {chunk.chunk_id} (Pages {chunk.page_numbers}): \"{it.question[:75]}...\"")

            if self.tracker:
                self.tracker.mark_chunk_done(stage_name, chunk.chunk_id)
            if self.slogger:
                self.slogger.log_event("INFO", stage_name, f"Generated {len(items)} Attribute Q&A for {chunk.chunk_id}")

            pbar.set_postfix({"Stage3_Q&A_Generated": len(all_stage_records), "Total_Live_Q&A": self.global_live_counter})

        return all_stage_records

    # -----------------------------------------------------------------------
    # STAGE 4: Combined Q&A (Prompt D)
    # -----------------------------------------------------------------------
    def generate_stage4_combined(
        self,
        chunks: List[ChunkRecord],
        approved_attribute_qa: List[RichQARecord]
    ) -> List[RichQARecord]:
        stage_name = "combined_qa"
        out_file = self.qa_dir / "combined_qa.jsonl"
        all_stage_records: List[RichQARecord] = []

        attr_by_chunk: Dict[str, List[RichQARecord]] = {}
        for att in approved_attribute_qa:
            for cid in att.source_chunk_ids:
                attr_by_chunk.setdefault(cid, []).append(att)

        system_prompt = (
            "Generate multi-fact questions grounded only in supplied evidence. Return JSON only."
        )

        pbar = tqdm(chunks, desc="[Live Q&A Stage 4: Combined]", unit="chunk", file=sys.stdout)
        for chunk in pbar:
            if self.tracker and self.tracker.is_chunk_done(stage_name, chunk.chunk_id):
                continue

            chunk_attr = attr_by_chunk.get(chunk.chunk_id, approved_attribute_qa[:10])
            batch_summary = "\n".join([f"- Attribute Q ({q.qa_id}): {q.question} | Ans: {q.answer}" for q in chunk_attr[:10]])

            user_prompt = (
                f"ORIGINAL_SOURCE_EVIDENCE:\n{chunk.content}\n\n"
                f"ALL_APPROVED_ATTRIBUTE_QA:\n{batch_summary}\n\n"
                f"TASK: Generate at least 10 COMBINED multi-fact questions. Each question must require two or more supported facts "
                "from this chunk (e.g. connecting an action and a date, or a product and its capacity). "
                "Answers must address every part of the question. Include all supporting chunk IDs and page numbers.\n"
                'Return items with question, answer, fact_ids, source_chunk_ids, source_pages and perspective="combined".'
            )

            resp = self.llm.generate(system_prompt, user_prompt, temperature=0.3)
            items = self._extract_qa_items(resp, "combined", chunk)

            for it in items:
                self.global_live_counter += 1
                append_jsonl_record(out_file, it)
                all_stage_records.append(it)
                tqdm.write(f"  [LIVE Q&A #{self.global_live_counter:04d}] [Combined] {chunk.chunk_id} (Pages {chunk.page_numbers}): \"{it.question[:75]}...\"")

            if self.tracker:
                self.tracker.mark_chunk_done(stage_name, chunk.chunk_id)
            if self.slogger:
                self.slogger.log_event("INFO", stage_name, f"Generated {len(items)} Combined Q&A for {chunk.chunk_id}")

            pbar.set_postfix({"Stage4_Q&A_Generated": len(all_stage_records), "Total_Live_Q&A": self.global_live_counter})

        return all_stage_records

    # -----------------------------------------------------------------------
    # STAGE 5: Cross-Page Q&A (Prompt E)
    # -----------------------------------------------------------------------
    def generate_stage5_cross_page(
        self,
        related_groups: List[RelatedChunkGroup]
    ) -> List[RichQARecord]:
        stage_name = "cross_page_qa"
        out_file = self.qa_dir / "cross_page_qa.jsonl"
        all_stage_records: List[RichQARecord] = []

        system_prompt = (
            "Generate multi-page Q&A only from supplied evidence. "
            "Do not assume similar names refer to the same entity unless evidence confirms it. Return JSON only."
        )

        pbar = tqdm(related_groups, desc="[Live Q&A Stage 5: Cross-Page]", unit="group", file=sys.stdout)
        for group in pbar:
            if self.tracker and self.tracker.is_chunk_done(stage_name, group.group_id):
                continue

            user_prompt = (
                f"VERIFIED ENTITY:\n{group.entity_name} ({group.verification_notes})\n\n"
                f"RELATED_SOURCE_CHUNKS:\n{group.combined_evidence}\n\n"
                "TASK: Generate at least 5-10 useful questions requiring information from at least two supplied chunks spanning different pages. "
                "Combine evidence accurately across pages. Cite every supporting chunk ID and page. Exclude unrelated chunks.\n"
                'Return items with question, answer, source_chunk_ids, source_pages and perspective="cross_page".'
            )

            dummy_chunk = ChunkRecord(
                chunk_id=group.group_id,
                page_numbers=group.page_numbers,
                content=group.combined_evidence,
                token_count=0
            )

            resp = self.llm.generate(system_prompt, user_prompt, temperature=0.2)
            items = self._extract_qa_items(resp, "cross_page", dummy_chunk)

            for it in items:
                self.global_live_counter += 1
                it.source_chunk_ids = group.chunk_ids
                it.source_pages = group.page_numbers
                append_jsonl_record(out_file, it)
                all_stage_records.append(it)
                tqdm.write(f"  [LIVE Q&A #{self.global_live_counter:04d}] [Cross-Page] {group.group_id} (Pages {group.page_numbers}): \"{it.question[:75]}...\"")

            if self.tracker:
                self.tracker.mark_chunk_done(stage_name, group.group_id)
            if self.slogger:
                self.slogger.log_event("INFO", stage_name, f"Generated {len(items)} Cross-Page Q&A for {group.group_id}")

            pbar.set_postfix({"Stage5_Q&A_Generated": len(all_stage_records), "Total_Live_Q&A": self.global_live_counter})

        return all_stage_records

    def _extract_json_block(self, text: str) -> str:
        text = text.strip()
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if match:
            return match.group(1).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return text[start:end+1]
        return text
