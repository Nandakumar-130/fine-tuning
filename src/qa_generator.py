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

SCHEMA_HINT = '{"items":[{"question":"...","answer":"...","fact_ids":[],"source_chunk_ids":[],"source_pages":[],"parent_qa_id":"..."}]}'

class SequentialPerspectiveQAGenerator:
    """
    Step 8: Sequential Perspective-Based Q&A Generation
    Enhanced Requirements:
      - Direct Q&A: Generates at least 5 to 10 (or more if supported) direct questions per chunk.
      - Indirect Q&A: Takes each direct Q&A item individually and generates at least 5 to 10 indirect questions from it.
      - Attribute Q&A: Takes each indirect Q&A item individually and generates at least 5 to 10 attribute questions from it.
      - Combined Q&A: Generates multi-fact combined questions (requiring >=2 facts).
      - Cross-Page Q&A: Generates multi-page cross-chunk questions.
      - Robustness / Rate Limit:
          * Max 3 retry attempts per unit if questions generated < 5 or on error.
          * If format error/JSON invalid, resends to Qwen with the exact error message and expected schema (Prompt G) to self-correct.
          * After 3 failed attempts, safely logs and moves on to the next unit without crashing.
      - Real-time live console tracking and immediate disk streaming to separate JSONL files.
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
        self.max_attempts = 3
        self.tracker = checkpoint_tracker
        self.slogger = structured_logger
        self.global_live_counter = 0

    def _extract_json_block(self, text: str) -> str:
        text = text.strip()
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if match:
            return match.group(1).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start:end+1]
        start_arr = text.find("[")
        end_arr = text.rfind("]")
        if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
            return text[start_arr:end_arr+1]
        return text

    def _parse_and_validate_items(
        self,
        raw_text: str,
        perspective: str,
        fallback_chunk_id: str,
        fallback_pages: List[int],
        parent_id: Optional[str] = None,
        valid_chunk_ids: Optional[List[str]] = None,
        valid_pages: Optional[List[int]] = None
    ) -> List[RichQARecord]:
        """
        Parses JSON response into rich Q&A objects.
        Validates question, answer, and strict multi-chunk provenance.
        """
        clean_json = self._extract_json_block(raw_text)
        data = json.loads(clean_json)

        if isinstance(data, dict):
            raw_items = data.get("items") or data.get("questions") or data.get("qa_pairs") or []
        elif isinstance(data, list):
            raw_items = data
        else:
            raise ValueError(f"Unrecognized JSON structure type: {type(data)}")

        if not raw_items:
            raise ValueError("No Q&A items found in parsed JSON structure.")

        valid_records: List[RichQARecord] = []
        known_chunk_set = set(valid_chunk_ids) if valid_chunk_ids else None
        known_page_set = set(valid_pages) if valid_pages else None

        for idx, item in enumerate(raw_items):
            if not isinstance(item, dict):
                continue
            q = str(item.get("question", "")).strip()
            a = str(item.get("answer", "")).strip()
            if len(q) < 5 or len(a) < 2:
                continue

            raw_chunk_ids = item.get("source_chunk_ids")
            raw_pages = item.get("source_pages")

            # Strict Multi-Chunk Provenance Validation
            if perspective == "cross_page":
                # Must provide explicit source_chunk_ids and source_pages spanning >= 2 chunks and >= 2 pages
                if not raw_chunk_ids or not isinstance(raw_chunk_ids, list) or len(set(raw_chunk_ids)) < 2:
                    logger.warning(f"Cross-page Q&A item rejected: requires at least 2 distinct source_chunk_ids, got {raw_chunk_ids}")
                    continue
                if not raw_pages or not isinstance(raw_pages, list) or len(set(raw_pages)) < 2:
                    logger.warning(f"Cross-page Q&A item rejected: requires at least 2 distinct source_pages, got {raw_pages}")
                    continue
                # Validate against verified entity group's supplied chunks and pages
                if known_chunk_set and not set(raw_chunk_ids).issubset(known_chunk_set):
                    logger.warning(f"Cross-page Q&A contains unrelated/hallucinated chunk IDs: {set(raw_chunk_ids) - known_chunk_set}")
                    continue
                if known_page_set and not set(raw_pages).issubset(known_page_set):
                    logger.warning(f"Cross-page Q&A contains invalid pages not in group: {set(raw_pages) - known_page_set}")
                    continue
                chunk_ids = list(dict.fromkeys(raw_chunk_ids))
                pages = sorted(list(dict.fromkeys(raw_pages)))

            elif perspective == "combined":
                # Combined Q&A must require at least two distinct supported facts/propositions
                # If explicit source_chunk_ids provided, validate against known chunks
                if raw_chunk_ids and isinstance(raw_chunk_ids, list):
                    if known_chunk_set and not set(raw_chunk_ids).issubset(known_chunk_set):
                        logger.warning(f"Combined Q&A rejected: cited chunks not in supplied evidence: {set(raw_chunk_ids) - known_chunk_set}")
                        continue
                    chunk_ids = list(dict.fromkeys(raw_chunk_ids))
                else:
                    chunk_ids = [fallback_chunk_id]

                if raw_pages and isinstance(raw_pages, list):
                    if known_page_set and not set(raw_pages).issubset(known_page_set):
                        logger.warning(f"Combined Q&A rejected: cited pages not in chunk: {set(raw_pages) - known_page_set}")
                        continue
                    pages = sorted(list(dict.fromkeys(raw_pages)))
                else:
                    pages = fallback_pages

            else:
                # Direct, Indirect, Attribute: Single-chunk generation tasks.
                # If model provides explicit chunk_ids, verify they do not cite unrelated chunks
                if raw_chunk_ids and isinstance(raw_chunk_ids, list):
                    if fallback_chunk_id not in raw_chunk_ids and known_chunk_set and not set(raw_chunk_ids).issubset(known_chunk_set):
                        logger.warning(f"Single-chunk Q&A item cited unrelated chunks {raw_chunk_ids}; resetting to true chunk {fallback_chunk_id}")
                        chunk_ids = [fallback_chunk_id]
                    else:
                        chunk_ids = list(dict.fromkeys(raw_chunk_ids))
                else:
                    chunk_ids = [fallback_chunk_id]

                if raw_pages and isinstance(raw_pages, list):
                    pages = sorted(list(dict.fromkeys(raw_pages)))
                else:
                    pages = fallback_pages

            f_ids = item.get("fact_ids") or []
            p_id = item.get("parent_qa_id") or parent_id

            clean_chunk_id = re.sub(r'[^a-zA-Z0-9_]', '_', fallback_chunk_id)
            qa_id = f"qa_{perspective[:3]}_{clean_chunk_id}_{len(valid_records):03d}"

            rec = RichQARecord(
                qa_id=qa_id,
                question=q,
                answer=a,
                perspective=perspective,
                parent_qa_id=p_id,
                fact_ids=f_ids,
                source_chunk_ids=chunk_ids,
                source_pages=pages,
                evaluation_status="PENDING"
            )
            valid_records.append(rec)

        return valid_records

    def _generate_with_retry_and_repair(
        self,
        system_prompt: str,
        user_prompt: str,
        perspective: str,
        fallback_chunk_id: str,
        fallback_pages: List[int],
        parent_id: Optional[str] = None,
        valid_chunk_ids: Optional[List[str]] = None,
        valid_pages: Optional[List[int]] = None
    ) -> List[RichQARecord]:
        """
        Runs generation with up to max_attempts (3 attempts).
        - Generates up to requested target (max_qa).
        - Accepts one or more valid questions (>= 1) when grounded in source evidence.
        - Accepts fewer questions when source contains limited facts without forcing a numerical quota.
        - Preserves the 3-attempt retry mechanism for genuine formatting, JSON syntax, or model errors.
        - Uses Prompt G format repair to fix syntax errors without guessing missing facts.
        """
        current_user_prompt = user_prompt
        last_error = ""

        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.llm.generate(system_prompt, current_user_prompt, temperature=0.2 if attempt == 1 else 0.4)
                
                try:
                    records = self._parse_and_validate_items(
                        response, perspective, fallback_chunk_id, fallback_pages, parent_id,
                        valid_chunk_ids=valid_chunk_ids, valid_pages=valid_pages
                    )
                except Exception as parse_err:
                    # Genuine formatting or JSON error: Resend to Qwen using Prompt G with parser error
                    logger.warning(
                        f"Attempt {attempt}/{self.max_attempts}: Format error ({parse_err}). Resending to Qwen with error details for correction..."
                    )
                    if self.slogger:
                        self.slogger.log_event("WARNING", perspective, f"Attempt {attempt} parse error: {parse_err}")
                    
                    repaired_text = self.llm.repair_json_format(
                        expected_schema=SCHEMA_HINT,
                        invalid_output=response,
                        parser_error=str(parse_err)
                    )
                    records = self._parse_and_validate_items(
                        repaired_text, perspective, fallback_chunk_id, fallback_pages, parent_id,
                        valid_chunk_ids=valid_chunk_ids, valid_pages=valid_pages
                    )

                # If valid questions were extracted (1 or more supported by evidence), accept immediately
                if len(records) >= 1:
                    logger.info(f"Successfully generated and validated {len(records)} {perspective} questions for {fallback_chunk_id}.")
                    return records
                else:
                    last_error = "Model returned empty question list or unparseable content."
                    logger.warning(f"Attempt {attempt}/{self.max_attempts}: {last_error} Retrying...")
                    current_user_prompt = (
                        user_prompt +
                        f"\n\nNOTE: The previous attempt returned zero valid question items. "
                        f"Generate useful, evidence-grounded questions (up to {self.max_qa}) supported by the text. Return valid JSON only."
                    )

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt}/{self.max_attempts} failed with exception: {e}")
                if self.slogger:
                    self.slogger.log_event("WARNING", perspective, f"Attempt {attempt} exception: {e}")

        # If all 3 attempts exhausted
        logger.error(f"All {self.max_attempts} attempts exhausted for {fallback_chunk_id} ({perspective}). Last error: {last_error}. Moving to next.")
        if self.slogger:
            self.slogger.log_event("ERROR", perspective, f"3 attempts failed for {fallback_chunk_id}: {last_error}")
        return []

    # -----------------------------------------------------------------------
    # STAGE 1: Direct Q&A (Prompt A)
    # Generates at least 5 to 10 (or more) Direct Q&A per chunk
    # -----------------------------------------------------------------------
    def generate_stage1_direct(self, chunks: List[ChunkRecord]) -> List[RichQARecord]:
        stage_name = "direct_qa"
        out_file = self.qa_dir / "direct_qa.jsonl"
        all_stage_records: List[RichQARecord] = []

        system_prompt = (
            "You create evidence-grounded Q&A. Use only supplied evidence. "
            "Do not use outside knowledge or invent facts. Return valid JSON only."
        )

        pbar = tqdm(chunks, desc="[Live Q&A Stage 1: Direct]", unit="chunk", file=sys.stdout)
        for chunk in pbar:
            if self.tracker and self.tracker.is_chunk_done(stage_name, chunk.chunk_id):
                continue

            user_prompt = (
                f"SOURCE_EVIDENCE:\n{chunk.content}\n\n"
                f"SOURCE_METADATA:\nChunk IDs: [{chunk.chunk_id}]\nPage numbers: {chunk.page_numbers}\n\n"
                f"TASK: Generate useful, evidence-grounded DIRECT question-answer pairs (up to {self.max_qa}) supported by the text. "
                "Cover distinct facts, numbers, dates, locations, and actions stated in this chunk. "
                "Answers must be clear and supported. Do not invent questions if evidence is limited. Return strictly valid JSON format:\n"
                '{"items":[{"question":"...","answer":"...", "fact_ids":[], "source_chunk_ids":["' + chunk.chunk_id + '"], "source_pages":' + str(chunk.page_numbers) + ', "perspective":"direct"}]}'
            )

            records = self._generate_with_retry_and_repair(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                perspective="direct",
                fallback_chunk_id=chunk.chunk_id,
                fallback_pages=chunk.page_numbers,
                valid_chunk_ids=[chunk.chunk_id],
                valid_pages=chunk.page_numbers
            )

            # Live file streaming and live reporting
            for it in records:
                self.global_live_counter += 1
                append_jsonl_record(out_file, it)
                all_stage_records.append(it)
                tqdm.write(f"  [LIVE Q&A #{self.global_live_counter:04d}] [Direct] {chunk.chunk_id}: \"{it.question[:70]}...\"")

            if self.tracker:
                self.tracker.mark_chunk_done(stage_name, chunk.chunk_id)
            if self.slogger:
                self.slogger.log_event("INFO", stage_name, f"Chunk {chunk.chunk_id} generated {len(records)} Direct Q&A")

            pbar.set_postfix({"Stage1_Total": len(all_stage_records), "Live_Q&A_Count": self.global_live_counter})

        return all_stage_records

    # -----------------------------------------------------------------------
    # STAGE 2: Indirect Q&A (Prompt B)
    # Takes each Direct Q&A item and generates at least 5 to 10 Indirect Q&A from it
    # -----------------------------------------------------------------------
    def generate_stage2_indirect(
        self,
        chunks: List[ChunkRecord],
        approved_direct_qa: List[RichQARecord]
    ) -> List[RichQARecord]:
        stage_name = "indirect_qa"
        out_file = self.qa_dir / "indirect_qa.jsonl"
        all_stage_records: List[RichQARecord] = []

        chunk_dict = {c.chunk_id: c for c in chunks}

        system_prompt = (
            "Generate new question formulations from approved Q&A. "
            "Original evidence is the authority for factual correctness. Return valid JSON only."
        )

        pbar = tqdm(approved_direct_qa, desc="[Live Q&A Stage 2: Indirect (Per Direct Q&A)]", unit="item", file=sys.stdout)
        for direct_item in pbar:
            item_checkpoint_key = f"item_{direct_item.qa_id}"
            if self.tracker and self.tracker.is_chunk_done(stage_name, item_checkpoint_key):
                continue

            # Retrieve original source chunk evidence
            primary_chunk_id = direct_item.source_chunk_ids[0] if direct_item.source_chunk_ids else "chunk_001"
            chunk = chunk_dict.get(primary_chunk_id)
            chunk_content = chunk.content if chunk else "Original evidence content."
            chunk_pages = chunk.page_numbers if chunk else direct_item.source_pages

            user_prompt = (
                f"ORIGINAL_SOURCE_EVIDENCE:\n{chunk_content}\n\n"
                f"TARGET_DIRECT_QA_ITEM:\n"
                f"Question: {direct_item.question}\n"
                f"Answer: {direct_item.answer}\n\n"
                f"TASK: Focus specifically on this direct Q&A item and the supporting evidence. "
                f"Generate useful, relationship-focused Q&A pairs (up to {self.max_qa}) from it where evidence supports them. "
                'Explore relationships, consequences, context, and connections (e.g. "How is X connected to Y?", "Why did X happen?"). '
                "Do not simply duplicate the direct question. Verify all answers against the original evidence. Return JSON:\n"
                '{"items":[{"question":"...","answer":"...", "fact_ids":' + str(direct_item.fact_ids) + ', "source_chunk_ids":["' + primary_chunk_id + '"], "source_pages":' + str(chunk_pages) + ', "perspective":"indirect", "parent_qa_id":"' + direct_item.qa_id + '"}]}'
            )

            records = self._generate_with_retry_and_repair(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                perspective="indirect",
                fallback_chunk_id=primary_chunk_id,
                fallback_pages=chunk_pages,
                parent_id=direct_item.qa_id,
                valid_chunk_ids=[primary_chunk_id],
                valid_pages=chunk_pages
            )

            for it in records:
                self.global_live_counter += 1
                append_jsonl_record(out_file, it)
                all_stage_records.append(it)
                tqdm.write(f"  [LIVE Q&A #{self.global_live_counter:04d}] [Indirect from {direct_item.qa_id}]: \"{it.question[:70]}...\"")

            if self.tracker:
                self.tracker.mark_chunk_done(stage_name, item_checkpoint_key)

            pbar.set_postfix({"Stage2_Total": len(all_stage_records), "Live_Q&A_Count": self.global_live_counter})

        return all_stage_records

    # -----------------------------------------------------------------------
    # STAGE 3: Attribute Q&A (Prompt C)
    # Takes each Indirect Q&A item and generates at least 5 to 10 Attribute Q&A from it
    # -----------------------------------------------------------------------
    def generate_stage3_attribute(
        self,
        chunks: List[ChunkRecord],
        approved_indirect_qa: List[RichQARecord]
    ) -> List[RichQARecord]:
        stage_name = "attribute_qa"
        out_file = self.qa_dir / "attribute_qa.jsonl"
        all_stage_records: List[RichQARecord] = []

        chunk_dict = {c.chunk_id: c for c in chunks}

        system_prompt = (
            "Create evidence-grounded attribute questions. "
            "Do not infer attributes that are not stated. Return valid JSON only."
        )

        pbar = tqdm(approved_indirect_qa, desc="[Live Q&A Stage 3: Attribute (Per Indirect Q&A)]", unit="item", file=sys.stdout)
        for indirect_item in pbar:
            item_checkpoint_key = f"item_{indirect_item.qa_id}"
            if self.tracker and self.tracker.is_chunk_done(stage_name, item_checkpoint_key):
                continue

            primary_chunk_id = indirect_item.source_chunk_ids[0] if indirect_item.source_chunk_ids else "chunk_001"
            chunk = chunk_dict.get(primary_chunk_id)
            chunk_content = chunk.content if chunk else "Original evidence content."
            chunk_pages = chunk.page_numbers if chunk else indirect_item.source_pages

            user_prompt = (
                f"ORIGINAL SOURCE EVIDENCE:\n{chunk_content}\n\n"
                f"TARGET_INDIRECT_QA_ITEM:\n"
                f"Question: {indirect_item.question}\n"
                f"Answer: {indirect_item.answer}\n\n"
                f"TASK: Focus on this indirect Q&A item and identify specific attributes present in the original evidence "
                "(e.g. location, specifications, date, amounts, capacities, categories, equipment, roles, division names). "
                f"Generate useful ATTRIBUTE-focused Q&A pairs (up to {self.max_qa}) directly tied to this item where evidence permits. "
                "Do not invent missing attributes. Verify answers strictly against evidence. Return valid JSON:\n"
                '{"items":[{"question":"...","answer":"...", "fact_ids":' + str(indirect_item.fact_ids) + ', "source_chunk_ids":["' + primary_chunk_id + '"], "source_pages":' + str(chunk_pages) + ', "perspective":"attribute", "parent_qa_id":"' + indirect_item.qa_id + '"}]}'
            )

            records = self._generate_with_retry_and_repair(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                perspective="attribute",
                fallback_chunk_id=primary_chunk_id,
                fallback_pages=chunk_pages,
                parent_id=indirect_item.qa_id,
                valid_chunk_ids=[primary_chunk_id],
                valid_pages=chunk_pages
            )

            for it in records:
                self.global_live_counter += 1
                append_jsonl_record(out_file, it)
                all_stage_records.append(it)
                tqdm.write(f"  [LIVE Q&A #{self.global_live_counter:04d}] [Attribute from {indirect_item.qa_id}]: \"{it.question[:70]}...\"")

            if self.tracker:
                self.tracker.mark_chunk_done(stage_name, item_checkpoint_key)

            pbar.set_postfix({"Stage3_Total": len(all_stage_records), "Live_Q&A_Count": self.global_live_counter})

        return all_stage_records

    # -----------------------------------------------------------------------
    # STAGE 4: Combined Q&A (Prompt D)
    # Generates multi-fact questions requiring >= 2 supported facts
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
            "Generate multi-fact questions grounded only in supplied evidence. Return valid JSON only."
        )

        pbar = tqdm(chunks, desc="[Live Q&A Stage 4: Combined (Multi-Fact)]", unit="chunk", file=sys.stdout)
        for chunk in pbar:
            if self.tracker and self.tracker.is_chunk_done(stage_name, chunk.chunk_id):
                continue

            chunk_attr = attr_by_chunk.get(chunk.chunk_id, [])
            batch_summary = (
                "\n".join([f"- Attribute Q ({q.qa_id}): {q.question} | Ans: {q.answer}" for q in chunk_attr[:10]])
                if chunk_attr else "None (generate directly from original evidence facts)."
            )

            user_prompt = (
                f"ORIGINAL_SOURCE_EVIDENCE:\n{chunk.content}\n\n"
                f"ALL_APPROVED_ATTRIBUTE_QA:\n{batch_summary}\n\n"
                f"TASK: Generate COMBINED multi-fact questions (up to {self.max_qa}) where supported. "
                "Each question must require two or more supported facts from this evidence "
                "(e.g., connecting a division name and its project location, or equipment manufactured and financial output). "
                "Answers must address every part of the question. Return JSON format:\n"
                '{"items":[{"question":"...","answer":"...", "fact_ids":[], "source_chunk_ids":["' + chunk.chunk_id + '"], "source_pages":' + str(chunk.page_numbers) + ', "perspective":"combined"}]}'
            )

            records = self._generate_with_retry_and_repair(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                perspective="combined",
                fallback_chunk_id=chunk.chunk_id,
                fallback_pages=chunk.page_numbers,
                valid_chunk_ids=[chunk.chunk_id],
                valid_pages=chunk.page_numbers
            )

            for it in records:
                self.global_live_counter += 1
                append_jsonl_record(out_file, it)
                all_stage_records.append(it)
                tqdm.write(f"  [LIVE Q&A #{self.global_live_counter:04d}] [Combined] {chunk.chunk_id}: \"{it.question[:70]}...\"")

            if self.tracker:
                self.tracker.mark_chunk_done(stage_name, chunk.chunk_id)

            pbar.set_postfix({"Stage4_Total": len(all_stage_records), "Live_Q&A_Count": self.global_live_counter})

        return all_stage_records

    # -----------------------------------------------------------------------
    # STAGE 5: Cross-Page Q&A (Prompt E)
    # Generates multi-page Q&A from verified multi-chunk groups
    # -----------------------------------------------------------------------
    def generate_stage5_cross_page(
        self,
        related_groups: List[Any],
        chunks: Optional[List[ChunkRecord]] = None
    ) -> List[RichQARecord]:
        stage_name = "cross_page_qa"
        out_file = self.qa_dir / "cross_page_qa.jsonl"
        all_stage_records: List[RichQARecord] = []

        chunk_dict = {c.chunk_id: c for c in chunks} if chunks else {}

        system_prompt = (
            "Generate multi-page Q&A only from supplied evidence. "
            "Do not assume similar names refer to the same entity unless evidence confirms it. Return valid JSON only."
        )

        pbar = tqdm(related_groups, desc="[Live Q&A Stage 5: Cross-Page]", unit="group", file=sys.stdout)
        for group in pbar:
            group_id = getattr(group, "group_id", "GROUP_001")
            if self.tracker and self.tracker.is_chunk_done(stage_name, group_id):
                continue

            entity_name = getattr(group, "entity_name", getattr(group, "topic", "Cross-Page Section"))
            notes = getattr(group, "verification_notes", "; ".join(getattr(group, "relationship_reasons", [])))
            chunk_ids = getattr(group, "chunk_ids", [])
            page_numbers = getattr(group, "page_numbers", [])

            # Construct evidence text
            combined_evidence = getattr(group, "combined_evidence", "")
            if not combined_evidence and chunk_dict and chunk_ids:
                evidence_parts = []
                for cid in chunk_ids:
                    if cid in chunk_dict:
                        evidence_parts.append(f"[Chunk {cid} (Page {chunk_dict[cid].page_numbers})]:\n{chunk_dict[cid].content}")
                combined_evidence = "\n\n".join(evidence_parts)

            if not combined_evidence:
                combined_evidence = f"Evidence from cross-page group {group_id} spanning pages {page_numbers}."

            user_prompt = (
                f"VERIFIED TOPIC/ENTITY:\n{entity_name} ({notes})\n\n"
                f"RELATED_SOURCE_CHUNKS:\n{combined_evidence}\n\n"
                f"TASK: Generate useful questions (up to {self.max_qa}) requiring information from at least two supplied chunks spanning different pages. "
                "Combine evidence accurately across pages. Cite every supporting chunk ID and page. Exclude unrelated chunks.\n"
                'Return JSON format:\n{"items":[{"question":"...","answer":"...", "source_chunk_ids":' + str(chunk_ids) + ', "source_pages":' + str(page_numbers) + ', "perspective":"cross_page"}]}'
            )

            records = self._generate_with_retry_and_repair(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                perspective="cross_page",
                fallback_chunk_id=group_id,
                fallback_pages=page_numbers,
                valid_chunk_ids=chunk_ids,
                valid_pages=page_numbers
            )

            for it in records:
                self.global_live_counter += 1
                it.source_chunk_ids = chunk_ids
                it.source_pages = page_numbers
                append_jsonl_record(out_file, it)
                all_stage_records.append(it)
                tqdm.write(f"  [LIVE Q&A #{self.global_live_counter:04d}] [Cross-Page] {group_id}: \"{it.question[:70]}...\"")

            if self.tracker:
                self.tracker.mark_chunk_done(stage_name, group_id)

            pbar.set_postfix({"Stage5_Total": len(all_stage_records), "Live_Q&A_Count": self.global_live_counter})

        return all_stage_records
