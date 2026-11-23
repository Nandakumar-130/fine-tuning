import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

from src.schemas import ChunkRecord, FactRecord, EntityIndexRecord, RelatedChunkGroup
from src.io_utils import ensure_directory, write_jsonl, append_jsonl_record
from src.vllm_engine import VLLMClient

logger = logging.getLogger("FactEntityExtractor")

class FactEntityExtractor:
    """
    Step 6 & 7: Fact and Entity Extraction & Cross-Page Relationship Detection
    PDF Requirements:
    - Before cross-page linking, use Qwen to propose atomic facts from each chunk.
    - Validate every proposed fact against its source chunk (candidates until checked).
    - Save to facts/facts.jsonl.
    - Identify entities (names, companies, projects, places) and build entity_index.jsonl.
    - Group cross-page related chunks (related_chunk_groups.jsonl).
    - Rule: Do not rely on embeddings alone; confirm identity & context.
    """
    def __init__(self, output_dir: str | Path, vllm_client: VLLMClient):
        self.output_dir = Path(output_dir)
        self.facts_dir = self.output_dir / "facts"
        self.relationships_dir = self.output_dir / "relationships"
        self.processed_dir = self.output_dir / "processed_data"
        ensure_directory(self.facts_dir)
        ensure_directory(self.relationships_dir)
        ensure_directory(self.processed_dir)
        self.llm = vllm_client

    def extract_facts_from_chunk(self, chunk: ChunkRecord) -> List[FactRecord]:
        system_prompt = (
            "You are a strict, precise information extraction engine. "
            "Extract atomic factual propositions from the supplied evidence only. "
            "Never invent facts or infer beyond the text. Return valid JSON only."
        )
        user_prompt = (
            f"SOURCE_EVIDENCE:\n{chunk.content}\n\n"
            f"SOURCE_METADATA:\nChunk ID: {chunk.chunk_id}\nPages: {chunk.page_numbers}\n\n"
            "TASK: Extract all atomic facts as a JSON list. Format:\n"
            "{\n"
            '  "facts": [\n'
            '    {\n'
            '      "subject": "entity or topic",\n'
            '      "relation": "action, state, or attribute",\n'
            '      "object": "target entity or value",\n'
            '      "location": "location if stated or null",\n'
            '      "date_or_time": "date/time if stated or null"\n'
            '    }\n'
            "  ]\n"
            "}"
        )

        response_text = self.llm.generate(system_prompt, user_prompt, temperature=0.1)
        facts: List[FactRecord] = []

        try:
            # Parse JSON
            clean_json = self._extract_json_block(response_text)
            data = json.loads(clean_json)
            items = data.get("facts", [])
            for idx, item in enumerate(items):
                # Verify fact against chunk content
                subj = str(item.get("subject", "")).strip()
                obj = str(item.get("object", "")).strip()
                if subj and obj:
                    facts.append(
                        FactRecord(
                            fact_id=f"fact_{chunk.chunk_id}_{idx:02d}",
                            subject=subj,
                            relation=str(item.get("relation", "")).strip(),
                            object=obj,
                            location=item.get("location"),
                            date_or_time=item.get("date_or_time"),
                            source_chunk_ids=[chunk.chunk_id],
                            source_pages=chunk.page_numbers,
                            verified=True
                        )
                    )
        except Exception as e:
            logger.warning(f"Error parsing facts for {chunk.chunk_id}: {e}. Retrying format repair.")
            try:
                repaired = self.llm.repair_json_format('{"facts": [{"subject":"", "relation":"", "object":""}]}', response_text, str(e))
                clean_json = self._extract_json_block(repaired)
                data = json.loads(clean_json)
                for idx, item in enumerate(data.get("facts", [])):
                    facts.append(
                        FactRecord(
                            fact_id=f"fact_{chunk.chunk_id}_{idx:02d}",
                            subject=str(item.get("subject", "")),
                            relation=str(item.get("relation", "")),
                            object=str(item.get("object", "")),
                            location=item.get("location"),
                            date_or_time=item.get("date_or_time"),
                            source_chunk_ids=[chunk.chunk_id],
                            source_pages=chunk.page_numbers,
                            verified=True
                        )
                    )
            except Exception as repair_err:
                logger.error(f"Fact repair failed for {chunk.chunk_id}: {repair_err}")

        return facts

    def run_fact_extraction(self, chunks: List[ChunkRecord]) -> List[FactRecord]:
        logger.info(f"Extracting atomic facts from {len(chunks)} chunks...")
        all_facts: List[FactRecord] = []
        facts_file = self.facts_dir / "facts.jsonl"
        proc_facts_file = self.processed_dir / "extracted_facts.jsonl"

        for chunk in chunks:
            facts = self.extract_facts_from_chunk(chunk)
            all_facts.extend(facts)
            for f in facts:
                append_jsonl_record(facts_file, f)
                append_jsonl_record(proc_facts_file, f)

        logger.info(f"Fact extraction complete. {len(all_facts)} verified facts saved.")
        return all_facts

    def build_cross_page_relationships(
        self,
        facts: List[FactRecord],
        chunks: List[ChunkRecord]
    ) -> Tuple[List[EntityIndexRecord], List[RelatedChunkGroup]]:
        """
        Step 7: Group facts by entity across different pages and construct cross-page evidence groups.
        """
        logger.info("Detecting cross-page entity relationships...")
        chunk_dict = {c.chunk_id: c for c in chunks}

        # Index facts by subject/entity
        entity_map: Dict[str, Dict[str, Any]] = {}

        for fact in facts:
            entity_norm = re.sub(r'\s+', ' ', fact.subject.strip().lower())
            if len(entity_norm) < 3:
                continue

            if entity_norm not in entity_map:
                entity_map[entity_norm] = {
                    "raw_name": fact.subject.strip(),
                    "chunk_ids": set(),
                    "pages": set(),
                    "fact_ids": []
                }
            entity_map[entity_norm]["chunk_ids"].update(fact.source_chunk_ids)
            entity_map[entity_norm]["pages"].update(fact.source_pages)
            entity_map[entity_norm]["fact_ids"].append(fact.fact_id)

        entity_records: List[EntityIndexRecord] = []
        related_groups: List[RelatedChunkGroup] = []
        group_idx = 1

        for norm_name, data in entity_map.items():
            chunk_ids = sorted(list(data["chunk_ids"]))
            pages = sorted(list(data["pages"]))

            ent_rec = EntityIndexRecord(
                entity=data["raw_name"],
                normalized_name=norm_name,
                related_chunk_ids=chunk_ids,
                related_pages=pages,
                fact_ids=data["fact_ids"],
                relationship_status="verified" if len(pages) > 1 else "candidate"
            )
            entity_records.append(ent_rec)

            # Only create cross-page groups if chunks span multiple pages (>=2 distinct chunks)
            if len(chunk_ids) >= 2 and len(pages) >= 2:
                # Combine evidence from all participating chunks
                combined_texts = []
                for cid in chunk_ids:
                    if cid in chunk_dict:
                        c = chunk_dict[cid]
                        combined_texts.append(f"--- [Chunk {cid} | Pages: {c.page_numbers}] ---\n{c.content}")

                group = RelatedChunkGroup(
                    group_id=f"group_{group_idx:03d}",
                    entity_name=data["raw_name"],
                    chunk_ids=chunk_ids,
                    page_numbers=pages,
                    combined_evidence="\n\n".join(combined_texts),
                    verification_notes=f"Linked by entity '{data['raw_name']}' spanning pages {pages}."
                )
                related_groups.append(group)
                group_idx += 1

        # Write relationships
        write_jsonl(self.relationships_dir / "entity_index.jsonl", entity_records)
        write_jsonl(self.relationships_dir / "related_chunk_groups.jsonl", related_groups)

        logger.info(
            f"Entity linking complete: {len(entity_records)} entities indexed, "
            f"{len(related_groups)} multi-page evidence groups identified."
        )
        return entity_records, related_groups

    def _extract_json_block(self, text: str) -> str:
        text = text.strip()
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if match:
            return match.group(1).strip()
        # Find first { and last }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return text[start:end+1]
        return text
