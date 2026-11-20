import re
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from src.schemas import ChunkRecord, PageRecord, TableRecord, ImageRecord
from src.io_utils import ensure_directory, write_jsonl

logger = logging.getLogger("HybridChunker")

class HybridChunker:
    """
    Step 5: Structure-Aware Hybrid Chunking Strategy (Page 4 of PDF)
    1. Structure first: headings, paragraphs, lists, captions, and table boundaries.
    2. Split long paragraphs into sentences (regex / spaCy).
    3. Optional Sentence Transformers embeddings & cosine similarity for topic shifts:
       - Computes sentence embeddings.
       - A drop in cosine similarity between adjacent sentences signals a topic shift.
    4. Combine adjacent sentences discussing the same topic.
    5. Token limits enforced (400, 600, 800 Qwen tokens).
    6. Overlap around 10% (40-50, 50-60, 70-80 tokens).
    7. Runs Chunk-Size Experiments (A: 400, B: 600, C: 800) and saves comparison report.
    8. Retains chunk IDs, page numbers, image refs, table refs, source_text_ids.
    """
    def __init__(
        self,
        output_dir: str | Path,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        tokenizer: Optional[Any] = None,
        embedding_model_name: Optional[str] = "BAAI/bge-small-en-v1.5",
        use_embeddings: bool = True
    ):
        self.output_dir = Path(output_dir)
        self.chunks_dir = self.output_dir / "chunks"
        ensure_directory(self.chunks_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tokenizer
        self.embedding_model_name = embedding_model_name
        self.use_embeddings = use_embeddings

        self._embedder = None
        if self.use_embeddings:
            self._init_embedder()

    def _init_embedder(self):
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SentenceTransformers for topic shift detection: {self.embedding_model_name}")
            self._embedder = SentenceTransformer(self.embedding_model_name)
        except Exception as e:
            logger.warning(f"SentenceTransformers fallback (heuristic topic boundary will be used): {e}")
            self._embedder = None

    def count_tokens(self, text: str) -> int:
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        return max(1, len(text) // 4)

    def split_into_sentences(self, text: str) -> List[str]:
        # Split on sentence boundaries preserving abbreviations and numbers
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', text)
        return [s.strip() for s in sentences if s.strip()]

    def compute_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 1.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def detect_topic_shifts(self, sentences: List[str], threshold: float = 0.60) -> List[int]:
        """
        Calculates cosine similarity drops between adjacent sentences as topic boundary hints (PDF Page 4).
        """
        shift_indices = []
        if not self._embedder or len(sentences) < 2:
            return shift_indices

        try:
            embeddings = self._embedder.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
            for i in range(len(embeddings) - 1):
                sim = self.compute_cosine_similarity(embeddings[i], embeddings[i+1])
                # A sharp drop in similarity signals a topic boundary
                if sim < threshold:
                    shift_indices.append(i + 1)
        except Exception as e:
            logger.debug(f"Topic shift detection skipped: {e}")
        return shift_indices

    def _extract_linear_units(
        self,
        page_records: List[PageRecord],
        tables: List[TableRecord]
    ) -> List[Dict[str, Any]]:
        table_dict = {t.table_id: t for t in tables}
        units = []

        for page in page_records:
            for tb in page.text_blocks:
                sentences = self.split_into_sentences(tb.text)
                topic_shifts = set(self.detect_topic_shifts(sentences))
                for s_idx, s in enumerate(sentences):
                    units.append({
                        "type": "text",
                        "content": s,
                        "page_number": page.page_number,
                        "text_id": tb.text_id,
                        "tokens": self.count_tokens(s),
                        "is_topic_boundary": (s_idx in topic_shifts)
                    })
            for t_id in page.table_ids:
                if t_id in table_dict:
                    tab = table_dict[t_id]
                    # Include footnotes in table unit content
                    tab_content = tab.markdown_content
                    if tab.footnotes:
                        tab_content += "\nFootnotes: " + "; ".join(tab.footnotes)

                    units.append({
                        "type": "table",
                        "content": f"\n\n[Table {t_id} (Page {tab.page_number})]:\n{tab_content}\n\n",
                        "page_number": tab.page_number,
                        "table_id": t_id,
                        "tokens": self.count_tokens(tab_content),
                        "is_topic_boundary": True
                    })
        return units

    def _build_chunks_for_target(
        self,
        units: List[Dict[str, Any]],
        target_size: int,
        target_overlap: int,
        tag: str
    ) -> List[ChunkRecord]:
        chunks: List[ChunkRecord] = []
        chunk_idx = 1
        curr_indices: List[int] = []
        curr_tokens = 0
        i = 0

        while i < len(units):
            u = units[i]
            u_tokens = u["tokens"]

            # Break chunk if target size reached or natural topic boundary with sufficient length
            should_break = (curr_tokens + u_tokens > target_size and curr_indices) or (
                u.get("is_topic_boundary", False) and curr_tokens >= (target_size * 0.75)
            )

            if should_break:
                chunk_rec = self._assemble_chunk(chunk_idx, curr_indices, units, tag)
                chunks.append(chunk_rec)
                chunk_idx += 1

                # Calculate overlap
                overlap_accum = 0
                backtrack_pos = i
                for j in reversed(curr_indices):
                    overlap_accum += units[j]["tokens"]
                    if overlap_accum >= target_overlap:
                        backtrack_pos = j
                        break

                curr_indices = []
                curr_tokens = 0
                i = backtrack_pos
            else:
                curr_indices.append(i)
                curr_tokens += u_tokens
                i += 1

        if curr_indices:
            chunks.append(self._assemble_chunk(chunk_idx, curr_indices, units, tag))

        return chunks

    def _assemble_chunk(
        self,
        chunk_num: int,
        indices: List[int],
        all_units: List[Dict[str, Any]],
        tag: str
    ) -> ChunkRecord:
        content_parts = []
        pages = set()
        text_ids = []
        table_refs = []

        for idx in indices:
            u = all_units[idx]
            content_parts.append(u["content"])
            pages.add(u["page_number"])
            if "text_id" in u and u["text_id"] not in text_ids:
                text_ids.append(u["text_id"])
            if "table_id" in u and u["table_id"] not in table_refs:
                table_refs.append(u["table_id"])

        combined_text = " ".join(content_parts).strip()
        tokens = self.count_tokens(combined_text)

        return ChunkRecord(
            chunk_id=f"chunk_{tag}_{chunk_num:03d}",
            page_numbers=sorted(list(pages)),
            content=combined_text,
            image_refs=[],
            table_refs=table_refs,
            token_count=tokens,
            source_text_ids=text_ids,
            experiment_tag=tag
        )

    def run_chunking_experiments(
        self,
        page_records: List[PageRecord],
        tables: List[TableRecord]
    ) -> Dict[str, Any]:
        """
        Task 6: Chunk and compare 400/600/800 token settings.
        Outputs comparison report to chunks/chunk_experiments_comparison.json.
        """
        logger.info("Executing Chunk-Size Experiments (400 vs 600 vs 800 Qwen tokens)...")
        units = self._extract_linear_units(page_records, tables)

        exp_configs = [
            {"tag": "exp_400", "size": 400, "overlap": 45},
            {"tag": "exp_600", "size": 600, "overlap": 55},
            {"tag": "exp_800", "size": 800, "overlap": 75}
        ]

        results = {}
        for cfg in exp_configs:
            chunks = self._build_chunks_for_target(units, cfg["size"], cfg["overlap"], cfg["tag"])
            token_counts = [c.token_count for c in chunks]
            avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
            results[cfg["tag"]] = {
                "target_size": cfg["size"],
                "overlap": cfg["overlap"],
                "total_chunks": len(chunks),
                "avg_tokens_per_chunk": round(avg_tokens, 2),
                "min_tokens": min(token_counts) if token_counts else 0,
                "max_tokens": max(token_counts) if token_counts else 0
            }

        comparison_file = self.chunks_dir / "chunk_experiments_comparison.json"
        with open(comparison_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Chunk experiments comparison saved to: {comparison_file}")
        return results

    def chunk_document(
        self,
        page_records: List[PageRecord],
        tables: List[TableRecord],
        images: List[ImageRecord]
    ) -> List[ChunkRecord]:
        logger.info(f"Generating primary chunks (Target: {self.chunk_size} tokens, Overlap: {self.chunk_overlap})...")
        units = self._extract_linear_units(page_records, tables)
        
        chunks = self._build_chunks_for_target(units, self.chunk_size, self.chunk_overlap, "primary")

        # Map images on same pages to image_refs
        img_by_page = {}
        for img in images:
            img_by_page.setdefault(img.page_number, []).append(img.image_id)

        for c in chunks:
            for p in c.page_numbers:
                if p in img_by_page:
                    c.image_refs.extend(img_by_page[p])

        out_file = self.chunks_dir / "chunks.jsonl"
        write_jsonl(out_file, chunks)
        logger.info(f"Primary chunks saved: {len(chunks)} chunks -> {out_file}")

        # Run 400/600/800 token experiment comparison
        self.run_chunking_experiments(page_records, tables)

        return chunks
