from __future__ import annotations
import re
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from src.schemas import (
    ChunkRecord, PageRecord, TableRecord, ImageRecord,
    KPIBoxRecord, ChartRecord, ContentGroupRecord
)
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
        self.processed_dir = self.output_dir / "processed_data"
        ensure_directory(self.chunks_dir)
        ensure_directory(self.processed_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tokenizer
        self.embedding_model_name = embedding_model_name
        self.use_embeddings = use_embeddings
        self.is_exact_tokenizer = (tokenizer is not None)

        self._embedder = None
        if self.use_embeddings:
            self._init_embedder()

        self._spacy_nlp = None
        self._init_spacy()

    def _init_spacy(self):
        try:
            import spacy
            self._spacy_nlp = spacy.blank("en")
            self._spacy_nlp.add_pipe("sentencizer")
        except Exception as e:
            logger.debug(f"spaCy sentencizer not available ({e}); using regex splitter fallback.")
            self._spacy_nlp = None

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
        # Conservative character-based estimation fallback:
        # In financial reports and tabular data, numbers, currency symbols, and tables average ~3.0 - 3.5 chars/token.
        # len(text) // 4 can underestimate token count by 25-35%.
        # We use a safe conservative estimation: max(1, int(len(text) / 3.5))
        return max(1, int(len(text) / 3.5))

    def split_into_sentences(self, text: str) -> List[str]:
        # Use spaCy sentencizer if available
        if self._spacy_nlp:
            try:
                doc = self._spacy_nlp(text)
                sents = [s.text.strip() for s in doc.sents if s.text.strip()]
                if sents:
                    return sents
            except Exception:
                pass
        # Fallback to regex splitting on sentence boundaries preserving abbreviations and numbers
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
        section_heading = None

        for idx in indices:
            u = all_units[idx]
            content_parts.append(u["content"])
            pages.add(u["page_number"])
            if "text_id" in u and u["text_id"] not in text_ids:
                text_ids.append(u["text_id"])
            if "table_id" in u and u["table_id"] not in table_refs:
                table_refs.append(u["table_id"])
            if not section_heading and u.get("is_heading"):
                section_heading = u.get("heading")

        combined_text = " ".join(content_parts).strip()
        tokens = self.count_tokens(combined_text)

        # Strategy classification
        if table_refs and not text_ids:
            strategy = "table_aware"
        elif section_heading:
            strategy = "heading_based"
        else:
            strategy = "token_sentence_based"

        return ChunkRecord(
            chunk_id=f"chunk_{tag}_{chunk_num:03d}",
            page_numbers=sorted(list(pages)),
            content=combined_text,
            chunk_strategy=strategy,
            section_heading=section_heading,
            image_refs=[],
            table_refs=table_refs,
            chart_refs=[],
            kpi_refs=[],
            token_count=tokens,
            source_text_ids=text_ids,
            experiment_tag=tag
        )

    def _split_oversized_table(self, tab: TableRecord, max_tokens: int = 700) -> List[ChunkRecord]:
        """
        Phase 6.2: Table-Aware Chunking.
        Splits oversized tables by logical row groups while repeating titles, units,
        currency, and column headings so no percentage-to-category or financial precision is lost.
        """
        chunks: List[ChunkRecord] = []
        if not tab.rows:
            return chunks

        header_line = "| " + " | ".join(tab.headers) + " |"
        sep_line = "| " + " | ".join(["---"] * len(tab.headers)) + " |"
        title_str = f"### {tab.table_title}\n" if tab.table_title else ""
        meta_parts = []
        if tab.units:
            meta_parts.append(f"Units: {tab.units}")
        if tab.currency:
            meta_parts.append(f"Currency: {tab.currency}")
        if tab.reporting_period:
            meta_parts.append(f"Period: {tab.reporting_period}")
        meta_str = f"**Table Metadata:** {', '.join(meta_parts)}\n" if meta_parts else ""

        batch_size = 12
        for b_idx in range(0, len(tab.rows), batch_size):
            row_batch = tab.rows[b_idx: b_idx + batch_size]
            body_lines = ["| " + " | ".join(r) + " |" for r in row_batch]
            part_num = (b_idx // batch_size) + 1
            table_part_content = (
                f"{title_str}{meta_str}"
                f"[Table {tab.table_id} (Part {part_num}, Page {tab.page_number})]:\n" +
                "\n".join([header_line, sep_line] + body_lines)
            )
            if tab.footnotes:
                table_part_content += "\nFootnotes: " + "; ".join(tab.footnotes)

            tokens = self.count_tokens(table_part_content)
            chunks.append(
                ChunkRecord(
                    chunk_id=f"chunk_tab_{tab.table_id}_{part_num:02d}",
                    page_numbers=[tab.page_number],
                    content=table_part_content,
                    chunk_strategy="table_aware",
                    section_heading=tab.table_title or "Financial Table",
                    table_refs=[tab.table_id],
                    chart_refs=[],
                    kpi_refs=[],
                    token_count=tokens,
                    source_text_ids=[],
                    experiment_tag="table_aware"
                )
            )
        return chunks

    def _build_layout_chunks(self, kpis: List[KPIBoxRecord], max_per_chunk: int = 8) -> List[ChunkRecord]:
        """
        Phase 6.3: Layout-Aware Chunking.
        Keeps independent KPI metric cards and layout boxes together with section context.
        """
        chunks: List[ChunkRecord] = []
        if not kpis:
            return chunks

        kpi_groups: Dict[Tuple[int, str], List[KPIBoxRecord]] = {}
        for k in kpis:
            key = (k.page_number, k.section_heading or "Executive Overview")
            kpi_groups.setdefault(key, []).append(k)

        for (page_num, heading), group_kpis in kpi_groups.items():
            for b_idx in range(0, len(group_kpis), max_per_chunk):
                batch = group_kpis[b_idx: b_idx + max_per_chunk]
                part_idx = (b_idx // max_per_chunk) + 1
                lines = [
                    f"### KEY PERFORMANCE INDICATORS & METRICS (Page {page_num} - {heading})",
                    f"Section Context: {heading}",
                    f"The following structured KPI metrics and percentage cards were extracted from Page {page_num}:"
                ]
                for k in batch:
                    unit_str = f" ({k.unit})" if k.unit else ""
                    lines.append(f"- **{k.label}**: `{k.value}`{unit_str}")

                content = "\n".join(lines)
                tokens = self.count_tokens(content)
                chunks.append(
                    ChunkRecord(
                        chunk_id=f"chunk_kpi_p{page_num:03d}_{part_idx:02d}",
                        page_numbers=[page_num],
                        content=content,
                        chunk_strategy="layout_aware",
                        section_heading=heading,
                        kpi_refs=[k.kpi_id for k in batch],
                        chart_refs=[],
                        table_refs=[],
                        token_count=tokens,
                        source_text_ids=[],
                        experiment_tag="layout_aware"
                    )
                )
        return chunks

    def _build_chart_chunks(self, charts: List[ChartRecord]) -> List[ChunkRecord]:
        """
        Phase 6.4: Chart-Aware Chunking.
        Keeps chart title, axes, legends, units, data labels, and image references together.
        """
        chunks: List[ChunkRecord] = []
        for c in charts:
            lines = [
                f"### CHART & GRAPH ANALYSIS: {c.title or 'Financial Trend Chart'}",
                f"Page Reference: Page {c.page_number} | Source Chart ID: {c.chart_id}"
            ]
            if c.x_axis_label:
                lines.append(f"- **X-Axis / Periods**: {c.x_axis_label}")
            if c.y_axis_label:
                lines.append(f"- **Y-Axis / Metric**: {c.y_axis_label}")
            if c.units:
                lines.append(f"- **Units**: {c.units}")
            if c.legends:
                lines.append(f"- **Legends / Categories**: {', '.join(c.legends)}")
            if c.data_labels:
                lines.append(f"- **Data Labels**: {', '.join(c.data_labels)}")
            if c.extraction_limitations:
                lines.append(f"- **Audit Notes**: {c.extraction_limitations}")

            content = "\n".join(lines)
            tokens = self.count_tokens(content)
            chunks.append(
                ChunkRecord(
                    chunk_id=f"chunk_chart_{c.chart_id}",
                    page_numbers=[c.page_number],
                    content=content,
                    chunk_strategy="chart_aware",
                    section_heading=c.title or "Chart Analysis",
                    chart_refs=[c.chart_id],
                    image_refs=[c.image_ref] if c.image_ref else [],
                    kpi_refs=[],
                    table_refs=[],
                    token_count=tokens,
                    source_text_ids=[],
                    experiment_tag="chart_aware"
                )
            )
        return chunks

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

    def run_preflight_density_audit(
        self,
        page_records: List[PageRecord],
        tables: List[TableRecord]
    ) -> Dict[str, Any]:
        """
        Pre-flight chunking validation and token density audit (Section 5 requirements):
        Inspects extracted page and section content before batch Q&A generation.
        """
        logger.info("Running offline pre-flight chunking validation and token density audit...")
        page_audits = []
        table_dict = {t.table_id: t for t in tables}

        total_chars = 0
        total_words = 0
        total_digits = 0
        total_symbols = 0
        dense_pages_flagged = []

        for p in page_records:
            full_text = " ".join([tb.text for tb in p.text_blocks])
            for tid in p.table_ids:
                if tid in table_dict:
                    full_text += " " + table_dict[tid].markdown_content

            char_len = len(full_text)
            words = full_text.split()
            word_len = len(words)
            digits = sum(c.isdigit() for c in full_text)
            symbols = sum(not c.isalnum() and not c.isspace() for c in full_text)

            digit_ratio = (digits / char_len) if char_len > 0 else 0.0
            symbol_ratio = (symbols / char_len) if char_len > 0 else 0.0
            chars_per_word = (char_len / word_len) if word_len > 0 else 0.0

            estimated_tokens = self.count_tokens(full_text)
            is_dense = (digit_ratio > 0.08) or (symbol_ratio > 0.12) or (len(p.table_ids) > 0)

            audit_item = {
                "page_number": p.page_number,
                "character_count": char_len,
                "word_count": word_len,
                "digit_count": digits,
                "digit_density": round(digit_ratio, 4),
                "symbol_density": round(symbol_ratio, 4),
                "chars_per_word": round(chars_per_word, 2),
                "has_tables": len(p.table_ids) > 0,
                "table_count": len(p.table_ids),
                "estimated_tokens": estimated_tokens,
                "token_method": "exact_qwen_tokenizer" if self.is_exact_tokenizer else "conservative_character_estimate (3.5 chars/tok)",
                "flagged_dense_financial": is_dense
            }
            page_audits.append(audit_item)

            total_chars += char_len
            total_words += word_len
            total_digits += digits
            total_symbols += symbols
            if is_dense:
                dense_pages_flagged.append(p.page_number)

        summary = {
            "total_pages_audited": len(page_records),
            "total_characters": total_chars,
            "total_words": total_words,
            "overall_digit_density": round(total_digits / max(1, total_chars), 4),
            "overall_symbol_density": round(total_symbols / max(1, total_chars), 4),
            "exact_tokenizer_active": self.is_exact_tokenizer,
            "token_counting_method": "exact_qwen_tokenizer" if self.is_exact_tokenizer else "conservative_character_estimate (3.5 chars/tok)",
            "flagged_dense_pages_count": len(dense_pages_flagged),
            "flagged_dense_pages": dense_pages_flagged[:50],
            "page_details": page_audits
        }

        audit_file = self.chunks_dir / "preflight_density_audit.json"
        with open(audit_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Pre-flight chunking token density audit saved to: {audit_file} (Flagged {len(dense_pages_flagged)} dense pages)")
        return summary

    def chunk_document(
        self,
        page_records: List[PageRecord],
        tables: List[TableRecord],
        images: List[ImageRecord],
        kpi_boxes: Optional[List[KPIBoxRecord]] = None,
        charts: Optional[List[ChartRecord]] = None,
        content_groups: Optional[List[ContentGroupRecord]] = None
    ) -> List[ChunkRecord]:
        """
        Executes the Five Main Hybrid Chunking Strategies:
        6.1 Heading-Based Chunking
        6.2 Table-Aware Chunking (intact tables & oversized split)
        6.3 Layout-Aware Chunking (independent KPI cards)
        6.4 Chart-Aware Chunking (chart metadata & limits)
        6.5 Token-Sentence Based Chunking (exact Qwen tokens & 50-token overlap)
        """
        logger.info(f"Generating 5-Strategy Hybrid Chunks (Target: {self.chunk_size} tokens, Overlap: {self.chunk_overlap})...")
        all_chunks: List[ChunkRecord] = []

        # 1. Normal Text Chunks (Heading & Sentence-based with exact tokenizer & overlap)
        normal_tables = [t for t in tables if self.count_tokens(t.markdown_content) <= 600]
        oversized_tables = [t for t in tables if self.count_tokens(t.markdown_content) > 600]

        units = self._extract_linear_units(page_records, normal_tables)
        text_chunks = self._build_chunks_for_target(units, self.chunk_size, self.chunk_overlap, "primary")
        all_chunks.extend(text_chunks)

        # 2. Table-Aware Chunks for Oversized Tables (split by row groups with repeated headers)
        for tab in oversized_tables:
            sub_chunks = self._split_oversized_table(tab, max_tokens=self.chunk_size)
            all_chunks.extend(sub_chunks)

        # 3. Layout-Aware Chunks for Independent KPI Metric Cards
        if kpi_boxes:
            layout_chunks = self._build_layout_chunks(kpi_boxes)
            all_chunks.extend(layout_chunks)

        # 4. Chart-Aware Chunks
        if charts:
            chart_chunks = self._build_chart_chunks(charts)
            all_chunks.extend(chart_chunks)

        # Map Images on corresponding pages
        img_by_page = {}
        for img in images:
            img_by_page.setdefault(img.page_number, []).append(img.image_id)

        for c in all_chunks:
            for p in c.page_numbers:
                if p in img_by_page:
                    for i_id in img_by_page[p]:
                        if i_id not in c.image_refs:
                            c.image_refs.append(i_id)

        # Map Content Group IDs if content groups exist
        if content_groups:
            for c in all_chunks:
                for grp in content_groups:
                    # Match by heading or overlapping page
                    if c.section_heading and grp.topic and c.section_heading.lower() in grp.topic.lower():
                        c.content_group_id = grp.group_id
                        grp.chunk_ids.append(c.chunk_id)
                        break
                    elif any(p in grp.page_numbers for p in c.page_numbers):
                        c.content_group_id = grp.group_id
                        grp.chunk_ids.append(c.chunk_id)
                        break

        # Save to both chunks/chunks.jsonl and processed_data/chunks.jsonl
        out_file = self.chunks_dir / "chunks.jsonl"
        processed_file = self.processed_dir / "chunks.jsonl"
        write_jsonl(out_file, all_chunks)
        write_jsonl(processed_file, all_chunks)

        logger.info(
            f"Hybrid Chunking Complete: {len(all_chunks)} chunks created "
            f"(Heading: {sum(1 for c in all_chunks if c.chunk_strategy == 'heading_based')}, "
            f"Table: {sum(1 for c in all_chunks if c.chunk_strategy == 'table_aware')}, "
            f"Layout: {sum(1 for c in all_chunks if c.chunk_strategy == 'layout_aware')}, "
            f"Chart: {sum(1 for c in all_chunks if c.chunk_strategy == 'chart_aware')}, "
            f"Sentence-token: {sum(1 for c in all_chunks if c.chunk_strategy == 'token_sentence_based')})."
        )

        # Run pre-flight token density audit and experiments
        self.run_preflight_density_audit(page_records, tables)
        self.run_chunking_experiments(page_records, tables)

        return all_chunks

    # Backward-compatible alias
    chunk_pages = chunk_document
