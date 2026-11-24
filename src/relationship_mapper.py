import re
import json
import logging
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Tuple, Set

from src.schemas import (
    TextBlockRecord, TableRecord, ImageRecord,
    KPIBoxRecord, ChartRecord, ParagraphRecord,
    RelationshipRecord, ContentGroupRecord
)
from src.io_utils import ensure_directory, write_jsonl

logger = logging.getLogger("RelationshipMapper")

class RelationshipMapper:
    """
    Phase 5: Relationship Mapping
    Identifies related extracted content records across document hierarchy,
    section/subsection headings, table titles, chart titles, captions,
    cross-references, page continuity, and semantic similarity.
    Creates relationships.json and content_groups.json.
    """
    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.extracted_dir = self.output_dir / "extracted_data"
        self.processed_dir = self.output_dir / "processed_data"
        ensure_directory(self.extracted_dir)
        ensure_directory(self.processed_dir)

    def _normalize_text(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip().lower()

    def _extract_keywords(self, text: str) -> Set[str]:
        words = re.findall(r'[a-zA-Z]{3,}', text.lower())
        stopwords = {
            'the', 'and', 'for', 'with', 'from', 'this', 'that', 'were', 'have',
            'been', 'also', 'will', 'such', 'more', 'about', 'than', 'into',
            'annual', 'report', 'tracxn', 'technologies', 'limited', 'page', 'table'
        }
        return set(words) - stopwords

    def map_relationships(
        self,
        paragraphs: List[ParagraphRecord],
        tables: List[TableRecord],
        charts: List[ChartRecord],
        kpis: List[KPIBoxRecord],
        images: List[ImageRecord]
    ) -> Tuple[List[RelationshipRecord], List[ContentGroupRecord]]:
        """
        Builds fine-grained pairwise relationships and aggregates them into coherent
        content groups (e.g. GROUP_001 - Operating Expenses).
        """
        relationships: List[RelationshipRecord] = []
        rel_counter = 1

        # 1. Map all items with section/heading tags
        items_by_page: Dict[int, List[Dict[str, Any]]] = {}
        all_items: List[Dict[str, Any]] = []

        for p in paragraphs:
            entry = {
                "id": p.para_id, "type": "paragraph", "page": p.page_number,
                "text": p.text, "heading": p.section_heading or ""
            }
            all_items.append(entry)
            items_by_page.setdefault(p.page_number, []).append(entry)

        for t in tables:
            entry = {
                "id": t.table_id, "type": "table", "page": t.page_number,
                "text": f"{t.table_title or ''} {t.markdown_content}",
                "heading": t.table_title or "Financial Table"
            }
            all_items.append(entry)
            items_by_page.setdefault(t.page_number, []).append(entry)

        for c in charts:
            entry = {
                "id": c.chart_id, "type": "chart", "page": c.page_number,
                "text": f"{c.title or ''} {' '.join(c.legends)} {' '.join(c.data_labels)}",
                "heading": c.title or "Chart & Graph"
            }
            all_items.append(entry)
            items_by_page.setdefault(c.page_number, []).append(entry)

        for k in kpis:
            entry = {
                "id": k.kpi_id, "type": "kpi", "page": k.page_number,
                "text": f"{k.value} {k.label}",
                "heading": k.section_heading or "KPI Metric"
            }
            all_items.append(entry)
            items_by_page.setdefault(k.page_number, []).append(entry)

        for img in images:
            entry = {
                "id": img.image_id, "type": "image", "page": img.page_number,
                "text": f"{img.caption or ''} {img.description or ''}",
                "heading": img.caption or "Image"
            }
            all_items.append(entry)
            items_by_page.setdefault(img.page_number, []).append(entry)

        # 2. Pairwise Relationship Discovery
        # Criteria:
        # A. Heading match: Same normalized section heading
        # B. Caption / Table reference: Mention of table ID / title in text
        # C. Page continuity + shared key concepts
        for i in range(len(all_items)):
            item_a = all_items[i]
            words_a = self._extract_keywords(item_a["text"] + " " + item_a["heading"])
            if not words_a:
                continue

            for j in range(i + 1, min(len(all_items), i + 40)):  # Check nearby items across pages
                item_b = all_items[j]
                if item_a["id"] == item_b["id"]:
                    continue

                words_b = self._extract_keywords(item_b["text"] + " " + item_b["heading"])
                overlap = words_a.intersection(words_b)
                reasons = []
                rel_type = "semantic_similarity"
                status = "uncertain"
                confidence = 0.70

                # Check Heading Match
                if item_a["heading"] and item_b["heading"] and self._normalize_text(item_a["heading"]) == self._normalize_text(item_b["heading"]):
                    reasons.append(f"Exact matching section heading: '{item_a['heading']}'")
                    rel_type = "heading_match"
                    status = "verified"
                    confidence = 0.95

                # Check Cross-Reference (e.g. text mentions "Note 3" or table title)
                if len(item_a["heading"]) > 5 and item_a["heading"].lower() in item_b["text"].lower():
                    reasons.append(f"Text in {item_b['id']} directly references heading '{item_a['heading']}'")
                    rel_type = "table_reference" if item_a["type"] == "table" else "section_hierarchy"
                    status = "verified"
                    confidence = 0.90

                # Check Page Continuity with Significant Keyword Overlap
                page_diff = abs(item_a["page"] - item_b["page"])
                if page_diff <= 2 and len(overlap) >= 3:
                    reasons.append(f"Page continuity (pages {item_a['page']} & {item_b['page']}) with shared concepts: {sorted(list(overlap))[:4]}")
                    if status != "verified":
                        status = "verified" if len(overlap) >= 5 else "uncertain"
                        confidence = 0.85 if status == "verified" else 0.75
                        rel_type = "cross_page_continuity"

                if reasons:
                    relationships.append(
                        RelationshipRecord(
                            relationship_id=f"REL_{rel_counter:04d}",
                            source_record_id=item_a["id"],
                            target_record_id=item_b["id"],
                            relationship_type=rel_type,
                            reasons=reasons,
                            status=status,
                            confidence=confidence
                        )
                    )
                    rel_counter += 1

        # 3. Aggregate into Coherent Content Groups
        # Group items by section heading or shared verified relationships
        groups: List[ContentGroupRecord] = []
        heading_groups: Dict[str, List[Dict[str, Any]]] = {}

        for item in all_items:
            heading = item["heading"] or "General Information"
            # Normalize common headings
            norm_heading = self._normalize_text(heading)
            if not norm_heading:
                norm_heading = f"Page_{item['page']}"
            heading_groups.setdefault(norm_heading, []).append(item)

        group_idx = 1
        for norm_heading, members in heading_groups.items():
            if not members:
                continue

            rec_ids = [m["id"] for m in members]
            page_nums = sorted(list({m["page"] for m in members}))
            display_title = members[0]["heading"] if members[0]["heading"] else norm_heading.title()

            reasons = [
                f"Content belongs to section '{display_title}'",
                f"Spans {len(page_nums)} page(s): {page_nums[:5]}",
                f"Contains {len(members)} mixed records: {list({m['type'] for m in members})}"
            ]

            groups.append(
                ContentGroupRecord(
                    group_id=f"GROUP_{group_idx:03d}",
                    topic=display_title,
                    section=display_title,
                    source_record_ids=rec_ids,
                    chunk_ids=[],  # Populated during chunking
                    page_numbers=page_nums,
                    relationship_reasons=reasons,
                    status="verified" if len(members) >= 2 else "uncertain",
                    confidence=0.92 if len(members) >= 2 else 0.75
                )
            )
            group_idx += 1

        # Save outputs to both extracted_data and processed_data
        self._save_relationship_files(relationships, groups)
        return relationships, groups

    def _save_relationship_files(
        self,
        relationships: List[RelationshipRecord],
        groups: List[ContentGroupRecord]
    ):
        rel_data = [r.model_dump() for r in relationships]
        grp_data = [g.model_dump() for g in groups]

        # In extracted_data/
        with open(self.extracted_dir / "relationships.json", "w", encoding="utf-8") as f:
            json.dump(rel_data, f, indent=2, ensure_ascii=False)

        # In processed_data/
        with open(self.processed_dir / "relationships.json", "w", encoding="utf-8") as f:
            json.dump(rel_data, f, indent=2, ensure_ascii=False)

        with open(self.processed_dir / "content_groups.json", "w", encoding="utf-8") as f:
            json.dump(grp_data, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Relationship mapping complete: {len(relationships)} relationships, "
            f"{len(groups)} content groups generated."
        )
