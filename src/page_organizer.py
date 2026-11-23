from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from src.schemas import PageRecord, TextBlockRecord, TableRecord, ImageRecord, VerifiedRecord
from src.io_utils import ensure_directory, write_jsonl

logger = logging.getLogger("PageOrganizer")

class PageOrganizer:
    """
    Step 4: Page-Level Organization
    PDF Requirements:
    - Intermediate organization step: assemble page records containing text blocks,
      OCR blocks, table references, image references, and captions.
    - Save to pages/page_records.jsonl.
    """
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.pages_dir = self.output_dir / "pages"
        ensure_directory(self.pages_dir)

    def assemble_pages(
        self,
        text_blocks: Optional[List[TextBlockRecord]] = None,
        tables: Optional[List[TableRecord]] = None,
        images: Optional[List[ImageRecord]] = None,
        verified_records: Optional[List[VerifiedRecord]] = None,
        text_records: Optional[List[TextBlockRecord]] = None
    ) -> List[PageRecord]:
        logger.info("Assembling page-level unified records...")
        blocks_to_use = text_blocks if text_blocks is not None else (text_records or [])
        tables = tables or []
        images = images or []

        # If verified records provided, map verified text back into text blocks
        verified_map = {}
        if verified_records:
            for vr in verified_records:
                # Strip prefix 'ver_' to get original text_id
                t_id = vr.record_id.replace("ver_", "")
                verified_map[t_id] = vr

        # Create updated blocks with verified text propagated
        processed_blocks = []
        for tb in blocks_to_use:
            if tb.text_id in verified_map:
                vr = verified_map[tb.text_id]
                updated_tb = TextBlockRecord(
                    text_id=tb.text_id,
                    page_number=tb.page_number,
                    text=vr.corrected_text,  # Propagate verified & normalized text
                    source=tb.source,
                    bbox=tb.bbox,
                    reading_order_idx=tb.reading_order_idx,
                    confidence=tb.confidence,
                    verified=(vr.verification_status == "PASS")
                )
                processed_blocks.append(updated_tb)
            else:
                processed_blocks.append(tb)

        # Find total page span
        all_pages = set()
        for t in processed_blocks:
            all_pages.add(t.page_number)
        for tab in tables:
            all_pages.add(tab.page_number)
        for img in images:
            all_pages.add(img.page_number)

        sorted_pages = sorted(list(all_pages))
        page_records: List[PageRecord] = []

        for p_num in sorted_pages:
            p_texts = [t for t in processed_blocks if t.page_number == p_num]
            # Sort by reading order
            p_texts.sort(key=lambda x: x.reading_order_idx)

            p_tables = [tab.table_id for tab in tables if tab.page_number == p_num]
            p_images = [img.image_id for img in images if img.page_number == p_num]

            char_count = sum(len(t.text) for t in p_texts)
            has_ocr = any(t.source == "ocr" for t in p_texts)

            rec = PageRecord(
                page_number=p_num,
                text_blocks=p_texts,
                table_ids=p_tables,
                image_ids=p_images,
                total_characters=char_count,
                has_ocr_content=has_ocr
            )
            page_records.append(rec)

        out_file = self.pages_dir / "page_records.jsonl"
        write_jsonl(out_file, page_records)
        logger.info(f"Page organization complete: {len(page_records)} pages saved to {out_file}")
        return page_records
