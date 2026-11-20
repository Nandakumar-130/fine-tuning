from pathlib import Path
from typing import List, Dict, Any
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
        text_blocks: List[TextBlockRecord],
        tables: List[TableRecord],
        images: List[ImageRecord]
    ) -> List[PageRecord]:
        logger.info("Assembling page-level unified records...")

        # Find total page span
        all_pages = set()
        for t in text_blocks:
            all_pages.add(t.page_number)
        for tab in tables:
            all_pages.add(tab.page_number)
        for img in images:
            all_pages.add(img.page_number)

        sorted_pages = sorted(list(all_pages))
        page_records: List[PageRecord] = []

        for p_num in sorted_pages:
            p_texts = [t for t in text_blocks if t.page_number == p_num]
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
