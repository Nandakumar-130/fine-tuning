import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
from src.io_utils import ensure_directory, write_jsonl

logger = logging.getLogger("ContentInventory")

class ContentInventoryScanner:
    """
    Task 1: Inventory PDF Content Types
    Output: Page/content inventory (saved to original/content_inventory.jsonl)
    Surveys:
    - Native selectable text blocks & characters
    - Image objects & xrefs
    - Drawing paths / vector graphics
    - Tables / structured grids
    - Page dimensions & rotation
    """
    def __init__(self, pdf_path: str | Path, output_dir: str | Path):
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path(output_dir)
        self.original_dir = self.output_dir / "original"
        ensure_directory(self.original_dir)

    def scan_inventory(self, page_range: Optional[Tuple[int, int]] = None) -> List[Dict[str, Any]]:
        logger.info(f"Inventorying content types for: {self.pdf_path.name}")
        doc = fitz.open(self.pdf_path)
        inventory: List[Dict[str, Any]] = []

        total_pages = len(doc)
        start_idx = 0
        end_idx = total_pages
        if page_range:
            start_idx = max(0, page_range[0] - 1)
            end_idx = min(total_pages, page_range[1])

        total_text_blocks = 0
        total_images = 0
        total_tables = 0

        for page_idx in range(start_idx, end_idx):
            page_num = page_idx + 1
            page = doc[page_idx]

            # Text blocks
            blocks = page.get_text("blocks")
            text_blocks = [b for b in blocks if b[6] == 0]
            raw_text = page.get_text()
            char_count = len(raw_text.strip())

            # Images
            img_list = page.get_images(full=True)

            # Tables
            tables = []
            try:
                tabs = page.find_tables()
                tables = list(tabs)
            except Exception:
                tables = []

            # Drawings
            drawings = page.get_drawings()

            # Classify page primary nature
            has_scanned = (len(text_blocks) == 0 and len(img_list) > 0)
            page_type = "scanned" if has_scanned else ("mixed" if len(img_list) > 0 else "text_heavy")

            page_info = {
                "page_number": page_num,
                "page_type": page_type,
                "dimensions": [float(page.rect.width), float(page.rect.height)],
                "rotation": page.rotation,
                "native_text_blocks": len(text_blocks),
                "character_count": char_count,
                "embedded_images_count": len(img_list),
                "detected_tables_count": len(tables),
                "vector_drawings_count": len(drawings),
                "requires_ocr": (char_count < 50 and len(img_list) > 0)
            }
            inventory.append(page_info)
            total_text_blocks += len(text_blocks)
            total_images += len(img_list)
            total_tables += len(tables)

        doc.close()

        out_file = self.original_dir / "content_inventory.jsonl"
        write_jsonl(out_file, inventory)
        logger.info(
            f"Content inventory completed: {total_pages} pages scanned "
            f"({total_text_blocks} text blocks, {total_images} images, {total_tables} tables). "
            f"Saved to: {out_file}"
        )
        return inventory

    # Alias for method name consistency
    scan_document = scan_inventory
