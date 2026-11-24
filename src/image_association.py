from __future__ import annotations
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from src.schemas import ImageRecord, TextBlockRecord
from src.io_utils import ensure_directory, write_jsonl

logger = logging.getLogger("ImageAssociation")

class ImageAssociationManager:
    """
    Step 3: Image Understanding & Text Association
    PDF Requirements:
    - Qwen-VL or image bytes are used to describe images when visual models are present.
    - Confirm association using captions, figure labels, nearby text, explicit figure references.
    - If caption/figure reference does not explicitly confirm identity, mark status as 'uncertain'.
    - Keep uncertain images separate rather than forcing a false relationship.
    - Store: image_id, path, page_number, description, caption, related_text_ids, association_status.
    """
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.extracted_dir = self.output_dir / "extracted"

    def associate_images_with_text(
        self,
        images: List[ImageRecord],
        text_blocks: List[TextBlockRecord],
        vllm_client: Optional[Any] = None
    ) -> List[ImageRecord]:
        logger.info(f"Processing text association for {len(images)} images...")

        # Build text lookup by page
        text_by_page: Dict[int, List[TextBlockRecord]] = {}
        for tb in text_blocks:
            text_by_page.setdefault(tb.page_number, []).append(tb)

        updated_images: List[ImageRecord] = []

        for img in images:
            page_num = img.page_number
            page_texts = text_by_page.get(page_num, [])

            # 1. Search for caption patterns: "Figure 1", "Fig. 2", "Chart 3", "Image 1"
            found_caption: Optional[str] = None
            related_ids: List[str] = []
            association_status = "uncertain"

            for tb in page_texts:
                match = re.search(
                    r'(figure|fig\.|chart|diagram|image|table)\s*(\d+|[a-zA-Z])[:\.\-–]\s*(.*)',
                    tb.text,
                    re.IGNORECASE
                )
                if match:
                    found_caption = tb.text.strip()
                    related_ids.append(tb.text_id)
                    association_status = "verified"
                    break

            # 2. Check nearby text references on the same page
            if not found_caption:
                for tb in page_texts:
                    if re.search(r'\b(shown in figure|as illustrated|see image|refer to figure)\b', tb.text, re.IGNORECASE):
                        related_ids.append(tb.text_id)
                        association_status = "uncertain"

            # 3. Vision Description via Qwen-VL if available
            description = img.description
            if not description and vllm_client and hasattr(vllm_client, "describe_image"):
                try:
                    desc = vllm_client.describe_image(img.image_path)
                    description = desc
                except Exception as e:
                    logger.warning(f"Failed to generate VL description for {img.image_id}: {e}")
            
            if not description:
                description = f"Image extracted from page {img.page_number} (dimensions: {img.width}x{img.height})."

            img.caption = found_caption
            img.related_text_ids = related_ids
            img.association_status = association_status
            img.description = description

            updated_images.append(img)

        # Update saved records
        write_jsonl(self.extracted_dir / "image_records.jsonl", updated_images)
        logger.info(f"Image association finished. Updated {len(updated_images)} records.")
        return updated_images
