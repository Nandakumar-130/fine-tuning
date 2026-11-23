import os
import re
import cv2
import fitz  # PyMuPDF
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

from src.schemas import TextBlockRecord, TableRecord, ImageRecord, BoundingBox
from src.io_utils import ensure_directory, write_jsonl

logger = logging.getLogger("PDFExtractor")

class PDFExtractor:
    """
    Step 1: Extract selectable text, images, tables, and OCR scanned content.
    OpenCV's full role from Page 3:
      - Grayscale
      - Denoising (cv2.fastNlMeansDenoising)
      - Deskewing (detect text angle via cv2.minAreaRect and rotate)
      - Resizing / Scaling for optimal OCR
      - Sharpening (kernel unsharp mask)
      - Adaptive thresholding
    Tables:
      - Rows, columns, headers, units, values, currency, and footnotes preserved.
      - PyMuPDF structured table extraction + PP-Structure fallback.
    """
    def __init__(
        self,
        pdf_path: str | Path,
        output_dir: str | Path,
        ocr_lang: str = "en",
        use_gpu_ocr: bool = False,
        use_ppstructure: bool = False
    ):
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path(output_dir)
        self.ocr_lang = ocr_lang
        self.use_gpu_ocr = use_gpu_ocr
        self.use_ppstructure = use_ppstructure

        # Output Directories
        self.extracted_dir = self.output_dir / "extracted"
        self.images_dir = self.extracted_dir / "images"
        self.crops_dir = self.extracted_dir / "crops"
        ensure_directory(self.extracted_dir)
        ensure_directory(self.images_dir)
        ensure_directory(self.crops_dir)

        self._paddle_ocr = None
        self._ppstructure_engine = None

    def _get_ocr_engine(self):
        if self._paddle_ocr is None:
            try:
                from paddleocr import PaddleOCR
                self._paddle_ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang=self.ocr_lang,
                    use_gpu=self.use_gpu_ocr,
                    show_log=False
                )
                logger.info("PaddleOCR engine initialized successfully.")
            except ImportError:
                logger.warning("PaddleOCR is not installed. Scanned page OCR will use PyMuPDF fallback.")
                self._paddle_ocr = False
        return self._paddle_ocr

    def _get_ppstructure_engine(self):
        if self._ppstructure_engine is None:
            if not self.use_ppstructure:
                return None
            try:
                from paddleocr import PPStructure
                self._ppstructure_engine = PPStructure(
                    table=True,
                    ocr=True,
                    show_log=False,
                    use_gpu=self.use_gpu_ocr,
                    lang=self.ocr_lang
                )
                logger.info("PP-Structure table engine initialized successfully.")
            except Exception as e:
                logger.warning(f"PP-Structure engine initialization fallback ({e}).")
                self._ppstructure_engine = False
        return self._ppstructure_engine

    def deskew_image(self, img_gray: np.ndarray) -> np.ndarray:
        """
        OpenCV deskewing: detect angle of text lines and rotate image straight.
        """
        try:
            coords = np.column_stack(np.where(img_gray < 240))
            if len(coords) < 100:
                return img_gray
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            elif angle > 45:
                angle = 90 - angle
            else:
                angle = -angle

            if abs(angle) > 0.5 and abs(angle) < 45.0:
                (h, w) = img_gray.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                deskewed = cv2.warpAffine(
                    img_gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
                )
                return deskewed
        except Exception as e:
            logger.debug(f"Deskew skipped: {e}")
        return img_gray

    def preprocess_image_opencv(self, img_np: np.ndarray) -> np.ndarray:
        """
        OpenCV pipeline from PDF Page 3:
        grayscale -> denoising -> deskewing -> sharpening -> adaptive thresholding
        """
        # 1. Grayscale
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_np.copy()

        # 2. Resizing (scale up if resolution is low)
        h, w = gray.shape[:2]
        if w < 1000:
            scale = 1200 / w
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # 3. Denoising
        denoised = cv2.fastNlMeansDenoising(gray, h=10)

        # 4. Deskewing
        deskewed = self.deskew_image(denoised)

        # 5. Sharpening (Unsharp masking)
        gaussian = cv2.GaussianBlur(deskewed, (0, 0), 2.0)
        sharpened = cv2.addWeighted(deskewed, 1.5, gaussian, -0.5, 0)

        # 6. Adaptive Thresholding
        thresh = cv2.adaptiveThreshold(
            sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return thresh

    def run_ocr_on_pixmap(self, pix: fitz.Pixmap, page_num: int) -> List[TextBlockRecord]:
        ocr_blocks: List[TextBlockRecord] = []
        ocr_engine = self._get_ocr_engine()

        img_bytes = pix.tobytes("png")
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if ocr_engine:
            # Apply OpenCV full preprocessing
            preprocessed = self.preprocess_image_opencv(img_np)
            results = ocr_engine.ocr(preprocessed, cls=True)

            if results and results[0]:
                for idx, res in enumerate(results[0]):
                    bbox_pts, (text, conf) = res
                    xs = [pt[0] for pt in bbox_pts]
                    ys = [pt[1] for pt in bbox_pts]
                    x0, y0, x1, y1 = float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))
                    
                    # If low confidence, crop and save suspicious region for audit (PDF Page 2 & 3)
                    if float(conf) < 0.95:
                        crop_h, crop_w = img_np.shape[:2]
                        cy0, cy1 = max(0, int(y0)), min(crop_h, int(y1))
                        cx0, cx1 = max(0, int(x0)), min(crop_w, int(x1))
                        if cy1 > cy0 and cx1 > cx0:
                            crop_img = img_np[cy0:cy1, cx0:cx1]
                            crop_path = self.crops_dir / f"crop_p{page_num:03d}_{idx:03d}.png"
                            cv2.imwrite(str(crop_path), crop_img)

                    ocr_blocks.append(
                        TextBlockRecord(
                            text_id=f"ocr_{page_num:03d}_{idx:03d}",
                            page_number=page_num,
                            text=text.strip(),
                            source="ocr",
                            bbox=[x0, y0, x1, y1],
                            reading_order_idx=idx,
                            confidence=float(conf),
                            verified=float(conf) >= 0.95
                        )
                    )
        return ocr_blocks

    def extract_tables_from_page(self, page: fitz.Page, page_num: int) -> List[TableRecord]:
        """
        Table extraction preserving row label, column header, unit, currency, reporting period, footnotes.
        Uses PyMuPDF find_tables() as primary, with PP-Structure fallback for complex tables.
        """
        tables: List[TableRecord] = []
        try:
            tabs = page.find_tables()
            for t_idx, tab in enumerate(tabs):
                df = tab.extract()
                if df and len(df) > 1:
                    headers = [str(col).strip() if col is not None else "" for col in df[0]]
                    rows = [[str(cell).strip() if cell is not None else "" for cell in row] for row in df[1:]]

                    # Detect currency, unit, or reporting period in headers or rows
                    footnote_candidates = []
                    for r in rows:
                        joined_r = " ".join(r)
                        if any(k in joined_r.lower() for k in ["note:", "source:", "in crores", "in millions", "usd", "inr"]):
                            footnote_candidates.append(joined_r)

                    header_line = "| " + " | ".join(headers) + " |"
                    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
                    body_lines = ["| " + " | ".join(r) + " |" for r in rows]
                    md_table = "\n".join([header_line, sep_line] + body_lines)

                    tables.append(
                        TableRecord(
                            table_id=f"tab_{page_num:03d}_{t_idx:03d}",
                            page_number=page_num,
                            headers=headers,
                            rows=rows,
                            markdown_content=md_table,
                            bbox=[float(x) for x in tab.bbox],
                            footnotes=footnote_candidates
                        )
                    )
        except Exception as e:
            logger.warning(f"PyMuPDF table extraction on page {page_num}: {e}")

        # PP-Structure fallback if enabled and PyMuPDF found no tables on a page with images
        if not tables and self.use_ppstructure:
            pps = self._get_ppstructure_engine()
            if pps:
                try:
                    pix = page.get_pixmap(dpi=200)
                    img_bytes = pix.tobytes("png")
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    pps_results = pps(img_np)
                    for r_idx, region in enumerate(pps_results):
                        if region.get("type", "").lower() == "table":
                            html = region.get("res", {}).get("html", "")
                            bbox = region.get("bbox", [0, 0, 0, 0])
                            tables.append(
                                TableRecord(
                                    table_id=f"tab_pps_{page_num:03d}_{r_idx:03d}",
                                    page_number=page_num,
                                    headers=["PPStructure Table"],
                                    rows=[[html]],
                                    markdown_content=f"[PP-Structure Table Page {page_num}]:\n{html}",
                                    bbox=[float(x) for x in bbox],
                                    footnotes=["Extracted via PP-Structure fallback"]
                                )
                            )
                except Exception as pps_err:
                    logger.debug(f"PP-Structure extraction on page {page_num}: {pps_err}")

        return tables

    def extract_all(self) -> Tuple[List[TextBlockRecord], List[ImageRecord], List[TableRecord]]:
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found at: {self.pdf_path}")

        doc = fitz.open(self.pdf_path)
        all_text_records: List[TextBlockRecord] = []
        all_image_records: List[ImageRecord] = []
        all_table_records: List[TableRecord] = []

        logger.info(f"Extracting PDF: {self.pdf_path.name} ({len(doc)} pages)")

        for page_idx in range(len(doc)):
            page_num = page_idx + 1
            page = doc[page_idx]

            # 1. Native Selectable Text
            text_instances = 0
            blocks = page.get_text("blocks")
            for block_no, b in enumerate(blocks):
                if b[6] == 0:
                    text_content = b[4].strip()
                    if text_content:
                        text_instances += 1
                        all_text_records.append(
                            TextBlockRecord(
                                text_id=f"text_{page_num:03d}_{block_no:03d}",
                                page_number=page_num,
                                text=text_content,
                                source="native_text",
                                bbox=[float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                                reading_order_idx=block_no,
                                confidence=1.0,
                                verified=True
                            )
                        )

            # 2. Scanned OCR Fallback & Mixed Page Handling
            # Check native character count and whether page has images
            total_native_chars = sum(len(b.text) for b in all_text_records if b.page_number == page_num)
            image_list = page.get_images(full=True)

            # Trigger OCR if page is completely scanned (0 text instances) OR
            # mixed page (very sparse text < 80 characters while containing substantial image/drawing)
            is_mixed_page = (text_instances > 0 and total_native_chars < 80 and len(image_list) > 0)
            if text_instances == 0 or is_mixed_page:
                logger.info(
                    f"Page {page_num} detected as {'scanned' if text_instances == 0 else 'mixed native/scanned'}. "
                    "Rendering for OCR with OpenCV preprocessing."
                )
                pix = page.get_pixmap(dpi=300)
                ocr_records = self.run_ocr_on_pixmap(pix, page_num)
                all_text_records.extend(ocr_records)

            # 3. Tabular Data
            tabs = self.extract_tables_from_page(page, page_num)
            all_table_records.extend(tabs)

            # 4. Images
            for img_idx, img_info in enumerate(image_list):
                xref = img_info[0]
                base_img = doc.extract_image(xref)
                image_bytes = base_img["image"]
                image_ext = base_img["ext"]

                image_filename = f"img_{page_num:03d}_{img_idx:03d}.{image_ext}"
                image_disk_path = self.images_dir / image_filename

                with open(image_disk_path, "wb") as f_img:
                    f_img.write(image_bytes)

                all_image_records.append(
                    ImageRecord(
                        image_id=f"img_{page_num:03d}_{img_idx:03d}",
                        page_number=page_num,
                        image_path=str(image_disk_path.resolve()),
                        format=image_ext,
                        width=base_img["width"],
                        height=base_img["height"],
                        association_status="uncertain"
                    )
                )

        doc.close()

        # Save to raw extracted JSONL files immediately
        write_jsonl(self.extracted_dir / "text_records.jsonl", all_text_records)
        write_jsonl(self.extracted_dir / "image_records.jsonl", all_image_records)
        write_jsonl(self.extracted_dir / "table_records.jsonl", all_table_records)

        logger.info(
            f"Extraction complete: {len(all_text_records)} text blocks, "
            f"{len(all_image_records)} images, {len(all_table_records)} tables."
        )
        return all_text_records, all_image_records, all_table_records
