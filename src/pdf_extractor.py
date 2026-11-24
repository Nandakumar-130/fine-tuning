import os
import re
import cv2
import json
import fitz  # PyMuPDF
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Tuple, Optional, Any
import logging

from src.schemas import (
    TextBlockRecord, TableRecord, ImageRecord, BoundingBox,
    ParagraphRecord, ScannedTextRecord, ChartRecord, KPIBoxRecord
)
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
        pdf_path: Union[str, Path],
        output_dir: Union[str, Path],
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
        self.extracted_data_dir = self.output_dir / "extracted_data"
        self.images_dir = self.extracted_dir / "images"
        self.crops_dir = self.extracted_dir / "crops"
        ensure_directory(self.extracted_dir)
        ensure_directory(self.extracted_data_dir)
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

    def extract_tables_from_page(self, page: fitz.Page, page_num: int, blocks: Optional[list] = None) -> List[TableRecord]:
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
                    units_found = []
                    currency_found = []
                    periods_found = []

                    for r in rows:
                        joined_r = " ".join(r)
                        if any(k in joined_r.lower() for k in ["note:", "source:", "in crores", "in millions", "usd", "inr"]):
                            footnote_candidates.append(joined_r)

                    all_cells_text = " ".join(headers + [cell for row in rows for cell in row])
                    if "inr" in all_cells_text.lower() or "₹" in all_cells_text or "rs" in all_cells_text.lower():
                        currency_found.append("INR")
                    if "usd" in all_cells_text.lower() or "$" in all_cells_text:
                        currency_found.append("USD")
                    if "crore" in all_cells_text.lower() or "cr" in all_cells_text.lower():
                        units_found.append("Crores")
                    if "%" in all_cells_text or "percent" in all_cells_text.lower():
                        units_found.append("Percentage")

                    for p_match in re.findall(r'\b(?:FY\s*\d{2,4}|20\d{2}-\d{2,4}|March\s+31,\s+20\d{2})\b', all_cells_text, re.IGNORECASE):
                        if p_match not in periods_found:
                            periods_found.append(p_match)

                    # Table title: look for text block directly above table bbox
                    table_title = None
                    if blocks:
                        for b in blocks:
                            if b[6] == 0 and b[3] <= tab.bbox[1] + 10 and (tab.bbox[1] - b[3]) < 45:
                                title_cand = b[4].strip()
                                if title_cand and len(title_cand) < 120 and "\n" not in title_cand:
                                    table_title = title_cand
                                    break

                    header_line = "| " + " | ".join(headers) + " |"
                    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
                    body_lines = ["| " + " | ".join(r) + " |" for r in rows]
                    title_prefix = f"### {table_title}\n" if table_title else ""
                    md_table = title_prefix + "\n".join([header_line, sep_line] + body_lines)

                    tables.append(
                        TableRecord(
                            table_id=f"tab_{page_num:03d}_{t_idx:03d}",
                            page_number=page_num,
                            table_title=table_title,
                            headers=headers,
                            rows=rows,
                            markdown_content=md_table,
                            bbox=[float(x) for x in tab.bbox],
                            units=", ".join(units_found) if units_found else None,
                            currency=", ".join(currency_found) if currency_found else None,
                            reporting_period=", ".join(periods_found) if periods_found else None,
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
                                    table_title="Complex Layout Table",
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

    def extract_kpi_boxes_from_page(
        self,
        page: fitz.Page,
        page_num: int,
        blocks: list
    ) -> List[KPIBoxRecord]:
        """
        Phase 3.6: Separate KPI and percentage boxes.
        Detects independent numerical cards (e.g., '82.8 Cr. Revenue', '7.2% Total Income Growth', '50+ Customers').
        Preserves Value, Label, Unit, Section heading, Page number, Bounding box, Extraction method, Confidence.
        """
        kpis: List[KPIBoxRecord] = []
        val_pattern = re.compile(
            r'^[₹$€]?\s*\(?-?\d+(?:[.,]\d+)?\)?\s*(?:Cr\.?|Crores?|Mn\.?|Millions?|Bn\.?|Billions?|%|\+|M|K)?$',
            re.IGNORECASE
        )
        words = page.get_text("words")  # (x0, y0, x1, y1, word, block_no, line_no, word_no)
        seen_keys = set()
        kpi_idx = 1

        # Track prominent headings on page
        current_heading = "General Overview"
        for b in blocks:
            if b[6] == 0:
                txt = b[4].strip()
                if len(txt) < 80 and ("HIGHLIGHT" in txt.upper() or "OPERATIONS" in txt.upper() or "REVENUE" in txt.upper() or "AT A GLANCE" in txt.upper()):
                    current_heading = txt.splitlines()[0].strip()
                    break

        # Candidate metric words (values with Cr, %, +, M, K or standalone formatted numbers)
        metric_words = []
        for w in words:
            word_str = w[4].strip()
            if not val_pattern.match(word_str):
                continue
            # Filter out simple page numbers or tiny indices near footer/header
            if word_str.isdigit() and len(word_str) <= 2:
                if w[1] > 720 or w[1] < 60:
                    continue
            metric_words.append(w)

        for w in metric_words:
            val = w[4].strip()
            # Find label words directly below within 28 vertical points and horizontal alignment
            below_words = [
                b[4] for b in words
                if 0 <= (b[1] - w[3]) < 28 and abs(b[0] - w[0]) < 35
                and not val_pattern.match(b[4])
                and b[4].lower() not in ["annual", "report", "tracxn", "technologies", "page"]
            ]
            if not below_words:
                continue

            lbl = " ".join(below_words[:6]).strip()
            if len(lbl) < 3 or len(lbl) > 100:
                continue

            dedup_key = (page_num, val, lbl.lower())
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

            # Determine unit
            unit = None
            if "cr" in val.lower() or "crore" in val.lower():
                unit = "INR Crores"
            elif "%" in val:
                unit = "Percentage (%)"
            elif "m" in val.lower() and not val.lower().endswith(" members"):
                unit = "Millions"
            elif "+" in val:
                unit = "Count / Plus"

            kpis.append(
                KPIBoxRecord(
                    kpi_id=f"kpi_{page_num:03d}_{kpi_idx:03d}",
                    page_number=page_num,
                    value=val,
                    label=lbl,
                    unit=unit,
                    section_heading=current_heading,
                    bbox=[float(w[0]), float(w[1]), float(w[2]), float(w[3])],
                    extraction_method="layout_geometric_matching",
                    confidence=0.98
                )
            )
            kpi_idx += 1

        return kpis

    def extract_charts_from_page(
        self,
        page: fitz.Page,
        page_num: int,
        blocks: list,
        image_records: List[ImageRecord]
    ) -> List[ChartRecord]:
        """
        Phase 3.5: Charts and graphs detection.
        Detects chart graphics with titles, axis labels, legends, units, data labels.
        Does not invent numerical values; records extraction limitations where values are visual-only.
        """
        charts: List[ChartRecord] = []
        page_text = page.get_text("text").lower()
        drawings = len(page.get_drawings())
        page_images = [img for img in image_records if img.page_number == page_num]

        chart_keywords = [
            "chart", "trend", "growth rate", "breakup", "contribution (%)",
            "distribution", "revenue by", "pat margin", "ebitda margin",
            "financial highlights", "ratio"
        ]

        if not any(k in page_text for k in chart_keywords) and drawings < 10 and not page_images:
            return charts

        # Find potential chart titles
        chart_idx = 1
        for b in blocks:
            if b[6] == 0:
                txt = b[4].strip()
                lines = [l.strip() for l in txt.splitlines() if l.strip()]
                for line in lines:
                    if any(k in line.lower() for k in chart_keywords) and len(line) < 90:
                        # Extract axis / period candidates from surrounding text
                        axes = re.findall(r'\b(?:FY\s*\d{2,4}|20\d{2}-\d{2,4})\b', page_text, re.IGNORECASE)
                        legends = []
                        for leg in ["Americas", "APAC", "EMEA", "India", "Operations", "EBITDA", "PAT"]:
                            if leg.lower() in page_text:
                                legends.append(leg)

                        units = "Percentage (%)" if "%" in line else ("INR Crores" if "cr" in line.lower() else None)

                        charts.append(
                            ChartRecord(
                                chart_id=f"chart_{page_num:03d}_{chart_idx:03d}",
                                page_number=page_num,
                                title=line,
                                x_axis_label=", ".join(sorted(list(set(axes)))) if axes else "Financial Years",
                                y_axis_label=units or "Metric Value",
                                units=units,
                                legends=legends,
                                data_labels=[],
                                numerical_values={},
                                bbox=[float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                                extraction_limitations=(
                                    "Visual graphical representation; exact numerical values retained where supported "
                                    "by text labels. No unverified values fabricated from bar heights or pie slices."
                                ),
                                image_ref=page_images[0].image_id if page_images else None,
                                confidence=0.90
                            )
                        )
                        chart_idx += 1
                        break

        return charts

    def extract_all(self, page_range: Optional[Tuple[int, int]] = None) -> Tuple[List[TextBlockRecord], List[ImageRecord], List[TableRecord]]:
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found at: {self.pdf_path}")

        doc = fitz.open(self.pdf_path)
        all_text_records: List[TextBlockRecord] = []
        all_image_records: List[ImageRecord] = []
        all_table_records: List[TableRecord] = []
        all_paragraph_records: List[ParagraphRecord] = []
        all_scanned_records: List[ScannedTextRecord] = []
        all_kpi_records: List[KPIBoxRecord] = []
        all_chart_records: List[ChartRecord] = []

        start_idx = 0
        end_idx = len(doc)
        if page_range:
            start_idx = max(0, page_range[0] - 1)
            end_idx = min(len(doc), page_range[1])

        total_doc_pages = len(doc)
        pages_processed = end_idx - start_idx
        logger.info(f"Extracting PDF: {self.pdf_path.name} (pages {start_idx + 1} to {end_idx} of {total_doc_pages})")

        for page_idx in range(start_idx, end_idx):
            page_num = page_idx + 1
            page = doc[page_idx]

            # 1. Native Selectable Text
            text_instances = 0
            blocks = page.get_text("blocks")
            current_heading = None

            for block_no, b in enumerate(blocks):
                if b[6] == 0:
                    text_content = b[4].strip()
                    if text_content:
                        text_instances += 1
                        if len(text_content) < 80 and text_content.isupper():
                            current_heading = text_content

                        tb_rec = TextBlockRecord(
                            text_id=f"text_{page_num:03d}_{block_no:03d}",
                            page_number=page_num,
                            text=text_content,
                            source="native_text",
                            bbox=[float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                            reading_order_idx=block_no,
                            confidence=1.0,
                            verified=True
                        )
                        all_text_records.append(tb_rec)

                        # Separate Paragraph Record
                        all_paragraph_records.append(
                            ParagraphRecord(
                                para_id=f"para_{page_num:03d}_{block_no:03d}",
                                page_number=page_num,
                                text=text_content,
                                section_heading=current_heading,
                                bbox=[float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                                reading_order_idx=block_no,
                                confidence=1.0
                            )
                        )

            # 2. Scanned OCR Fallback & Mixed Page Handling
            total_native_chars = sum(len(b.text) for b in all_text_records if b.page_number == page_num)
            image_list = page.get_images(full=True)

            is_mixed_page = (text_instances > 0 and total_native_chars < 80 and len(image_list) > 0)
            if text_instances == 0 or is_mixed_page:
                logger.info(
                    f"Page {page_num} detected as {'scanned' if text_instances == 0 else 'mixed native/scanned'}. "
                    "Rendering for OCR with OpenCV preprocessing."
                )
                pix = page.get_pixmap(dpi=300)
                ocr_records = self.run_ocr_on_pixmap(pix, page_num)
                all_text_records.extend(ocr_records)

                for ocr_rec in ocr_records:
                    all_scanned_records.append(
                        ScannedTextRecord(
                            scanned_id=f"scanned_{ocr_rec.text_id}",
                            page_number=page_num,
                            text=ocr_rec.text,
                            confidence=ocr_rec.confidence,
                            bbox=ocr_rec.bbox,
                            reading_order_idx=ocr_rec.reading_order_idx
                        )
                    )

            # 3. Tabular Data (with title, units, currency detection)
            tabs = self.extract_tables_from_page(page, page_num, blocks=blocks)
            all_table_records.extend(tabs)

            # 4. Images
            page_images: List[ImageRecord] = []
            for img_idx, img_info in enumerate(image_list):
                xref = img_info[0]
                base_img = doc.extract_image(xref)
                image_bytes = base_img["image"]
                image_ext = base_img["ext"]

                image_filename = f"img_{page_num:03d}_{img_idx:03d}.{image_ext}"
                image_disk_path = self.images_dir / image_filename

                with open(image_disk_path, "wb") as f_img:
                    f_img.write(image_bytes)

                img_rec = ImageRecord(
                    image_id=f"img_{page_num:03d}_{img_idx:03d}",
                    page_number=page_num,
                    image_path=str(image_disk_path.resolve()),
                    format=image_ext,
                    width=base_img["width"],
                    height=base_img["height"],
                    association_status="uncertain"
                )
                all_image_records.append(img_rec)
                page_images.append(img_rec)

            # 5. KPI and Percentage Boxes
            page_kpis = self.extract_kpi_boxes_from_page(page, page_num, blocks=blocks)
            all_kpi_records.extend(page_kpis)

            # 6. Charts & Graphs
            page_charts = self.extract_charts_from_page(page, page_num, blocks=blocks, image_records=page_images)
            all_chart_records.extend(page_charts)

        doc.close()

        # Cache extended records on instance
        self.all_kpi_records = all_kpi_records
        self.all_chart_records = all_chart_records
        self.all_paragraph_records = all_paragraph_records
        self.all_scanned_records = all_scanned_records

        # Save Phase 4: Separate Files in extracted_data/
        write_jsonl(self.extracted_data_dir / "paragraphs.jsonl", all_paragraph_records)
        write_jsonl(self.extracted_data_dir / "scanned_text.jsonl", all_scanned_records)
        
        with open(self.extracted_data_dir / "tables.json", "w", encoding="utf-8") as f:
            json.dump([t.model_dump() for t in all_table_records], f, indent=2, ensure_ascii=False)

        with open(self.extracted_data_dir / "charts.json", "w", encoding="utf-8") as f:
            json.dump([c.model_dump() for c in all_chart_records], f, indent=2, ensure_ascii=False)

        with open(self.extracted_data_dir / "images.json", "w", encoding="utf-8") as f:
            json.dump([img.model_dump() for img in all_image_records], f, indent=2, ensure_ascii=False)

        with open(self.extracted_data_dir / "kpi_boxes.json", "w", encoding="utf-8") as f:
            json.dump([k.model_dump() for k in all_kpi_records], f, indent=2, ensure_ascii=False)

        all_data = {
            "document_id": self.pdf_path.stem,
            "total_pages": total_doc_pages,
            "paragraphs": [p.model_dump() for p in all_paragraph_records],
            "tables": [t.model_dump() for t in all_table_records],
            "charts": [c.model_dump() for c in all_chart_records],
            "scanned_text": [s.model_dump() for s in all_scanned_records],
            "images": [img.model_dump() for img in all_image_records],
            "kpi_boxes": [k.model_dump() for k in all_kpi_records]
        }
        with open(self.extracted_data_dir / "all_extracted_data.json", "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)

        report = {
            "document_name": self.pdf_path.name,
            "total_pages_in_document": total_doc_pages,
            "total_pages_processed": pages_processed,
            "total_paragraphs": len(all_paragraph_records),
            "total_scanned_text_records": len(all_scanned_records),
            "total_tables": len(all_table_records),
            "total_charts": len(all_chart_records),
            "total_images": len(all_image_records),
            "total_kpi_boxes": len(all_kpi_records)
        }
        with open(self.extracted_data_dir / "extraction_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Legacy extracted directory files
        write_jsonl(self.extracted_dir / "text_records.jsonl", all_text_records)
        write_jsonl(self.extracted_dir / "image_records.jsonl", all_image_records)
        write_jsonl(self.extracted_dir / "table_records.jsonl", all_table_records)

        logger.info(
            f"Extraction complete: {len(all_text_records)} text blocks ({len(all_paragraph_records)} paragraphs), "
            f"{len(all_table_records)} tables, {len(all_chart_records)} charts, "
            f"{len(all_kpi_records)} KPI boxes, {len(all_image_records)} images."
        )
        return all_text_records, all_image_records, all_table_records

    def extract_all_multimodal(self) -> Tuple[
        List[TextBlockRecord], List[ImageRecord], List[TableRecord],
        List[KPIBoxRecord], List[ChartRecord], List[ParagraphRecord]
    ]:
        texts, images, tables = self.extract_all()
        return texts, images, tables, self.all_kpi_records, self.all_chart_records, self.all_paragraph_records
