import re
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import logging

from src.schemas import TextBlockRecord, VerifiedRecord
from src.io_utils import ensure_directory, write_jsonl

logger = logging.getLogger("QualityVerifier")

class ExtractionQualityVerifier:
    """
    Step 2: Extraction Quality Verification
    Implements all checks from PDF Page 2 & 3:
      - OCR confidence (0.95-1.0 provisional accept, 0.80-0.94 inspect, <0.80 recheck/review)
      - Spelling: SymSpell dictionary lookup
      - Grammar: LanguageTool warning audit
      - Numbers / Decimals: Python regex + Decimal validation (missing separators, 'O' replacing '0')
      - Dates: python-dateutil + regex format validation
      - Coordinates: Coordinates retained for inspection of suspicious original crops
    """
    def __init__(
        self,
        output_dir: str | Path,
        ocr_accept: float = 0.95,
        ocr_review: float = 0.80,
        enable_languagetool: bool = False
    ):
        self.output_dir = Path(output_dir)
        self.verified_dir = self.output_dir / "verified"
        ensure_directory(self.verified_dir)
        self.ocr_accept = ocr_accept
        self.ocr_review = ocr_review
        self.enable_languagetool = enable_languagetool

        self._symspell = None
        self._init_symspell()

        self._languagetool = None
        if self.enable_languagetool:
            self._init_languagetool()

    def _init_symspell(self):
        try:
            import pkg_resources
            from symspellpy import SymSpell
            self._symspell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
            dict_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
            self._symspell.load_dictionary(dict_path, term_index=0, count_index=1)
            logger.info("SymSpell initialized for spelling audits.")
        except Exception as e:
            logger.warning(f"SymSpell initialization fallback: {e}")
            self._symspell = None

    def _init_languagetool(self):
        try:
            import language_tool_python
            self._languagetool = language_tool_python.LanguageTool('en-US')
            logger.info("LanguageTool initialized for grammar audits.")
        except Exception as e:
            logger.warning(f"LanguageTool fallback: {e}")
            self._languagetool = None

    def verify_numbers_and_decimals_exact(self, text: str) -> Tuple[List[str], Optional[str]]:
        """
        Exact decimal and financial validation using Python Decimal & Regex (Page 2 & 3 example).
        Example in PDF:
          Original PDF: Revenue reached 25,000.50 in 2025.
          OCR output: Revenue reached 2500050 in 2025.
          Warnings: Currency symbol missing, Decimal separator missing, Letter 'O' replaced zero.
        """
        warnings = []
        correction_reason = None

        # Check 1: Letter 'O' or 'o' replacing zero in numeric patterns (e.g. 2O25 or 10O0)
        o_zero_matches = re.findall(r'\b\d+[oO]\d*\b|\b[oO]\d+\b', text)
        if o_zero_matches:
            warnings.append(f"Letter 'O' may have replaced number zero: {o_zero_matches}")

        # Check 2: Potential missing decimal separator in large numbers (e.g. 2500050)
        large_ints = re.findall(r'\b\d{6,}\b', text)
        for num_str in large_ints:
            # Check context: if preceded by financial terms like revenue, invested, crore, $
            if re.search(rf'(revenue|profit|cost|invested|amount|price|\$|₹|€)\s*(?:reached|was|of|is)?\s*{num_str}', text, re.IGNORECASE):
                warnings.append(f"Financial number {num_str} lacks decimal separator or comma formatting; inspect source crop.")

        # Check 3: Broken currency symbols or isolated symbols
        if re.search(r'(\$|₹|€|£)\s{2,}\d+', text):
            warnings.append("Disjoint currency symbol with excessive whitespace.")

        # Check 4: Strict Decimal parsing of identified numbers
        numeric_tokens = re.findall(r'\b\d+(?:\.\d+)?\b', text)
        for tok in numeric_tokens:
            try:
                _ = Decimal(tok)
            except InvalidOperation:
                warnings.append(f"Invalid decimal token detected: '{tok}'")

        return warnings, correction_reason

    def verify_dates(self, text: str) -> List[str]:
        warnings = []
        # Check malformed year (e.g., 202O)
        if re.search(r'\b(19|20)\d[oO]\b', text):
            warnings.append("Malformed year representation detected (e.g. 'O' in place of '0').")

        # Parse date strings with dateutil
        try:
            from dateutil import parser
            date_matches = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b[A-Za-z]+ \d{1,2},? \d{4}\b', text)
            for d_str in date_matches:
                try:
                    _ = parser.parse(d_str, fuzzy=False)
                except Exception:
                    warnings.append(f"Suspicious date string detected: '{d_str}'")
        except ImportError:
            pass
        return warnings

    def verify_spelling(self, text: str) -> List[str]:
        warnings = []
        if not self._symspell:
            return warnings
        from symspellpy import Verbosity
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text)
        for w in words:
            if w[0].isupper():
                continue  # Never blindly change or flag proper names (Page 2)
            suggestions = self._symspell.lookup(w.lower(), Verbosity.CLOSEST, max_edit_distance=2)
            if suggestions and suggestions[0].term != w.lower() and suggestions[0].distance > 0:
                if suggestions[0].count > 5000:
                    warnings.append(f"Spelling check: '{w}' may be '{suggestions[0].term}'")
                    if len(warnings) >= 3:
                        break
        return warnings

    def verify_grammar(self, text: str) -> List[str]:
        warnings = []
        if not self._languagetool:
            return warnings
        try:
            matches = self._languagetool.check(text[:500])
            for m in matches[:2]:
                warnings.append(f"Grammar notice (warning only): {m.message}")
        except Exception:
            pass
        return warnings

    def verify_text_block(self, block: TextBlockRecord) -> VerifiedRecord:
        raw_text = block.text
        corrected_text = unicodedata.normalize("NFKC", raw_text)
        warnings = []
        status = "PASS"
        correction_reasons = []

        # 1. OCR Confidence check
        if block.source == "ocr":
            if block.confidence < self.ocr_review:
                warnings.append(f"Low OCR confidence ({block.confidence:.2f} < {self.ocr_review}); requires inspection of original crop.")
                status = "MANUAL_REVIEW"
            elif block.confidence < self.ocr_accept:
                warnings.append(f"Moderate OCR confidence ({block.confidence:.2f} < {self.ocr_accept})")
                status = "WARNING"

        # 2. Number & Decimal audit with Python Decimal
        num_warnings, num_corr = self.verify_numbers_and_decimals_exact(corrected_text)
        if num_warnings:
            warnings.extend(num_warnings)
            if status != "MANUAL_REVIEW":
                status = "WARNING"

        # 3. Date audit with dateutil
        date_warnings = self.verify_dates(corrected_text)
        if date_warnings:
            warnings.extend(date_warnings)
            if status != "MANUAL_REVIEW":
                status = "WARNING"

        # 4. Spelling audit with SymSpell
        spell_warnings = self.verify_spelling(corrected_text)
        if spell_warnings:
            warnings.extend(spell_warnings)
            if status != "MANUAL_REVIEW":
                status = "WARNING"

        # 5. Grammar audit with LanguageTool
        if self.enable_languagetool:
            gram_warnings = self.verify_grammar(corrected_text)
            if gram_warnings:
                warnings.extend(gram_warnings)

        # 6. Conservative whitespace normalization
        cleaned = re.sub(r'[ \t]+', ' ', corrected_text).strip()
        if cleaned != raw_text:
            correction_reasons.append("Whitespace normalized & NFKC standardized.")
            corrected_text = cleaned

        reason_str = "; ".join(correction_reasons) if correction_reasons else None

        return VerifiedRecord(
            record_id=f"ver_{block.text_id}",
            page_number=block.page_number,
            raw_text=raw_text,
            corrected_text=corrected_text,
            verification_status=status,
            warnings=warnings,
            correction_reason=reason_str,
            bbox=block.bbox
        )

    def verify_all(self, text_records: List[TextBlockRecord]) -> List[VerifiedRecord]:
        logger.info(f"Verifying {len(text_records)} text blocks with exact numbers/decimals, SymSpell & dateutil...")
        verified_records: List[VerifiedRecord] = []
        for block in text_records:
            rec = self.verify_text_block(block)
            verified_records.append(rec)

        out_file = self.verified_dir / "verified_records.jsonl"
        write_jsonl(out_file, verified_records)
        logger.info(f"Verification complete. Saved to: {out_file}")
        return verified_records
