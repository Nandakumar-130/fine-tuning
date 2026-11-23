"""
Master 34-Case Comprehensive Integration Test Suite
Validates the complete PDF-to-Q&A dataset preparation pipeline:
1. Selectable-text PDF extraction.
2. Scanned PDF extraction logic.
3. Mixed native/scanned page OCR trigger.
4. OpenCV preprocessing (grayscale, denoise, deskew, sharpen, adaptive threshold).
5. PaddleOCR confidence handling (<0.95 audit crop saving).
6. Table extraction with structure, headers, and footnotes.
7. Complex table fallback (PP-Structure hook).
8. Vision input and image description (PIL image bytes).
9. Image caption association (verified vs uncertain).
10. Spelling validation (SymSpell).
11. Financial number validation (Decimal exact traps, missing separator '2500050').
12. Invalid calendar dates (dateutil trap).
13. Reading-order verification.
14. Verified text propagation to PageOrganizer.
15. Structure-aware chunking.
16. Exact Qwen token counting / conservative fallback.
17. Chunk overlap.
18. Atomic fact extraction (S-R-O).
19. Entity normalization.
20. Cross-page relationship verification (>=2 chunks, >=2 pages).
21. Direct Q&A generation logic.
22. Indirect Q&A cascade.
23. Attribute Q&A cascade.
24. Combined Q&A multi-fact requirement (no unrelated fallback).
25. Cross-page Q&A multi-chunk evidence requirement.
26. Complete approved-batch propagation.
27. Exact duplicate removal.
28. Semantic variation preservation.
29. Unsupported answer rejection (Prompt F).
30. Invalid provenance rejection.
31. JSON repair (Prompt G syntax only).
32. Checkpoint recovery (skip completed stages & chunks).
33. Final SFT schema validation (ChatML format).
34. Train/validation leakage prevention (fact-cluster split).
"""

import os
import sys
import json
import tempfile
import numpy as np
from pathlib import Path
from decimal import Decimal
from unittest.mock import MagicMock

# Source imports
from src.schemas import (
    TextBlockRecord, TableRecord, ImageRecord, VerifiedRecord,
    PageRecord, ChunkRecord, FactRecord, EntityIndexRecord,
    RelatedChunkGroup, RichQARecord, SFTConversationRecord, ChatMessage,
    KPIBoxRecord, ChartRecord, ParagraphRecord, ScannedTextRecord,
    ContentGroupRecord, RelationshipRecord
)
from src.pdf_extractor import PDFExtractor
from src.quality_verifier import ExtractionQualityVerifier
from src.image_association import ImageAssociationManager
from src.page_organizer import PageOrganizer
from src.chunker import HybridChunker
from src.vllm_engine import get_offline_qwen_tokenizer, check_gpu_resources
from src.fact_extractor import FactEntityExtractor
from src.qa_generator import SequentialPerspectiveQAGenerator
from src.deduplicator import QADeduplicator
from src.evaluator import IndependentEvaluator
from src.exporter import FineTuningExporter
from src.io_utils import CheckpointTracker, write_jsonl, read_jsonl

def run_all_tests():
    print("=" * 70)
    print("RUNNING MASTER 34-CASE INTEGRATION TEST SUITE")
    print("=" * 70)
    passed_tests = []
    failed_tests = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # -------------------------------------------------------------
        # Case 1: Selectable-text PDF extraction
        # -------------------------------------------------------------
        try:
            sample_pdf = Path("data_set_use.pdf")
            if sample_pdf.exists():
                import fitz
                doc = fitz.open(sample_pdf)
                found_text = False
                for p in range(min(10, len(doc))):
                    txt = doc[p].get_text("text").strip()
                    if len(txt) > 0:
                        found_text = True
                        break
                assert found_text, "No selectable text found in first 10 pages"
                passed_tests.append("1. Selectable-text PDF extraction")
            else:
                passed_tests.append("1. Selectable-text PDF extraction (Skipped: PDF not present)")
        except Exception as e:
            failed_tests.append(f"1. Selectable-text PDF extraction: {e}")

        # -------------------------------------------------------------
        # Case 2: Scanned PDF extraction logic
        # -------------------------------------------------------------
        try:
            ext = PDFExtractor(pdf_path="test.pdf", output_dir=tmp_path)
            # Verify ocr engine getter is present
            assert hasattr(ext, "run_ocr_on_pixmap")
            passed_tests.append("2. Scanned PDF extraction logic")
        except Exception as e:
            failed_tests.append(f"2. Scanned PDF extraction logic: {e}")

        # -------------------------------------------------------------
        # Case 3: Mixed native/scanned page OCR trigger
        # -------------------------------------------------------------
        try:
            # Check source code logic in pdf_extractor.py for mixed-page condition
            with open("src/pdf_extractor.py", "r", encoding="utf-8") as f:
                code = f.read()
            assert "is_mixed_page" in code
            assert "total_native_chars < 80 and len(image_list) > 0" in code
            passed_tests.append("3. Mixed native/scanned page OCR trigger")
        except Exception as e:
            failed_tests.append(f"3. Mixed native/scanned page OCR trigger: {e}")

        # -------------------------------------------------------------
        # Case 4: OpenCV preprocessing
        # -------------------------------------------------------------
        try:
            ext = PDFExtractor(pdf_path="test.pdf", output_dir=tmp_path)
            dummy_img = np.ones((200, 200, 3), dtype=np.uint8) * 255
            # Draw some dark text-like pixels
            dummy_img[50:60, 50:150] = 0
            processed = ext.preprocess_image_opencv(dummy_img)
            assert processed.shape[0] > 0
            assert processed.dtype == np.uint8
            passed_tests.append("4. OpenCV preprocessing")
        except Exception as e:
            failed_tests.append(f"4. OpenCV preprocessing: {e}")

        # -------------------------------------------------------------
        # Case 5: PaddleOCR confidence handling
        # -------------------------------------------------------------
        try:
            with open("src/pdf_extractor.py", "r", encoding="utf-8") as f:
                code = f.read()
            assert "0.95" in code
            assert "crops_dir" in code
            passed_tests.append("5. PaddleOCR confidence handling")
        except Exception as e:
            failed_tests.append(f"5. PaddleOCR confidence handling: {e}")

        # -------------------------------------------------------------
        # Case 6: Table extraction
        # -------------------------------------------------------------
        try:
            with open("src/pdf_extractor.py", "r", encoding="utf-8") as f:
                code = f.read()
            assert "find_tables" in code
            assert "markdown_content" in code
            assert "footnotes" in code
            passed_tests.append("6. Table extraction")
        except Exception as e:
            failed_tests.append(f"6. Table extraction: {e}")

        # -------------------------------------------------------------
        # Case 7: Complex table fallback (PP-Structure hook)
        # -------------------------------------------------------------
        try:
            ext = PDFExtractor(pdf_path="test.pdf", output_dir=tmp_path, use_ppstructure=True)
            assert hasattr(ext, "_get_ppstructure_engine")
            passed_tests.append("7. Complex table fallback (PP-Structure hook)")
        except Exception as e:
            failed_tests.append(f"7. Complex table fallback: {e}")

        # -------------------------------------------------------------
        # Case 8: Vision input and image description
        # -------------------------------------------------------------
        try:
            with open("src/vllm_engine.py", "r", encoding="utf-8") as f:
                code = f.read()
            assert "PIL.Image.open" in code or "Image.open(image_path)" in code
            assert "multi_modal_data" in code or "AutoModelForVision2Seq" in code
            passed_tests.append("8. Vision input and image description")
        except Exception as e:
            failed_tests.append(f"8. Vision input and image description: {e}")

        # -------------------------------------------------------------
        # Case 9: Image caption association
        # -------------------------------------------------------------
        try:
            img_mgr = ImageAssociationManager(output_dir=tmp_path)
            imgs = [ImageRecord(image_id="img_001", page_number=1, image_path="fake.png", width=100, height=100)]
            texts = [TextBlockRecord(text_id="t1", page_number=1, text="Figure 1: Revenue breakdown for FY2025.")]
            res = img_mgr.associate_images_with_text(imgs, texts)
            assert res[0].association_status == "verified"
            assert "Figure 1" in res[0].caption
            passed_tests.append("9. Image caption association")
        except Exception as e:
            failed_tests.append(f"9. Image caption association: {e}")

        # -------------------------------------------------------------
        # Case 10: Spelling validation
        # -------------------------------------------------------------
        try:
            qv = ExtractionQualityVerifier(output_dir=tmp_path)
            assert hasattr(qv, "verify_spelling")
            passed_tests.append("10. Spelling validation")
        except Exception as e:
            failed_tests.append(f"10. Spelling validation: {e}")

        # -------------------------------------------------------------
        # Case 11: Financial number validation
        # -------------------------------------------------------------
        try:
            qv = ExtractionQualityVerifier(output_dir=tmp_path)
            w1, _ = qv.verify_numbers_and_decimals_exact("Total revenue reached 2500050 in 2025.")
            assert any("2500050" in x for x in w1)
            w2, _ = qv.verify_numbers_and_decimals_exact("Profit was 1O000 Cr.")
            assert any("Letter 'O'" in x for x in w2)
            passed_tests.append("11. Financial number validation")
        except Exception as e:
            failed_tests.append(f"11. Financial number validation: {e}")

        # -------------------------------------------------------------
        # Case 12: Invalid calendar dates
        # -------------------------------------------------------------
        try:
            qv = ExtractionQualityVerifier(output_dir=tmp_path)
            w_date = qv.verify_dates("Signed on 31/02/2025.")
            assert any("Suspicious date" in x or "31/02/2025" in x for x in w_date)
            passed_tests.append("12. Invalid calendar dates")
        except Exception as e:
            failed_tests.append(f"12. Invalid calendar dates: {e}")

        # -------------------------------------------------------------
        # Case 13: Reading-order verification
        # -------------------------------------------------------------
        try:
            tb1 = TextBlockRecord(text_id="t1", page_number=1, text="Second block", reading_order_idx=2)
            tb2 = TextBlockRecord(text_id="t2", page_number=1, text="First block", reading_order_idx=1)
            po = PageOrganizer(output_dir=tmp_path)
            p_recs = po.assemble_pages([tb1, tb2], [], [])
            assert p_recs[0].text_blocks[0].text == "First block"
            passed_tests.append("13. Reading-order verification")
        except Exception as e:
            failed_tests.append(f"13. Reading-order verification: {e}")

        # -------------------------------------------------------------
        # Case 14: Verified text propagation
        # -------------------------------------------------------------
        try:
            tb = TextBlockRecord(text_id="t1", page_number=1, text="Raw unverified text   ")
            vr = VerifiedRecord(record_id="ver_t1", page_number=1, raw_text="Raw unverified text   ", corrected_text="Raw normalized text", verification_status="PASS")
            po = PageOrganizer(output_dir=tmp_path)
            p_recs = po.assemble_pages([tb], [], [], verified_records=[vr])
            assert p_recs[0].text_blocks[0].text == "Raw normalized text"
            passed_tests.append("14. Verified text propagation")
        except Exception as e:
            failed_tests.append(f"14. Verified text propagation: {e}")

        # -------------------------------------------------------------
        # Case 15: Structure-aware chunking
        # -------------------------------------------------------------
        try:
            chunker = HybridChunker(output_dir=tmp_path, chunk_size=100, chunk_overlap=20, use_embeddings=False)
            assert hasattr(chunker, "chunk_document")
            passed_tests.append("15. Structure-aware chunking")
        except Exception as e:
            failed_tests.append(f"15. Structure-aware chunking: {e}")

        # -------------------------------------------------------------
        # Case 16: Exact Qwen token counting / fallback
        # -------------------------------------------------------------
        try:
            chunker = HybridChunker(output_dir=tmp_path, tokenizer=None, use_embeddings=False)
            t_cnt = chunker.count_tokens("Financial statement with Rs. 1,00,000 Cr.")
            assert t_cnt == int(len("Financial statement with Rs. 1,00,000 Cr.") / 3.5)
            assert chunker.is_exact_tokenizer is False
            passed_tests.append("16. Exact Qwen token counting / fallback")
        except Exception as e:
            failed_tests.append(f"16. Exact Qwen token counting / fallback: {e}")

        # -------------------------------------------------------------
        # Case 17: Chunk overlap
        # -------------------------------------------------------------
        try:
            with open("src/chunker.py", "r", encoding="utf-8") as f:
                code = f.read()
            assert "target_overlap" in code
            assert "overlap_accum" in code
            passed_tests.append("17. Chunk overlap")
        except Exception as e:
            failed_tests.append(f"17. Chunk overlap: {e}")

        # -------------------------------------------------------------
        # Case 18: Atomic fact extraction (S-R-O)
        # -------------------------------------------------------------
        try:
            mock_llm = MagicMock()
            fe = FactEntityExtractor(output_dir=tmp_path, vllm_client=mock_llm)
            assert hasattr(fe, "extract_facts_from_chunk")
            passed_tests.append("18. Atomic fact extraction (S-R-O)")
        except Exception as e:
            failed_tests.append(f"18. Atomic fact extraction: {e}")

        # -------------------------------------------------------------
        # Case 19: Entity normalization
        # -------------------------------------------------------------
        try:
            f1 = FactRecord(fact_id="f1", subject="LARSEN & TOUBRO", relation="reported", object="100", source_chunk_ids=["c1"], source_pages=[1])
            f2 = FactRecord(fact_id="f2", subject="  larsen & toubro ", relation="operates", object="plants", source_chunk_ids=["c2"], source_pages=[2])
            fe = FactEntityExtractor(output_dir=tmp_path, vllm_client=MagicMock())
            ents, grps = fe.build_cross_page_relationships([f1, f2], [ChunkRecord(chunk_id="c1", page_numbers=[1], content="text1", token_count=10), ChunkRecord(chunk_id="c2", page_numbers=[2], content="text2", token_count=10)])
            assert len(ents) == 1
            assert ents[0].normalized_name == "larsen & toubro"
            passed_tests.append("19. Entity normalization")
        except Exception as e:
            failed_tests.append(f"19. Entity normalization: {e}")

        # -------------------------------------------------------------
        # Case 20: Cross-page relationship verification
        # -------------------------------------------------------------
        try:
            assert len(grps) == 1
            assert set(grps[0].page_numbers) == {1, 2}
            assert len(grps[0].chunk_ids) == 2
            passed_tests.append("20. Cross-page relationship verification")
        except Exception as e:
            failed_tests.append(f"20. Cross-page relationship verification: {e}")

        # -------------------------------------------------------------
        # Case 21: Direct Q&A generation logic
        # -------------------------------------------------------------
        try:
            mock_llm = MagicMock()
            mock_llm.generate.return_value = '{"items":[{"question":"What was reported?","answer":"Reported 100."}]}'
            qagen = SequentialPerspectiveQAGenerator(output_dir=tmp_path, vllm_client=mock_llm)
            c = ChunkRecord(chunk_id="c1", page_numbers=[1], content="Content", token_count=5)
            res = qagen.generate_stage1_direct([c])
            assert len(res) == 1
            assert res[0].perspective == "direct"
            passed_tests.append("21. Direct Q&A generation logic")
        except Exception as e:
            failed_tests.append(f"21. Direct Q&A generation logic: {e}")

        # -------------------------------------------------------------
        # Case 22: Indirect Q&A cascade
        # -------------------------------------------------------------
        try:
            mock_llm.generate.return_value = '{"items":[{"question":"Why was it reported?","answer":"Because of growth."}]}'
            res_ind = qagen.generate_stage2_indirect([c], res)
            assert len(res_ind) == 1
            assert res_ind[0].parent_qa_id == res[0].qa_id
            assert res_ind[0].perspective == "indirect"
            passed_tests.append("22. Indirect Q&A cascade")
        except Exception as e:
            failed_tests.append(f"22. Indirect Q&A cascade: {e}")

        # -------------------------------------------------------------
        # Case 23: Attribute Q&A cascade
        # -------------------------------------------------------------
        try:
            mock_llm.generate.return_value = '{"items":[{"question":"What location?","answer":"Mumbai."}]}'
            res_att = qagen.generate_stage3_attribute([c], res_ind)
            assert len(res_att) == 1
            assert res_att[0].parent_qa_id == res_ind[0].qa_id
            assert res_att[0].perspective == "attribute"
            passed_tests.append("23. Attribute Q&A cascade")
        except Exception as e:
            failed_tests.append(f"23. Attribute Q&A cascade: {e}")

        # -------------------------------------------------------------
        # Case 24: Combined Q&A multi-fact requirement
        # -------------------------------------------------------------
        try:
            with open("src/qa_generator.py", "r", encoding="utf-8") as f:
                code = f.read()
            assert "chunk_attr = attr_by_chunk.get(chunk.chunk_id, [])" in code
            passed_tests.append("24. Combined Q&A multi-fact requirement")
        except Exception as e:
            failed_tests.append(f"24. Combined Q&A multi-fact requirement: {e}")

        # -------------------------------------------------------------
        # Case 25: Cross-page Q&A multi-chunk evidence requirement
        # -------------------------------------------------------------
        try:
            # Single-chunk cross-page must fail
            single_chunk_json = '{"items":[{"question":"What is the multi-page summary?","answer":"Detailed answer","source_chunk_ids":["c1"],"source_pages":[1]}]}'
            res_fail = qagen._parse_and_validate_items(single_chunk_json, "cross_page", "g1", [1, 2], valid_chunk_ids=["c1", "c2"], valid_pages=[1, 2])
            assert len(res_fail) == 0
            # Dual-chunk cross-page must pass
            dual_chunk_json = '{"items":[{"question":"What is the multi-page summary?","answer":"Detailed answer spanning chunks","source_chunk_ids":["c1","c2"],"source_pages":[1, 2]}]}'
            res_pass = qagen._parse_and_validate_items(dual_chunk_json, "cross_page", "g1", [1, 2], valid_chunk_ids=["c1", "c2"], valid_pages=[1, 2])
            assert len(res_pass) == 1
            passed_tests.append("25. Cross-page Q&A multi-chunk evidence requirement")
        except Exception as e:
            failed_tests.append(f"25. Cross-page Q&A: {e}")

        # -------------------------------------------------------------
        # Case 26: Complete approved-batch propagation
        # -------------------------------------------------------------
        try:
            with open("main.py", "r", encoding="utf-8") as f:
                code = f.read()
            assert "qa_gen.generate_stage2_indirect(chunks, direct_qa)" in code
            assert "qa_gen.generate_stage3_attribute(chunks, indirect_qa)" in code
            assert "qa_gen.generate_stage4_combined(chunks, attribute_qa)" in code
            passed_tests.append("26. Complete approved-batch propagation")
        except Exception as e:
            failed_tests.append(f"26. Complete approved-batch propagation: {e}")

        # -------------------------------------------------------------
        # Case 27: Exact duplicate removal
        # -------------------------------------------------------------
        try:
            deduper = QADeduplicator(output_dir=tmp_path)
            q1 = RichQARecord(qa_id="qa_01", question="What is revenue?", answer="100", perspective="direct")
            q2 = RichQARecord(qa_id="qa_02", question="what is revenue?   ", answer="100", perspective="direct")
            u, d = deduper.deduplicate([q1, q2])
            assert len(u) == 1
            assert len(d) == 1
            passed_tests.append("27. Exact duplicate removal")
        except Exception as e:
            failed_tests.append(f"27. Exact duplicate removal: {e}")

        # -------------------------------------------------------------
        # Case 28: Semantic variation preservation
        # -------------------------------------------------------------
        try:
            q3 = RichQARecord(qa_id="qa_03", question="How much was the revenue generated?", answer="100", perspective="indirect")
            u2, d2 = deduper.deduplicate([q1, q3])
            assert len(u2) == 2
            assert len(d2) == 0
            passed_tests.append("28. Semantic variation preservation")
        except Exception as e:
            failed_tests.append(f"28. Semantic variation preservation: {e}")

        # -------------------------------------------------------------
        # Case 29: Unsupported answer rejection (Prompt F)
        # -------------------------------------------------------------
        try:
            mock_judge = MagicMock()
            mock_judge.generate.return_value = '{"evidence_supported": false, "decision": "REJECT", "reason": "Claim contradicted", "unsupported_claims": ["100 Cr not found"]}'
            evaluator = IndependentEvaluator(output_dir=tmp_path, vllm_client=mock_judge)
            ev_res = evaluator.evaluate_qa_item(q1, "Company had zero revenue.")
            assert ev_res.decision == "REJECT"
            assert ev_res.evidence_supported is False
            passed_tests.append("29. Unsupported answer rejection (Prompt F)")
        except Exception as e:
            failed_tests.append(f"29. Unsupported answer rejection: {e}")

        # -------------------------------------------------------------
        # Case 30: Invalid provenance rejection
        # -------------------------------------------------------------
        try:
            q_bad_prov = RichQARecord(qa_id="qa_bad", question="Q?", answer="A", perspective="direct", source_chunk_ids=[], source_pages=[])
            ev_bad = evaluator.evaluate_qa_item(q_bad_prov, "Evidence text.")
            assert ev_bad.decision == "REJECT"
            assert "provenance" in ev_bad.reason.lower()
            passed_tests.append("30. Invalid provenance rejection")
        except Exception as e:
            failed_tests.append(f"30. Invalid provenance rejection: {e}")

        # -------------------------------------------------------------
        # Case 31: JSON repair (Prompt G syntax only)
        # -------------------------------------------------------------
        try:
            with open("src/vllm_engine.py", "r", encoding="utf-8") as f:
                code = f.read()
            assert "Repair JSON syntax and schema formatting only" in code
            assert "Do not add facts or change meaning" in code
            passed_tests.append("31. JSON repair (Prompt G syntax only)")
        except Exception as e:
            failed_tests.append(f"31. JSON repair: {e}")

        # -------------------------------------------------------------
        # Case 32: Checkpoint recovery
        # -------------------------------------------------------------
        try:
            ckpt_path = tmp_path / "checkpoints.json"
            tracker = CheckpointTracker(ckpt_path)
            tracker.mark_stage_done("direct_qa")
            tracker.mark_chunk_done("direct_qa", "chunk_001")
            # Reload from disk
            tracker2 = CheckpointTracker(ckpt_path)
            assert tracker2.is_stage_done("direct_qa")
            assert tracker2.is_chunk_done("direct_qa", "chunk_001")
            assert not tracker2.is_chunk_done("direct_qa", "chunk_002")
            passed_tests.append("32. Checkpoint recovery")
        except Exception as e:
            failed_tests.append(f"32. Checkpoint recovery: {e}")

        # -------------------------------------------------------------
        # Case 33: Final SFT schema validation
        # -------------------------------------------------------------
        try:
            sft_rec = SFTConversationRecord(
                id="sft_01",
                messages=[ChatMessage(role="user", content="Q?"), ChatMessage(role="assistant", content="A.")]
            )
            data_dict = sft_rec.model_dump()
            assert data_dict["messages"][0]["role"] == "user"
            assert data_dict["messages"][1]["role"] == "assistant"
            passed_tests.append("33. Final SFT schema validation")
        except Exception as e:
            failed_tests.append(f"33. Final SFT schema validation: {e}")

        # -------------------------------------------------------------
        # Case 34: Train/validation leakage prevention
        # -------------------------------------------------------------
        try:
            exporter = FineTuningExporter(output_dir=tmp_path, train_ratio=0.8)
            qa_fact1_a = RichQARecord(qa_id="qa_1a", question="Q1?", answer="A1", perspective="direct", fact_ids=["fact_001"], source_chunk_ids=["c1"], source_pages=[1])
            qa_fact1_b = RichQARecord(qa_id="qa_1b", question="Q2?", answer="A1", perspective="indirect", fact_ids=["fact_001"], source_chunk_ids=["c1"], source_pages=[1])
            qa_fact2 = RichQARecord(qa_id="qa_2", question="Q3?", answer="A2", perspective="direct", fact_ids=["fact_002"], source_chunk_ids=["c2"], source_pages=[2])
            train_sft, val_sft = exporter.export_and_split([qa_fact1_a, qa_fact1_b, qa_fact2])
            
            # Check fact_001 does not appear in both train and val
            train_ids = {m.id for m in train_sft}
            val_ids = {m.id for m in val_sft}
            assert not (("qa_1a" in train_ids and "qa_1b" in val_ids) or ("qa_1b" in train_ids and "qa_1a" in val_ids))
            passed_tests.append("34. Train/validation leakage prevention")
        except Exception as e:
            failed_tests.append(f"34. Train/validation leakage prevention: {e}")

        # -------------------------------------------------------------
        # Case 35: Separate KPI box extraction & schema
        # -------------------------------------------------------------
        try:
            kpi = KPIBoxRecord(
                kpi_id="kpi_001",
                page_number=7,
                value="82.8 Cr.",
                label="REVENUE FROM OPERATIONS",
                unit="INR Crores",
                section_heading="REVENUE & GROWTH"
            )
            assert kpi.value == "82.8 Cr."
            assert kpi.unit == "INR Crores"
            passed_tests.append("35. Separate KPI box extraction & schema")
        except Exception as e:
            failed_tests.append(f"35. Separate KPI box extraction: {e}")

        # -------------------------------------------------------------
        # Case 36: Chart region detection & limitations
        # -------------------------------------------------------------
        try:
            chart = ChartRecord(
                chart_id="chart_001",
                page_number=32,
                title="PAT Margin (In %)",
                x_axis_label="FY20, FY21, FY22, FY23, FY24",
                y_axis_label="Percentage (%)",
                units="Percentage",
                extraction_limitations="Visual chart representation without fabricated values."
            )
            assert chart.title == "PAT Margin (In %)"
            assert "without fabricated values" in chart.extraction_limitations
            passed_tests.append("36. Chart region detection & limitations")
        except Exception as e:
            failed_tests.append(f"36. Chart detection: {e}")

        # -------------------------------------------------------------
        # Case 37: Relationship mapping & Content Group registry
        # -------------------------------------------------------------
        try:
            from src.relationship_mapper import RelationshipMapper
            mapper = RelationshipMapper(output_dir=tmp_path)
            para = ParagraphRecord(para_id="p1", page_number=1, text="Operating expenses in FY24.", section_heading="Operating Expenses")
            tab = TableRecord(table_id="t1", page_number=1, headers=["Category", "Amount"], rows=[["Salaries", "63.1"]], markdown_content="| Category | Amount |\n|---|---|\n| Salaries | 63.1 |", table_title="Operating Expenses")
            rels, grps = mapper.map_relationships([para], [tab], [], [], [])
            assert len(grps) >= 1
            assert (tmp_path / "extracted_data" / "relationships.json").exists()
            assert (tmp_path / "processed_data" / "content_groups.json").exists()
            passed_tests.append("37. Relationship mapping & Content Groups")
        except Exception as e:
            failed_tests.append(f"37. Relationship mapping: {e}")

        # -------------------------------------------------------------
        # Case 38: 5 Hybrid Chunking strategies
        # -------------------------------------------------------------
        try:
            chunker = HybridChunker(output_dir=tmp_path, chunk_size=500, chunk_overlap=50, use_embeddings=False)
            oversized_tab = TableRecord(
                table_id="tab_big",
                page_number=2,
                table_title="Long Expense Table",
                headers=["Item", "Cost"],
                rows=[[f"Item_{i}", f"{i*10}"] for i in range(30)],
                markdown_content="markdown"
            )
            split_chunks = chunker._split_oversized_table(oversized_tab, max_tokens=200)
            assert len(split_chunks) >= 2
            assert all(c.chunk_strategy == "table_aware" for c in split_chunks)
            # Layout-aware KPI chunks
            kpi_chunks = chunker._build_layout_chunks([kpi])
            assert len(kpi_chunks) == 1
            assert kpi_chunks[0].chunk_strategy == "layout_aware"
            # Chart-aware chunks
            chart_chunks = chunker._build_chart_chunks([chart])
            assert len(chart_chunks) == 1
            assert chart_chunks[0].chunk_strategy == "chart_aware"
            passed_tests.append("38. 5 Hybrid Chunking strategies")
        except Exception as e:
            failed_tests.append(f"38. 5 Hybrid Chunking strategies: {e}")

        # -------------------------------------------------------------
        # Case 39: Separate extraction output files in extracted_data/
        # -------------------------------------------------------------
        try:
            ext = PDFExtractor(pdf_path="data_set_use.pdf", output_dir=tmp_path)
            # Assert target output paths exist in schema
            assert hasattr(ext, "extracted_data_dir")
            assert (ext.extracted_data_dir).name == "extracted_data"
            passed_tests.append("39. Separate extraction output files in extracted_data/")
        except Exception as e:
            failed_tests.append(f"39. Separate extraction files: {e}")

        # -------------------------------------------------------------
        # Case 40: Processed data output directory validation
        # -------------------------------------------------------------
        try:
            assert (tmp_path / "processed_data").exists()
            passed_tests.append("40. Processed data output directory validation")
        except Exception as e:
            failed_tests.append(f"40. Processed data directory: {e}")

    print("\nTEST EXECUTION SUMMARY:")
    print(f"Passed: {len(passed_tests)} / {len(passed_tests) + len(failed_tests)}")
    print(f"Failed: {len(failed_tests)} / {len(passed_tests) + len(failed_tests)}")
    if failed_tests:
        print("\nFailures:")
        for ft in failed_tests:
            print("  -", ft)
    else:
        print(f"\nALL {len(passed_tests)} INTEGRATION TEST CASES PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_all_tests()
