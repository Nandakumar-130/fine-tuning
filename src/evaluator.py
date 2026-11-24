from __future__ import annotations
import json
import re
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

from src.schemas import RichQARecord, EvaluationResult, ChunkRecord
from src.io_utils import ensure_directory, write_jsonl, append_jsonl_record
from src.vllm_engine import VLLMClient

logger = logging.getLogger("IndependentEvaluator")

class IndependentEvaluator:
    """
    Step 10: Independent Evaluation & Regenerative Error Recovery
    PDF Requirements (Pages 8, 10, 12 Prompt F):
    - Independent Qwen-as-a-Judge: Runs in a completely separate, clean context from generation.
    - System role: Strict evidence verifier. Does not assume a plausible answer is correct.
    - Rigorous Verification Criteria:
        1. Evidence Support: Every single factual claim must be explicitly supported by source.
        2. Relevance: Question must directly relate to source evidence.
        3. Hallucination Guard: Identifies any invented or contradicted details.
        4. Completeness: Answer addresses every part of the question.
        5. Answerability: Evidence contains enough information to answer.
        6. Precision: Exact validation for names, numbers, dates, and units.
        7. Provenance: Valid chunk IDs and page references.
    - Decision: PASS, REJECT, or REVIEW.
    - Regenerative Feedback Loop (Requested Feature):
        When a Q&A is REJECTED due to unsupported claims or inaccuracy, the evaluator feeds
        the exact rejection reason and unsupported claims back to Qwen to regenerate a corrected,
        fully grounded Q&A pair. If it passes upon re-verification, it is saved; otherwise rejected.
    """
    def __init__(self, output_dir: str | Path, vllm_client: VLLMClient):
        self.output_dir = Path(output_dir)
        self.quality_dir = self.output_dir / "quality"
        self.final_dir = self.output_dir / "final"
        self.processed_dir = self.output_dir / "processed_data"
        ensure_directory(self.quality_dir)
        ensure_directory(self.final_dir)
        ensure_directory(self.processed_dir)
        self.llm = vllm_client

    def _extract_json_block(self, text: str) -> str:
        text = text.strip()
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if match:
            return match.group(1).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start:end+1]
        return text

    def evaluate_qa_item(
        self,
        qa: RichQARecord,
        source_evidence: str
    ) -> EvaluationResult:
        """
        Qwen-as-a-Judge evaluation using Prompt F from PDF Page 12.
        """
        if not qa.source_chunk_ids or not qa.source_pages:
            return EvaluationResult(
                qa_id=qa.qa_id,
                evidence_supported=False,
                answerable=False,
                complete=False,
                relevant=False,
                unsupported_claims=["Missing valid source chunk IDs or page references."],
                decision="REJECT",
                reason="Invalid provenance."
            )

        system_prompt = (
            "You are a strict evidence verifier. You did not generate this Q&A. "
            "Judge only against the original evidence. Do not assume a plausible answer is correct. "
            "Return valid JSON only."
        )

        user_prompt = (
            f"ORIGINAL_SOURCE_EVIDENCE:\n{source_evidence}\n\n"
            f"QUESTION:\n{qa.question}\n\n"
            f"ANSWER:\n{qa.answer}\n\n"
            "TASK: Check evidence support for every claim, answerability, completeness, "
            "relevance, clarity, and exact names/numbers/dates/units. Identify unsupported or contradicted claims.\n"
            "Return JSON format:\n"
            "{\n"
            '  "evidence_supported": true,\n'
            '  "answerable": true,\n'
            '  "complete": true,\n'
            '  "relevant": true,\n'
            '  "unsupported_claims": [],\n'
            '  "decision": "PASS or REJECT or REVIEW",\n'
            '  "reason": "Short evidence-based explanation"\n'
            "}"
        )

        response_text = self.llm.generate(system_prompt, user_prompt, temperature=0.0)

        try:
            clean_json = self._extract_json_block(response_text)
            data = json.loads(clean_json)
            raw_decision = str(data.get("decision", "REVIEW")).upper()
            if "PASS" in raw_decision:
                decision = "PASS"
            elif "REJECT" in raw_decision:
                decision = "REJECT"
            else:
                decision = "REVIEW"

            return EvaluationResult(
                qa_id=qa.qa_id,
                evidence_supported=bool(data.get("evidence_supported", False)),
                answerable=bool(data.get("answerable", False)),
                complete=bool(data.get("complete", False)),
                relevant=bool(data.get("relevant", False)),
                unsupported_claims=data.get("unsupported_claims", []),
                decision=decision,
                reason=str(data.get("reason", "Evaluated by Prompt F"))
            )
        except Exception as e:
            logger.warning(f"Failed to parse evaluation response for {qa.qa_id}: {e}")
            return EvaluationResult(
                qa_id=qa.qa_id,
                evidence_supported=False,
                answerable=False,
                complete=False,
                relevant=False,
                unsupported_claims=["Format error in evaluator output"],
                decision="REVIEW",
                reason=f"Evaluation parsing exception: {e}"
            )

    def regenerate_rejected_qa(
        self,
        qa: RichQARecord,
        source_evidence: str,
        eval_result: EvaluationResult
    ) -> Tuple[RichQARecord, EvaluationResult]:
        """
        Feedback Loop: Tells Qwen that this Q&A was rejected, gives the exact judge reason
        and unsupported claims, and asks Qwen to regenerate properly grounded in source evidence.
        """
        rejection_feedback = (
            f"PREVIOUS QUESTION: {qa.question}\n"
            f"PREVIOUS REJECTED ANSWER: {qa.answer}\n"
            f"JUDGE REJECTION REASON: {eval_result.reason}\n"
            f"UNSUPPORTED CLAIMS: {eval_result.unsupported_claims}\n"
        )

        system_prompt = (
            "You are a strict, evidence-grounded Q&A generator. "
            "A previous Q&A formulation was REJECTED by an independent judge for hallucination or inaccuracy. "
            "Your task is to fix the error and generate a completely accurate, fully supported Q&A pair "
            "using ONLY the provided source evidence. Do not guess or invent details. Return valid JSON only."
        )

        user_prompt = (
            f"ORIGINAL_SOURCE_EVIDENCE:\n{source_evidence}\n\n"
            f"REJECTION_REPORT:\n{rejection_feedback}\n\n"
            f"TASK: Regenerate this {qa.perspective} Q&A item properly. "
            "Ensure the question is directly answerable from the source evidence and every part of the answer "
            "is 100% verified by the text. Format:\n"
            '{"question": "...", "answer": "..."}'
        )

        try:
            regen_response = self.llm.generate(system_prompt, user_prompt, temperature=0.1)
            clean_json = self._extract_json_block(regen_response)
            data = json.loads(clean_json)
            new_q = str(data.get("question", "")).strip()
            new_a = str(data.get("answer", "")).strip()

            if len(new_q) >= 5 and len(new_a) >= 2:
                regenerated_qa = RichQARecord(
                    qa_id=f"{qa.qa_id}_regen",
                    question=new_q,
                    answer=new_a,
                    perspective=qa.perspective,
                    parent_qa_id=qa.parent_qa_id,
                    fact_ids=qa.fact_ids,
                    source_chunk_ids=qa.source_chunk_ids,
                    source_pages=qa.source_pages,
                    evaluation_status="PENDING"
                )
                # Re-evaluate the regenerated Q&A through Qwen-as-a-Judge
                new_eval = self.evaluate_qa_item(regenerated_qa, source_evidence)
                return regenerated_qa, new_eval
        except Exception as e:
            logger.warning(f"Regeneration attempt failed for {qa.qa_id}: {e}")

        # If regeneration failed or was unparseable, return original rejection
        return qa, eval_result

    def evaluate_all(
        self,
        qa_records: List[RichQARecord],
        chunks: List[ChunkRecord]
    ) -> Tuple[List[RichQARecord], List[RichQARecord], List[RichQARecord]]:
        logger.info(f"Running independent Qwen-as-a-Judge evaluation for {len(qa_records)} Q&A pairs...")
        chunk_dict = {c.chunk_id: c.content for c in chunks}

        eval_results_file = self.quality_dir / "evaluation_results.jsonl"
        rejected_file = self.quality_dir / "rejected_qa.jsonl"
        approved_file = self.final_dir / "approved_qa.jsonl"
        proc_rejected_file = self.processed_dir / "rejected_qa.jsonl"

        passed: List[RichQARecord] = []
        rejected: List[RichQARecord] = []
        reviewed: List[RichQARecord] = []

        pbar = tqdm(qa_records, desc="[Qwen-as-a-Judge Evaluation]", unit="qa", file=sys.stdout)
        for qa in pbar:
            evidence_parts = [chunk_dict[cid] for cid in qa.source_chunk_ids if cid in chunk_dict]
            source_evidence = "\n\n".join(evidence_parts) if evidence_parts else "No evidence text available."

            eval_res = self.evaluate_qa_item(qa, source_evidence)

            # If rejected, execute feedback loop: tell Qwen it was rejected and regenerate
            if eval_res.decision == "REJECT":
                tqdm.write(f"  [JUDGE REJECTED] {qa.qa_id}: {eval_res.reason}. Resending to Qwen to regenerate...")
                regen_qa, new_eval = self.regenerate_rejected_qa(qa, source_evidence, eval_res)
                if new_eval.decision == "PASS":
                    tqdm.write(f"  [REGEN SUCCESS] {regen_qa.qa_id} successfully fixed and PASSED!")
                    qa = regen_qa
                    eval_res = new_eval
                else:
                    tqdm.write(f"  [REGEN FAILED] {regen_qa.qa_id} still unsupported ({new_eval.reason}). Final REJECT.")

            append_jsonl_record(eval_results_file, eval_res)

            qa.evaluation_status = eval_res.decision
            qa.evaluation_reason = eval_res.reason
            qa.unsupported_claims = eval_res.unsupported_claims

            if eval_res.decision == "PASS":
                passed.append(qa)
                append_jsonl_record(approved_file, qa)
            elif eval_res.decision == "REJECT":
                rejected.append(qa)
                append_jsonl_record(rejected_file, qa)
                append_jsonl_record(proc_rejected_file, qa)
            else:
                reviewed.append(qa)

            pbar.set_postfix({"PASS": len(passed), "REJECT": len(rejected), "REVIEW": len(reviewed)})

        # Save qa_evaluation_report.json
        eval_report = {
            "total_evaluated": len(qa_records),
            "approved_pass_count": len(passed),
            "rejected_count": len(rejected),
            "manual_review_count": len(reviewed),
            "pass_rate": round(len(passed) / max(1, len(qa_records)), 4),
            "rejection_reasons_summary": [
                {"qa_id": r.qa_id, "reason": r.evaluation_reason, "unsupported_claims": r.unsupported_claims}
                for r in rejected[:50]
            ]
        }
        with open(self.processed_dir / "qa_evaluation_report.json", "w", encoding="utf-8") as f:
            json.dump(eval_report, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Independent evaluation complete:\n"
            f"  - PASS (Approved): {len(passed)}\n"
            f"  - REJECT (Logged with reasons): {len(rejected)}\n"
            f"  - REVIEW (Flagged): {len(reviewed)}"
        )
        return passed, rejected, reviewed
