import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging

from src.schemas import RichQARecord, EvaluationResult, ChunkRecord
from src.io_utils import ensure_directory, write_jsonl, append_jsonl_record
from src.vllm_engine import VLLMClient

logger = logging.getLogger("IndependentEvaluator")

class IndependentEvaluator:
    """
    Step 10: Independent Evaluation and Error Recovery
    PDF Requirements:
    - Generation and evaluation MUST be separate tasks and separate contexts.
    - Uses Prompt F:
      SYSTEM: You are a strict evidence verifier. You did not generate this Q&A.
      Judge only against original evidence. Do not assume a plausible answer is correct.
    - Checks:
      1. Evidence support (every claim supported)
      2. Relevance to supplied evidence
      3. Hallucination (invented or contradicted detail)
      4. Completeness
      5. Answerability
      6. Clarity & precision of numbers, dates, units
      7. Provenance: chunk IDs and page references valid
    - Decisions:
      PASS: All required checks pass.
      REJECT: Unsupported, unanswerable or materially incorrect.
      REVIEW: Ambiguous evidence or uncertain association.
    - Saves to:
      quality/evaluation_results.jsonl
      quality/rejected_qa.jsonl
      final/approved_qa.jsonl
    """
    def __init__(self, output_dir: str | Path, vllm_client: VLLMClient):
        self.output_dir = Path(output_dir)
        self.quality_dir = self.output_dir / "quality"
        self.final_dir = self.output_dir / "final"
        ensure_directory(self.quality_dir)
        ensure_directory(self.final_dir)
        self.llm = vllm_client

    def evaluate_qa_item(
        self,
        qa: RichQARecord,
        source_evidence: str
    ) -> EvaluationResult:
        # Check basic provenance first
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
            decision = str(data.get("decision", "REVIEW")).upper()
            if "PASS" in decision:
                decision = "PASS"
            elif "REJECT" in decision:
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

    def evaluate_all(
        self,
        qa_records: List[RichQARecord],
        chunks: List[ChunkRecord]
    ) -> Tuple[List[RichQARecord], List[RichQARecord], List[RichQARecord]]:
        logger.info(f"Running independent evaluation for {len(qa_records)} Q&A pairs against original source evidence...")
        chunk_dict = {c.chunk_id: c.content for c in chunks}

        eval_results_file = self.quality_dir / "evaluation_results.jsonl"
        rejected_file = self.quality_dir / "rejected_qa.jsonl"
        approved_file = self.final_dir / "approved_qa.jsonl"

        passed: List[RichQARecord] = []
        rejected: List[RichQARecord] = []
        reviewed: List[RichQARecord] = []

        for qa in qa_records:
            # Gather exact original source evidence for this item
            evidence_parts = [chunk_dict[cid] for cid in qa.source_chunk_ids if cid in chunk_dict]
            source_evidence = "\n\n".join(evidence_parts) if evidence_parts else "No evidence text available."

            eval_res = self.evaluate_qa_item(qa, source_evidence)
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
            else:
                reviewed.append(qa)

        logger.info(
            f"Independent evaluation complete:\n"
            f"  - PASS: {len(passed)}\n"
            f"  - REJECT: {len(rejected)}\n"
            f"  - REVIEW: {len(reviewed)}"
        )
        return passed, rejected, reviewed

    def _extract_json_block(self, text: str) -> str:
        text = text.strip()
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if match:
            return match.group(1).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return text[start:end+1]
        return text
