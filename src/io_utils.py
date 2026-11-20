import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Generator, Optional, Type, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger("PipelineIO")

def ensure_directory(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def write_jsonl(file_path: Path | str, records: List[BaseModel | Dict[str, Any]], append: bool = False) -> None:
    path = Path(file_path)
    ensure_directory(path.parent)
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for record in records:
            if isinstance(record, BaseModel):
                line = record.model_dump_json()
            else:
                line = json.dumps(record, ensure_ascii=False)
            f.write(line + "\n")

def append_jsonl_record(file_path: Path | str, record: BaseModel | Dict[str, Any]) -> None:
    path = Path(file_path)
    ensure_directory(path.parent)
    with open(path, "a", encoding="utf-8") as f:
        if isinstance(record, BaseModel):
            line = record.model_dump_json()
        else:
            line = json.dumps(record, ensure_ascii=False)
        f.write(line + "\n")
        f.flush()  # Immediate write for live persistence

def read_jsonl(file_path: Path | str, model_cls: Optional[Type[T]] = None) -> List[Any]:
    path = Path(file_path)
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if model_cls:
                    records.append(model_cls.model_validate(data))
                else:
                    records.append(data)
            except Exception as e:
                logger.error(f"Error parsing line {line_num} in {file_path}: {e}")
    return records

def stream_jsonl(file_path: Path | str) -> Generator[Dict[str, Any], None, None]:
    path = Path(file_path)
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

# ---------------------------------------------------------------------------
# Structured Processing Logs & Checkpoint Tracker
# ---------------------------------------------------------------------------
class StructuredLogger:
    """
    Page 9 of PDF: logs/processing_logs.jsonl
    Logs every stage start, chunk completion, warning, and error with timestamps.
    """
    def __init__(self, log_dir: Path | str):
        self.log_dir = Path(log_dir)
        ensure_directory(self.log_dir)
        self.log_file = self.log_dir / "processing_logs.jsonl"

    def log_event(self, level: str, stage: str, message: str, details: Optional[Dict[str, Any]] = None):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level.upper(),
            "stage": stage,
            "message": message,
            "details": details or {}
        }
        append_jsonl_record(self.log_file, entry)

class CheckpointTracker:
    """
    Page 9 of PDF: logs/checkpoints.json
    Tracks completed stages and individual chunk IDs per stage.
    """
    def __init__(self, checkpoint_file: Path | str):
        self.checkpoint_file = Path(checkpoint_file)
        ensure_directory(self.checkpoint_file.parent)
        self.state: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}. Starting fresh.")
        return {
            "completed_stages": [],
            "completed_chunks_by_stage": {},
            "stage_metrics": {}
        }

    def _save(self) -> None:
        with open(self.checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)

    def is_chunk_done(self, stage_name: str, chunk_id: str) -> bool:
        chunks = self.state.get("completed_chunks_by_stage", {}).get(stage_name, [])
        return chunk_id in chunks

    def mark_chunk_done(self, stage_name: str, chunk_id: str) -> None:
        if "completed_chunks_by_stage" not in self.state:
            self.state["completed_chunks_by_stage"] = {}
        if stage_name not in self.state["completed_chunks_by_stage"]:
            self.state["completed_chunks_by_stage"][stage_name] = []
        if chunk_id not in self.state["completed_chunks_by_stage"][stage_name]:
            self.state["completed_chunks_by_stage"][stage_name].append(chunk_id)
            self._save()

    def mark_stage_done(self, stage_name: str) -> None:
        if stage_name not in self.state.get("completed_stages", []):
            self.state.setdefault("completed_stages", []).append(stage_name)
            self._save()

    def is_stage_done(self, stage_name: str) -> bool:
        return stage_name in self.state.get("completed_stages", [])

    def reset_stage(self, stage_name: str) -> None:
        if stage_name in self.state.get("completed_stages", []):
            self.state["completed_stages"].remove(stage_name)
        if stage_name in self.state.get("completed_chunks_by_stage", {}):
            del self.state["completed_chunks_by_stage"][stage_name]
        self._save()
