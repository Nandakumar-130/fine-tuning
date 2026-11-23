from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 1. Extraction Models
# ---------------------------------------------------------------------------
class BoundingBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float

class TextBlockRecord(BaseModel):
    text_id: str
    page_number: int
    text: str
    source: str = Field(default="native_text", description="'native_text' or 'ocr'")
    bbox: Optional[List[float]] = None
    reading_order_idx: int = 0
    confidence: float = 1.0
    verified: bool = True

class TableRecord(BaseModel):
    table_id: str
    page_number: int
    table_title: Optional[str] = None
    headers: List[str]
    rows: List[List[str]]
    markdown_content: str
    bbox: Optional[List[float]] = None
    units: Optional[str] = None
    currency: Optional[str] = None
    reporting_period: Optional[str] = None
    footnotes: List[str] = Field(default_factory=list)

class ParagraphRecord(BaseModel):
    para_id: str
    page_number: int
    text: str
    section_heading: Optional[str] = None
    bbox: Optional[List[float]] = None
    reading_order_idx: int = 0
    confidence: float = 1.0

class ScannedTextRecord(BaseModel):
    scanned_id: str
    page_number: int
    text: str
    confidence: float
    bbox: Optional[List[float]] = None
    reading_order_idx: int = 0

class ChartRecord(BaseModel):
    chart_id: str
    page_number: int
    title: Optional[str] = None
    x_axis_label: Optional[str] = None
    y_axis_label: Optional[str] = None
    units: Optional[str] = None
    legends: List[str] = Field(default_factory=list)
    data_labels: List[str] = Field(default_factory=list)
    numerical_values: Dict[str, Any] = Field(default_factory=dict)
    bbox: Optional[List[float]] = None
    extraction_limitations: Optional[str] = None
    image_ref: Optional[str] = None
    confidence: float = 1.0

class KPIBoxRecord(BaseModel):
    kpi_id: str
    page_number: int
    value: str
    label: str
    unit: Optional[str] = None
    section_heading: Optional[str] = None
    bbox: Optional[List[float]] = None
    extraction_method: str = "layout_box_detection"
    confidence: float = 1.0

class ImageRecord(BaseModel):
    image_id: str
    page_number: int
    image_path: str
    format: str = "png"
    width: int
    height: int
    bbox: Optional[List[float]] = None
    description: Optional[str] = None
    caption: Optional[str] = None
    related_text_ids: List[str] = Field(default_factory=list)
    association_status: str = Field(
        default="uncertain",
        description="'verified', 'uncertain', or 'not_related'"
    )

# ---------------------------------------------------------------------------
# 2. Verification Records
# ---------------------------------------------------------------------------
class VerifiedRecord(BaseModel):
    record_id: str
    page_number: int
    raw_text: str
    corrected_text: str
    verification_status: str = Field(
        default="PASS",
        description="'PASS', 'WARNING', 'MANUAL_REVIEW'"
    )
    warnings: List[str] = Field(default_factory=list)
    correction_reason: Optional[str] = None
    bbox: Optional[List[float]] = None

# ---------------------------------------------------------------------------
# 3. Page Organization Records
# ---------------------------------------------------------------------------
class PageRecord(BaseModel):
    page_number: int
    text_blocks: List[TextBlockRecord] = Field(default_factory=list)
    table_ids: List[str] = Field(default_factory=list)
    image_ids: List[str] = Field(default_factory=list)
    total_characters: int = 0
    has_ocr_content: bool = False

# ---------------------------------------------------------------------------
# 4. Chunk Record
# ---------------------------------------------------------------------------
class ChunkRecord(BaseModel):
    chunk_id: str
    page_numbers: List[int]
    content: str
    chunk_strategy: str = Field(
        default="heading_based",
        description="'heading_based', 'table_aware', 'layout_aware', 'chart_aware', or 'token_sentence_based'"
    )
    section_heading: Optional[str] = None
    content_group_id: Optional[str] = None
    image_refs: List[str] = Field(default_factory=list)
    table_refs: List[str] = Field(default_factory=list)
    chart_refs: List[str] = Field(default_factory=list)
    kpi_refs: List[str] = Field(default_factory=list)
    token_count: int
    source_text_ids: List[str] = Field(default_factory=list)
    experiment_tag: Optional[str] = None  # e.g., '400_tokens', '600_tokens', '800_tokens'

# ---------------------------------------------------------------------------
# 5. Relationship & Content Group Records
# ---------------------------------------------------------------------------
class RelationshipRecord(BaseModel):
    relationship_id: str
    source_record_id: str
    target_record_id: str
    relationship_type: str = Field(
        description="'heading_match', 'section_hierarchy', 'table_reference', 'chart_caption', 'cross_page_continuity', or 'semantic_similarity'"
    )
    reasons: List[str] = Field(default_factory=list)
    status: str = Field(
        default="verified",
        description="'verified' or 'uncertain'"
    )
    confidence: float = 1.0

class ContentGroupRecord(BaseModel):
    group_id: str
    topic: str
    section: str
    source_record_ids: List[str] = Field(default_factory=list)
    chunk_ids: List[str] = Field(default_factory=list)
    page_numbers: List[int] = Field(default_factory=list)
    relationship_reasons: List[str] = Field(default_factory=list)
    status: str = Field(default="verified", description="'verified' or 'uncertain'")
    confidence: float = 1.0

# ---------------------------------------------------------------------------
# 6. Facts & Entity Index Records
# ---------------------------------------------------------------------------
class FactRecord(BaseModel):
    fact_id: str
    subject: str
    relation: str
    object: str
    location: Optional[str] = None
    date_or_time: Optional[str] = None
    source_chunk_ids: List[str]
    source_pages: List[int]
    verified: bool = True
    confidence: float = 1.0

class EntityIndexRecord(BaseModel):
    entity: str
    normalized_name: str
    related_chunk_ids: List[str]
    related_pages: List[int]
    fact_ids: List[str] = Field(default_factory=list)
    relationship_status: str = Field(
        default="verified",
        description="'verified' or 'candidate'"
    )

class RelatedChunkGroup(BaseModel):
    group_id: str
    entity_name: str
    chunk_ids: List[str]
    page_numbers: List[int]
    combined_evidence: str
    verification_notes: Optional[str] = None

# ---------------------------------------------------------------------------
# 6. Q&A Generation & Audit Records
# ---------------------------------------------------------------------------
class QAPairItem(BaseModel):
    question: str
    answer: str
    perspective: str = Field(
        description="'direct', 'indirect', 'attribute', 'combined', or 'cross_page'"
    )
    fact_ids: List[str] = Field(default_factory=list)
    source_chunk_ids: List[str] = Field(default_factory=list)
    source_pages: List[int] = Field(default_factory=list)
    parent_qa_id: Optional[str] = None

class QAResponseSchema(BaseModel):
    items: List[QAPairItem] = Field(default_factory=list)

class RichQARecord(BaseModel):
    qa_id: str
    question: str
    answer: str
    perspective: str
    parent_qa_id: Optional[str] = None
    fact_ids: List[str] = Field(default_factory=list)
    source_chunk_ids: List[str] = Field(default_factory=list)
    source_pages: List[int] = Field(default_factory=list)
    evaluation_status: str = Field(
        default="PENDING",
        description="'PASS', 'REJECT', 'REVIEW', or 'PENDING'"
    )
    evaluation_reason: Optional[str] = None
    unsupported_claims: List[str] = Field(default_factory=list)
    normalized_question: Optional[str] = None

# ---------------------------------------------------------------------------
# 7. Independent Evaluation Schema
# ---------------------------------------------------------------------------
class EvaluationResult(BaseModel):
    qa_id: str
    evidence_supported: bool
    answerable: bool
    complete: bool
    relevant: bool
    unsupported_claims: List[str] = Field(default_factory=list)
    decision: str = Field(
        description="'PASS', 'REJECT', or 'REVIEW'"
    )
    reason: str

# ---------------------------------------------------------------------------
# 8. SFT Format (Chat Conversation)
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str

class SFTConversationRecord(BaseModel):
    id: Optional[str] = None
    messages: List[ChatMessage]
    # Keep metadata separate or audit reference
    metadata: Optional[Dict[str, Any]] = None
