from __future__ import annotations
import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # PDF & Path settings
    PDF_FILE_PATH: str = Field(
        default="data_set_use.pdf",
        description="Path to target input PDF file"
    )
    OUTPUT_DIR: str = Field(
        default="PDF_QA_PIPELINE",
        description="Base directory for all pipeline stage outputs"
    )

    # vLLM / Local Model Settings
    VLLM_MODEL_NAME_OR_PATH: str = Field(
        default="Qwen/Qwen2.5-3B-Instruct",
        description="Local path or HuggingFace ID for Qwen LLM"
    )
    VLLM_VL_MODEL_NAME_OR_PATH: str = Field(
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        description="Local path or HuggingFace ID for Qwen Vision-Language model"
    )
    VLLM_GPU_MEMORY_UTILIZATION: float = Field(
        default=0.85,
        description="GPU memory fraction reserved for vLLM"
    )
    VLLM_MAX_MODEL_LEN: int = Field(
        default=4096,
        description="Max sequence length for vLLM generation"
    )
    TENSOR_PARALLEL_SIZE: int = Field(
        default=1,
        description="Tensor parallel degree for multi-GPU setups"
    )
    USE_VLLM: bool = Field(
        default=True,
        description="Whether to use vLLM backend or HuggingFace Transformers fallback"
    )

    # Embedding model settings
    EMBEDDING_MODEL_NAME_OR_PATH: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Local or HuggingFace ID for SentenceTransformers embeddings"
    )
    SIMILARITY_THRESHOLD: float = Field(
        default=0.75,
        description="Cosine similarity threshold for cross-page grouping / topic shift"
    )

    # OCR Settings
    OCR_LANG: str = Field(default="en", description="PaddleOCR language")
    USE_GPU_OCR: bool = Field(default=False, description="Enable GPU for PaddleOCR if available")
    OCR_CONFIDENCE_ACCEPT: float = Field(default=0.95, description="Provisional accept threshold")
    OCR_CONFIDENCE_REVIEW: float = Field(default=0.80, description="Review threshold")

    # Chunking Settings
    CHUNK_SIZE: int = Field(default=500, description="Target Qwen tokens per chunk (test 400, 600, 800)")
    CHUNK_OVERLAP: int = Field(default=50, description="Overlap in tokens between chunks")

    # Q&A Target
    MAX_QA_PER_PERSPECTIVE: int = Field(default=10, description="Max Q&A pairs per stage (up to 10)")
    BATCH_SIZE: int = Field(default=4, description="Batch size for LLM inference")

    # SFT Split
    TRAIN_VAL_SPLIT_RATIO: float = Field(default=0.85, description="Train ratio for split by source fact/entity")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()
