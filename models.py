import time
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentRecord(BaseModel):
    id: str
    text: str
    source: str = "stream"
    metadata: Dict[str, str] = Field(default_factory=dict)


class QualityScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    word_count: int = Field(ge=0)
    char_count: int = Field(ge=0)
    alnum_ratio: float = Field(ge=0.0, le=1.0)
    unique_word_ratio: float = Field(ge=0.0, le=1.0)
    passed: bool
    reason: Optional[str] = None


class RedactionResult(BaseModel):
    cleaned_text: str
    pii_matches: int = 0
    toxicity_matches: int = 0
    redacted_types: List[str] = Field(default_factory=list)


class DeduplicationResult(BaseModel):
    is_duplicate: bool = False
    matched_id: Optional[str] = None
    method: Optional[str] = None
    similarity_score: float = 0.0


class AuditEntry(BaseModel):
    doc_id: str
    status: str
    simhash: Optional[int] = None
    quality_score: float = 0.0
    pii_redactions: int = 0
    toxicity_redactions: int = 0
    rejection_reason: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)


class PipelineConfig(BaseModel):
    min_quality_score: float = 0.4
    min_words: int = 5
    simhash_hamming_threshold: int = 3
    minhash_similarity_threshold: float = 0.75
    redact_pii: bool = True
    redact_toxicity: bool = True
    chunk_size: int = 50


class PipelineSummary(BaseModel):
    total_ingested: int = 0
    passed: int = 0
    deduplicated: int = 0
    low_quality_rejected: int = 0
    pii_redacted_count: int = 0
    toxicity_redacted_count: int = 0
    elapsed_time_seconds: float = 0.0
