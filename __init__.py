"""Training Data Intelligence Platform package."""

try:
    from .models import (
        DocumentRecord,
        QualityScore,
        RedactionResult,
        DeduplicationResult,
        AuditEntry,
        PipelineConfig,
        PipelineSummary,
    )
    from .engine import (
        compute_simhash,
        simhash_hamming_distance,
        compute_minhash,
        minhash_jaccard_similarity,
        score_text_quality,
        redact_pii_and_toxicity,
        AuditLedger,
        chunked_stream,
        run_pipeline,
    )
except ImportError:
    from models import (
        DocumentRecord,
        QualityScore,
        RedactionResult,
        DeduplicationResult,
        AuditEntry,
        PipelineConfig,
        PipelineSummary,
    )
    from engine import (
        compute_simhash,
        simhash_hamming_distance,
        compute_minhash,
        minhash_jaccard_similarity,
        score_text_quality,
        redact_pii_and_toxicity,
        AuditLedger,
        chunked_stream,
        run_pipeline,
    )

__all__ = [
    "DocumentRecord",
    "QualityScore",
    "RedactionResult",
    "DeduplicationResult",
    "AuditEntry",
    "PipelineConfig",
    "PipelineSummary",
    "compute_simhash",
    "simhash_hamming_distance",
    "compute_minhash",
    "minhash_jaccard_similarity",
    "score_text_quality",
    "redact_pii_and_toxicity",
    "AuditLedger",
    "chunked_stream",
    "run_pipeline",
]

