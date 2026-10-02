import hashlib
import random
import re
import time
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

try:
    from .models import (
        AuditEntry,
        DeduplicationResult,
        DocumentRecord,
        PipelineConfig,
        PipelineSummary,
        QualityScore,
        RedactionResult,
    )
except ImportError:
    from models import (
        AuditEntry,
        DeduplicationResult,
        DocumentRecord,
        PipelineConfig,
        PipelineSummary,
        QualityScore,
        RedactionResult,
    )

# Deterministic constants for 64 MinHash permutations
_MERSENNE_PRIME = 2147483647
_RNG = random.Random(42)
_MINHASH_A = [_RNG.randint(1, _MERSENNE_PRIME - 1) for _ in range(64)]
_MINHASH_B = [_RNG.randint(0, _MERSENNE_PRIME - 1) for _ in range(64)]

# Regex patterns for PII detection
_PII_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("PHONE", re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("CREDIT_CARD", re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")),
    ("IP_ADDRESS", re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b")),
    ("API_KEY", re.compile(r"\b(?:sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{20,}|AKIA[0-9A-Z]{16})\b")),
]

# Toxic / abusive terms heuristic pattern
_TOXIC_PATTERN = re.compile(
    r"\b(hate\s+speech|kill\s+yourself|die\s+in\s+a\s+fire|kys|asshole|bastard|slur|f[*u]ck|bitch)\b",
    re.IGNORECASE,
)


def chunked_stream(items: Iterable[DocumentRecord], chunk_size: int = 50) -> Iterator[List[DocumentRecord]]:
    """Yield successive chunks from an iterable stream."""
    chunk: List[DocumentRecord] = []
    for item in items:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def compute_simhash(text: str, bits: int = 64, k: int = 4) -> int:
    """Compute 64-bit SimHash fingerprint using character k-shingles."""
    cleaned = re.sub(r"\s+", " ", text.lower()).strip()
    if not cleaned:
        return 0
    if len(cleaned) < k:
        shingles = [cleaned]
    else:
        shingles = [cleaned[i : i + k] for i in range(len(cleaned) - k + 1)]

    v = [0] * bits
    for shingle in shingles:
        h = int(hashlib.md5(shingle.encode("utf-8")).hexdigest()[:16], 16)
        for i in range(bits):
            if (h >> i) & 1:
                v[i] += 1
            else:
                v[i] -= 1

    fingerprint = 0
    for i in range(bits):
        if v[i] > 0:
            fingerprint |= 1 << i
    return fingerprint


def simhash_hamming_distance(h1: int, h2: int) -> int:
    """Calculate the Hamming distance between two 64-bit hashes."""
    return bin(h1 ^ h2).count("1")


def _get_shingles(text: str, n: int = 3) -> List[str]:
    """Extract n-word shingles or falling back to words."""
    words = [w for w in re.findall(r"\w+", text.lower()) if w]
    if len(words) < n:
        return words
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def compute_minhash(text: str, num_perm: int = 64) -> List[int]:
    """Compute MinHash signature vector for text."""
    shingles = _get_shingles(text)
    if not shingles:
        return [0] * num_perm

    sig = [_MERSENNE_PRIME] * num_perm
    for shingle in shingles:
        h = int(hashlib.md5(shingle.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF
        for i in range(num_perm):
            val = (_MINHASH_A[i] * h + _MINHASH_B[i]) % _MERSENNE_PRIME
            if val < sig[i]:
                sig[i] = val
    return sig


def minhash_jaccard_similarity(sig1: List[int], sig2: List[int]) -> float:
    """Estimate Jaccard similarity between two MinHash signatures."""
    if not sig1 or not sig2 or len(sig1) != len(sig2):
        return 0.0
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)


def score_text_quality(text: str, min_words: int = 5, min_score: float = 0.4) -> QualityScore:
    """Evaluate text quality based on length, alphanumeric density, and repetition."""
    words = [w for w in re.findall(r"\w+", text) if w]
    word_count = len(words)
    char_count = len(text)

    if char_count == 0 or word_count == 0:
        return QualityScore(
            score=0.0,
            word_count=0,
            char_count=0,
            alnum_ratio=0.0,
            unique_word_ratio=0.0,
            passed=False,
            reason="Empty document text",
        )

    alnum_chars = sum(1 for c in text if c.isalnum() or c.isspace())
    alnum_ratio = round(alnum_chars / char_count, 4)
    unique_words = len(set(w.lower() for w in words))
    unique_ratio = round(unique_words / word_count, 4)

    length_factor = min(1.0, word_count / 30.0)
    score = round(0.35 * length_factor + 0.35 * alnum_ratio + 0.30 * unique_ratio, 4)

    if word_count < min_words:
        return QualityScore(
            score=score,
            word_count=word_count,
            char_count=char_count,
            alnum_ratio=alnum_ratio,
            unique_word_ratio=unique_ratio,
            passed=False,
            reason=f"Word count ({word_count}) below minimum threshold ({min_words})",
        )

    if alnum_ratio < 0.3:
        return QualityScore(
            score=score,
            word_count=word_count,
            char_count=char_count,
            alnum_ratio=alnum_ratio,
            unique_word_ratio=unique_ratio,
            passed=False,
            reason=f"Alphanumeric ratio ({alnum_ratio}) below acceptable threshold (0.3)",
        )

    if unique_ratio < 0.2 and word_count >= 15:
        return QualityScore(
            score=score,
            word_count=word_count,
            char_count=char_count,
            alnum_ratio=alnum_ratio,
            unique_word_ratio=unique_ratio,
            passed=False,
            reason=f"Excessive repetition ratio ({unique_ratio}) detected",
        )

    passed = score >= min_score
    reason = None if passed else f"Composite quality score ({score}) below threshold ({min_score})"

    return QualityScore(
        score=score,
        word_count=word_count,
        char_count=char_count,
        alnum_ratio=alnum_ratio,
        unique_word_ratio=unique_ratio,
        passed=passed,
        reason=reason,
    )


def redact_pii_and_toxicity(
    text: str, redact_pii: bool = True, redact_toxicity: bool = True
) -> RedactionResult:
    """Mask PII and toxic content using compiled regex heuristics."""
    cleaned = text
    pii_count = 0
    toxicity_count = 0
    redacted_types: List[str] = []

    if redact_pii:
        for pii_type, pattern in _PII_PATTERNS:
            matches = list(pattern.finditer(cleaned))
            if matches:
                pii_count += len(matches)
                redacted_types.append(pii_type)
                cleaned = pattern.sub(f"[REDACTED_{pii_type}]", cleaned)

    if redact_toxicity:
        matches = list(_TOXIC_PATTERN.finditer(cleaned))
        if matches:
            toxicity_count += len(matches)
            redacted_types.append("TOXICITY")
            cleaned = _TOXIC_PATTERN.sub("[REDACTED_TOXIC]", cleaned)

    return RedactionResult(
        cleaned_text=cleaned,
        pii_matches=pii_count,
        toxicity_matches=toxicity_count,
        redacted_types=redacted_types,
    )


class AuditLedger:
    """Append-only audit ledger tracking every document lifecycle step."""

    def __init__(self) -> None:
        self.entries: List[AuditEntry] = []
        self.start_time: float = time.time()

    def record(self, entry: AuditEntry) -> None:
        self.entries.append(entry)

    def summary(self) -> PipelineSummary:
        total = len(self.entries)
        passed = sum(1 for e in self.entries if e.status == "ACCEPTED")
        dedup = sum(1 for e in self.entries if e.status == "DEDUPLICATED")
        rejected = sum(1 for e in self.entries if e.status.startswith("REJECTED"))
        pii_redactions = sum(e.pii_redactions for e in self.entries)
        tox_redactions = sum(e.toxicity_redactions for e in self.entries)
        elapsed = round(time.time() - self.start_time, 4)

        return PipelineSummary(
            total_ingested=total,
            passed=passed,
            deduplicated=dedup,
            low_quality_rejected=rejected,
            pii_redacted_count=pii_redactions,
            toxicity_redacted_count=tox_redactions,
            elapsed_time_seconds=elapsed,
        )


def check_duplicate(
    doc_id: str,
    simhash_val: int,
    minhash_sig: List[int],
    simhash_index: Dict[str, int],
    minhash_index: Dict[str, List[int]],
    simhash_threshold: int = 3,
    minhash_threshold: float = 0.75,
) -> DeduplicationResult:
    """Check if document matches any previously indexed document via SimHash or MinHash."""
    # Check SimHash Hamming distance
    for existing_id, existing_hash in simhash_index.items():
        dist = simhash_hamming_distance(simhash_val, existing_hash)
        if dist <= simhash_threshold:
            similarity = round(1.0 - (dist / 64.0), 4)
            return DeduplicationResult(
                is_duplicate=True,
                matched_id=existing_id,
                method="simhash",
                similarity_score=similarity,
            )

    # Check MinHash Jaccard similarity
    for existing_id, existing_sig in minhash_index.items():
        jaccard = minhash_jaccard_similarity(minhash_sig, existing_sig)
        if jaccard >= minhash_threshold:
            return DeduplicationResult(
                is_duplicate=True,
                matched_id=existing_id,
                method="minhash",
                similarity_score=round(jaccard, 4),
            )

    return DeduplicationResult(is_duplicate=False)


def run_pipeline(
    records: Iterable[DocumentRecord],
    config: Optional[PipelineConfig] = None,
    ledger: Optional[AuditLedger] = None,
) -> List[DocumentRecord]:
    """Execute end-to-end streaming intelligence pipeline over document chunks."""
    cfg = config or PipelineConfig()
    audit = ledger if ledger is not None else AuditLedger()

    simhash_index: Dict[str, int] = {}
    minhash_index: Dict[str, List[int]] = {}
    accepted_docs: List[DocumentRecord] = []

    for chunk in chunked_stream(records, chunk_size=cfg.chunk_size):
        for doc in chunk:
            # 1. Quality scoring
            q_score = score_text_quality(
                doc.text, min_words=cfg.min_words, min_score=cfg.min_quality_score
            )
            if not q_score.passed:
                audit.record(
                    AuditEntry(
                        doc_id=doc.id,
                        status="REJECTED_LOW_QUALITY",
                        quality_score=q_score.score,
                        rejection_reason=q_score.reason,
                    )
                )
                continue

            # 2. Semantic deduplication hashes
            doc_simhash = compute_simhash(doc.text)
            doc_minhash = compute_minhash(doc.text)

            dup_check = check_duplicate(
                doc.id,
                doc_simhash,
                doc_minhash,
                simhash_index,
                minhash_index,
                simhash_threshold=cfg.simhash_hamming_threshold,
                minhash_threshold=cfg.minhash_similarity_threshold,
            )

            if dup_check.is_duplicate:
                audit.record(
                    AuditEntry(
                        doc_id=doc.id,
                        status="DEDUPLICATED",
                        simhash=doc_simhash,
                        quality_score=q_score.score,
                        rejection_reason=f"Duplicate of {dup_check.matched_id} via {dup_check.method} ({dup_check.similarity_score})",
                    )
                )
                continue

            # 3. PII & Toxicity Redaction
            redaction = redact_pii_and_toxicity(
                doc.text,
                redact_pii=cfg.redact_pii,
                redact_toxicity=cfg.redact_toxicity,
            )

            # 4. Index accepted document signatures
            simhash_index[doc.id] = doc_simhash
            minhash_index[doc.id] = doc_minhash

            # 5. Ledger record
            audit.record(
                AuditEntry(
                    doc_id=doc.id,
                    status="ACCEPTED",
                    simhash=doc_simhash,
                    quality_score=q_score.score,
                    pii_redactions=redaction.pii_matches,
                    toxicity_redactions=redaction.toxicity_matches,
                )
            )

            cleaned_doc = DocumentRecord(
                id=doc.id,
                text=redaction.cleaned_text,
                source=doc.source,
                metadata={
                    **doc.metadata,
                    "quality_score": str(q_score.score),
                    "pii_redacted": str(redaction.pii_matches),
                    "toxicity_redacted": str(redaction.toxicity_matches),
                },
            )
            accepted_docs.append(cleaned_doc)

    return accepted_docs
