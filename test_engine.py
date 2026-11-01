import pytest

try:
    from projects.training_data_intelligence_platform.models import (
        DocumentRecord,
        PipelineConfig,
    )
    from projects.training_data_intelligence_platform.engine import (
        AuditLedger,
        compute_simhash,
        simhash_hamming_distance,
        compute_minhash,
        minhash_jaccard_similarity,
        score_text_quality,
        redact_pii_and_toxicity,
        chunked_stream,
        run_pipeline,
    )
except ImportError:
    from models import (
        DocumentRecord,
        PipelineConfig,
    )
    from engine import (
        AuditLedger,
        compute_simhash,
        simhash_hamming_distance,
        compute_minhash,
        minhash_jaccard_similarity,
        score_text_quality,
        redact_pii_and_toxicity,
        chunked_stream,
        run_pipeline,
    )


def test_simhash_identical_and_near_duplicates():
    text1 = "Deep neural network architectures require high quality training datasets for model pre-training."
    text2 = "Deep neural network architectures require high quality training datasets for model pre-training!"
    text3 = "Quantum computing uses qubits and entanglement to solve discrete optimization problems."

    h1 = compute_simhash(text1)
    h2 = compute_simhash(text2)
    h3 = compute_simhash(text3)

    # Identical texts yield identical hashes (distance 0)
    assert simhash_hamming_distance(h1, h1) == 0

    # Near duplicates yield small hamming distance (<= 4)
    dist_near = simhash_hamming_distance(h1, h2)
    assert dist_near <= 4

    # Completely different text yields significantly larger hamming distance
    dist_diff = simhash_hamming_distance(h1, h3)
    assert dist_diff > 10


def test_minhash_jaccard_similarity():
    text1 = "Transformers revolutionize natural language processing and computer vision domains worldwide."
    text2 = "Transformers revolutionize natural language processing and computer vision domains globally."
    text3 = "Baking sourdough bread requires flour, water, salt, wild yeast culture, and patience."

    sig1 = compute_minhash(text1)
    sig2 = compute_minhash(text2)
    sig3 = compute_minhash(text3)

    # Identical
    assert minhash_jaccard_similarity(sig1, sig1) == 1.0

    # Near duplicates have high Jaccard similarity
    sim_near = minhash_jaccard_similarity(sig1, sig2)
    assert sim_near >= 0.70

    # Disjoint text has near zero similarity
    sim_diff = minhash_jaccard_similarity(sig1, sig3)
    assert sim_diff < 0.20


def test_quality_scoring_rejection_rules():
    # Empty text
    empty_res = score_text_quality("")
    assert not empty_res.passed
    assert empty_res.score == 0.0

    # Too short text (< 5 words)
    short_res = score_text_quality("Hello world", min_words=5)
    assert not short_res.passed
    assert "below minimum threshold" in short_res.reason

    # Gibberish / low alphanumeric density
    symbols_res = score_text_quality("@@@ ### $$$ %%% ^^^ &&& *** ((( ))) ___ +++ === ~~~", min_words=5)
    assert not symbols_res.passed

    # Repetition spam
    spam = "repeat " * 30
    spam_res = score_text_quality(spam, min_words=5)
    assert not spam_res.passed
    assert "repetition" in spam_res.reason.lower()

    # Valid high quality text
    valid_text = (
        "Supervised fine-tuning aligns foundation models with task specific guidelines "
        "and human feedback preferences through contrastive loss gradients."
    )
    valid_res = score_text_quality(valid_text, min_words=5)
    assert valid_res.passed
    assert valid_res.score > 0.65


def test_pii_and_toxicity_redaction():
    text = (
        "Contact developer alice@example.com or call +1-555-839-2918. "
        "User SSN is 123-45-6789 and API secret is sk-99887766554433221100abcdef123. "
        "Do not be an asshole in the forum."
    )
    result = redact_pii_and_toxicity(text, redact_pii=True, redact_toxicity=True)

    assert "[REDACTED_EMAIL]" in result.cleaned_text
    assert "alice@example.com" not in result.cleaned_text
    assert "[REDACTED_PHONE]" in result.cleaned_text
    assert "123-45-6789" not in result.cleaned_text
    assert "[REDACTED_SSN]" in result.cleaned_text
    assert "[REDACTED_API_KEY]" in result.cleaned_text
    assert "[REDACTED_TOXIC]" in result.cleaned_text
    assert "asshole" not in result.cleaned_text.lower()
    assert result.pii_matches >= 4
    assert result.toxicity_matches >= 1


def test_chunked_streaming():
    docs = [DocumentRecord(id=f"doc_{i}", text=f"Sample text document number {i} for streaming test.") for i in range(12)]
    chunks = list(chunked_stream(docs, chunk_size=5))
    
    assert len(chunks) == 3
    assert len(chunks[0]) == 5
    assert len(chunks[1]) == 5
    assert len(chunks[2]) == 2


def test_pipeline_deduplication_and_audit_ledger():
    ledger = AuditLedger()
    config = PipelineConfig(
        min_quality_score=0.4,
        min_words=5,
        simhash_hamming_threshold=3,
        minhash_similarity_threshold=0.75,
        chunk_size=10,
    )

    corpus = [
        # Original document 1
        DocumentRecord(
            id="doc_1",
            text="Reinforcement learning from human feedback trains language models to produce safe outputs.",
        ),
        # Near duplicate of doc 1 (should be deduplicated)
        DocumentRecord(
            id="doc_2",
            text="Reinforcement learning from human feedback trains language models to produce safer outputs.",
        ),
        # Low quality document (should be rejected)
        DocumentRecord(
            id="doc_3",
            text="bad #$@#",
        ),
        # Valid document with PII (should be accepted with redactions)
        DocumentRecord(
            id="doc_4",
            text="Customer care engineer can be reached at support@cloudplatform.org for urgent tickets.",
        ),
    ]

    accepted = run_pipeline(corpus, config=config, ledger=ledger)
    summary = ledger.summary()

    # Valid accepted docs should be doc_1 and doc_4
    assert len(accepted) == 2
    assert accepted[0].id == "doc_1"
    assert accepted[1].id == "doc_4"
    assert "[REDACTED_EMAIL]" in accepted[1].text

    # Verify audit summary
    assert summary.total_ingested == 4
    assert summary.passed == 2
    assert summary.deduplicated == 1
    assert summary.low_quality_rejected == 1
    assert summary.pii_redacted_count >= 1


def test_edge_cases_empty_and_minimal():
    # Empty stream
    empty_ledger = AuditLedger()
    accepted = run_pipeline([], ledger=empty_ledger)
    assert len(accepted) == 0
    assert empty_ledger.summary().total_ingested == 0

    # Simhash of empty text
    assert compute_simhash("") == 0
    # Minhash of empty text
    assert compute_minhash("") == [0] * 64


def test_minhash_deduplication_pipeline():
    ledger = AuditLedger()
    config = PipelineConfig(
        simhash_hamming_threshold=1,
        minhash_similarity_threshold=0.75,
    )
    docs = [
        DocumentRecord(
            id="doc_base",
            text="Distributed stream processing engines partition real-time event queues across resilient cluster nodes with high throughput and low end-to-end latency.",
        ),
        DocumentRecord(
            id="doc_variant",
            text="Distributed stream processing platforms partition real-time event queues across resilient cluster nodes with high throughput and low end-to-end latency.",
        ),
    ]
    accepted = run_pipeline(docs, config=config, ledger=ledger)
    assert len(accepted) == 1
    assert accepted[0].id == "doc_base"
    assert ledger.summary().deduplicated == 1


def test_multi_pii_and_metadata_retention():
    ledger = AuditLedger()
    text = (
        "Server admin IP 192.168.1.50 accessed AWS with key AKIAIOSFODNN7EXAMPLE "
        "and reported card 4532-1234-5678-9010 on phone (555) 019-2834."
    )
    doc = DocumentRecord(id="doc_sec", text=text, metadata={"source_env": "production"})
    accepted = run_pipeline([doc], ledger=ledger)

    assert len(accepted) == 1
    cleaned = accepted[0].text
    assert "[REDACTED_IP_ADDRESS]" in cleaned
    assert "[REDACTED_API_KEY]" in cleaned
    assert "[REDACTED_CREDIT_CARD]" in cleaned
    assert "[REDACTED_PHONE]" in cleaned
    assert accepted[0].metadata["source_env"] == "production"
    assert int(accepted[0].metadata["pii_redacted"]) >= 4
