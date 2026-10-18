import argparse
import random
import time

try:
    from .models import DocumentRecord, PipelineConfig
    from .engine import AuditLedger, run_pipeline
except ImportError:
    from models import DocumentRecord, PipelineConfig
    from engine import AuditLedger, run_pipeline

_BASE_SNIPPETS = [
    "Machine learning models require robust preprocessing pipelines to sanitize unstructured web crawls.",
    "Attention mechanisms calculate dynamic context embeddings across sequenced token representations.",
    "Distributed database systems partition storage across independent consensus clusters using Raft protocol.",
    "Gradient descent optimization iteratively updates weights using backward error propagation passes.",
    "Zero-shot reasoning benchmarks evaluate emergent cognitive heuristics across diverse foundation models.",
]

_PII_TEMPLATES = [
    "Inquiries can be directed to support_{i}@dataflow.org or reach our on-call at (555) 234-{i:04d}.",
    "Credential leak detected: private key sk-livedata{i:05d}xyz9876543210 allowed access to cluster.",
    "Customer billing card 4111-2222-3333-{i:04d} authorized cloud training instance allocation.",
    "Audit IP address 192.168.1.{ip_suffix} reported anomalous high rate scraping activity.",
]

_TOXIC_TEMPLATES = [
    "Forum discussion devolved into hate speech and violent attacks against moderators.",
    "Unmoderated comment: you are an idiot and should go die in a fire.",
    "Harassment incident recorded with abusive slur and insults directed at maintainers.",
]

_LOW_QUALITY_TEMPLATES = [
    "@@@ ### $$$ %%% click here now http://...",
    "buy cheap replica watches now now now",
    "short",
    "???! !!! @@@",
]


def generate_benchmark_corpus(num_records: int = 1000) -> list[DocumentRecord]:
    """Generate a realistic synthetic corpus containing diverse quality, duplicates, and PII."""
    rng = random.Random(42)
    corpus: list[DocumentRecord] = []

    for i in range(num_records):
        doc_type = rng.random()
        if doc_type < 0.50:
            # High-quality clean record
            base = rng.choice(_BASE_SNIPPETS)
            text = f"{base} Supplementary section {i}: Model architectures exhibit scalable performance improvements."
        elif doc_type < 0.70:
            # Near duplicate or duplicate of an earlier record
            target_idx = max(0, i - rng.randint(1, 5))
            if corpus:
                prev_text = corpus[target_idx].text
                # Minor modification (near-duplicate)
                text = prev_text.rstrip(".") + f" (revision {i % 3})."
            else:
                text = _BASE_SNIPPETS[0]
        elif doc_type < 0.85:
            # Record containing PII
            pii_snippet = rng.choice(_PII_TEMPLATES).format(i=i, ip_suffix=i % 250)
            text = f"Production cluster telemetry log {i}. {pii_snippet} System diagnostics operational."
        elif doc_type < 0.95:
            # Low quality / spam / gibberish
            text = rng.choice(_LOW_QUALITY_TEMPLATES)
        else:
            # Toxicity / abusive content
            text = rng.choice(_TOXIC_TEMPLATES)

        corpus.append(DocumentRecord(id=f"doc_{i:06d}", text=text, source="benchmark"))

    return corpus


def run_benchmark(sample_size: int = 1000, chunk_size: int = 100) -> None:
    """Run benchmark measuring throughput and filtering efficacy."""
    print("=" * 70)
    print("TRAINING DATA INTELLIGENCE PLATFORM: BENCHMARK SUITE")
    print("=" * 70)
    print(f"Generating synthetic streaming corpus ({sample_size} records)...")
    corpus = generate_benchmark_corpus(sample_size)
    print(f"Generated {len(corpus)} records. Starting streaming pipeline...")

    config = PipelineConfig(
        chunk_size=chunk_size,
        min_quality_score=0.4,
        min_words=5,
        simhash_hamming_threshold=3,
        minhash_similarity_threshold=0.75,
        redact_pii=True,
        redact_toxicity=True,
    )
    ledger = AuditLedger()

    start_time = time.perf_counter()
    accepted = run_pipeline(corpus, config=config, ledger=ledger)
    total_time = time.perf_counter() - start_time

    summary = ledger.summary()
    throughput = len(corpus) / max(0.0001, total_time)

    print("\n" + "-" * 70)
    print("BENCHMARK EXECUTION RESULTS")
    print("-" * 70)
    print(f"Total Ingested Documents : {summary.total_ingested:>8}")
    print(f"Accepted Clean Documents : {summary.passed:>8} ({(summary.passed / summary.total_ingested) * 100:>5.1f}%)")
    print(f"Deduplicated (Semantic)  : {summary.deduplicated:>8} ({(summary.deduplicated / summary.total_ingested) * 100:>5.1f}%)")
    print(f"Rejected (Low Quality)   : {summary.low_quality_rejected:>8} ({(summary.low_quality_rejected / summary.total_ingested) * 100:>5.1f}%)")
    print(f"PII Entities Redacted    : {summary.pii_redacted_count:>8}")
    print(f"Toxic Entities Redacted  : {summary.toxicity_redacted_count:>8}")
    print(f"Total Processing Time    : {total_time:>8.3f} seconds")
    print(f"Throughput               : {throughput:>8.1f} docs/sec")
    print("-" * 70)

    print("\nSAMPLE AUDIT LEDGER ENTRIES:")
    for entry in ledger.entries[:6]:
        reason_str = f" | {entry.rejection_reason}" if entry.rejection_reason else ""
        print(f"  [{entry.status:<20}] Doc {entry.doc_id} (Quality: {entry.quality_score:.2f}, PII: {entry.pii_redactions}, Tox: {entry.toxicity_redactions}){reason_str}")

    print("\nSAMPLE SANITIZED DOCUMENT PREVIEW:")
    for doc in accepted[:2]:
        preview = doc.text if len(doc.text) < 100 else doc.text[:97] + "..."
        print(f"  ID: {doc.id}")
        print(f"  Text: {preview}")
        print(f"  Metadata: {doc.metadata}\n")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Training Data Intelligence Platform CLI")
    parser.add_argument("--benchmark", action="store_true", default=True, help="Execute synthetic benchmark run")
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of records to benchmark (default: 1000)")
    parser.add_argument("--chunk-size", type=int, default=100, help="Streaming chunk size (default: 100)")
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark(sample_size=args.sample_size, chunk_size=args.chunk_size)


if __name__ == "__main__":
    main()
