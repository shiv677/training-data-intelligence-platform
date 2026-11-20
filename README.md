# Training Data Intelligence Platform

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2.13-green.svg)](https://docs.pydantic.dev/)
[![Tests Passing](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)](test_engine.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](../../LICENSE)

High-performance AI training data intelligence pipeline with distributed chunked streaming, dual-algorithm semantic deduplication (SimHash & MinHash), automated multi-signal quality scoring, regex/heuristic PII and toxicity redacting, and an immutable audit lineage ledger.

---

## Architecture Overview

```
                      ┌──────────────────────────────────────┐
                      │    Raw Training Document Stream      │
                      │  [Unstructured Web, Logs, Scrapes]   │
                      └──────────────────┬───────────────────┘
                                         │
                                         ▼
                      ┌──────────────────────────────────────┐
                      │      Distributed Chunked Stream      │
                      │  O(1) Memory Constant Ingestion      │
                      └──────────────────┬───────────────────┘
                                         │
                                         ▼
                      ┌──────────────────────────────────────┐
                      │     Automated Quality Scorer         │
                      │ Length · Alnum Ratio · Repetition    │
                      └─────────┬──────────────────┬─────────┘
                                │                  │
                       [Score < Threshold]   [Passes Quality]
                                │                  │
                                ▼                  ▼
                     ┌──────────────────┐ ┌──────────────────────────────────────┐
                     │ REJECTED_LOW_QUAL│ │  Dual-Engine Semantic Deduplication │
                     │ Audit Log Entry  │ │  • 64-bit SimHash (Hamming Dist <= 3)│
                     └──────────────────┘ │  • 64-Perm MinHash (Jaccard >= 0.75) │
                                          └──────┬────────────────────┬──────────┘
                                                 │                    │
                                          [Near-Duplicate]      [Unique Document]
                                                 │                    │
                                                 ▼                    ▼
                                         ┌──────────────┐ ┌──────────────────────┐
                                         │ DEDUPLICATED │ │ PII & Toxicity Mask │
                                         │  Audit Entry │ │ Emails, Keys, Slurs  │
                                         └──────────────┘ └──────────┬───────────┘
                                                                     │
                                                                     ▼
                                                          ┌──────────────────────┐
                                                          │   ACCEPTED RECORD    │
                                                          │ Sanitized + Metadata │
                                                          │ Lineage Audit Ledger │
                                                          └──────────────────────┘
```

---

## Sequence Flow

```
Document Stream ──▶ Chunk Generator ──▶ Quality Evaluator
                                              │
                    ┌─────────────────────────┴────────────────────────┐
                    ▼ (Quality < Threshold)                            ▼ (Valid)
           Log Rejection Reason                              Semantic Hash Generator
                    │                                        [SimHash & MinHash]
                    ▼                                                  │
            Audit Lineage                                              ▼
                                                             Deduplication Index
                                                                       │
                                      ┌────────────────────────────────┴────────────────────────┐
                                      ▼ (Collision / Jaccard Match)                             ▼ (Novel)
                             Log Duplicate Target                                      Regex & Toxicity Masker
                                      │                                                         │
                                      ▼                                                         ▼
                              Audit Lineage                                            Record ACCEPTED in Ledger
                                                                                                │
                                                                                                ▼
                                                                                       Clean Training Token
```

---

## Key Capabilities

- **Distributed Chunked Streaming**: Streams documents in configurable chunk partitions (`chunk_size`), preventing memory saturation over large dataset corpora.
- **Dual Semantic Deduplication**:
  - **SimHash**: Computes 64-bit vector fingerprints with character 4-shingles, matching near duplicates within bitwise Hamming distance $\le 3$.
  - **MinHash**: Uses 64 linear congruential hash permutations across Mersenne prime fields ($2^{31} - 1$) with 3-word shingles to estimate Jaccard similarity and isolate structural duplicates ($\ge 0.75$).
- **Multi-Signal Quality Scoring**:
  - Word count and character thresholds.
  - Alphanumeric symbol density heuristics (catches garbled text, OCR artifacts, symbol spam).
  - Repetition and vocabulary entropy detection (catches boilerplate, repetitive loops, and web scraping noise).
- **Comprehensive Redaction Suite**:
  - Masks emails, phone numbers, SSNs, credit cards, IP addresses, and high-entropy API secrets (`sk-...`, `ghp_...`, `AKIA...`).
  - Flags and replaces toxic phrases and abusive keywords with `[REDACTED_TOXIC]`.
- **Audit Lineage Ledger**:
  - Every document receives an immutable audit entry recording status (`ACCEPTED`, `DEDUPLICATED`, `REJECTED_LOW_QUALITY`), fingerprint, quality score, and redaction counts.
  - Generates full summary analytics and throughput telemetry.

---

## Quickstart

### 1. Requirements

```bash
pip install pydantic pytest
```

### 2. Run Pipeline Benchmark

Execute the driver with synthetic streaming datasets:

```bash
python main.py --sample-size 1000 --chunk-size 100
```

Sample output:
```text
======================================================================
TRAINING DATA INTELLIGENCE PLATFORM: BENCHMARK SUITE
======================================================================
Generating synthetic streaming corpus (1000 records)...
Generated 1000 records. Starting streaming pipeline...

----------------------------------------------------------------------
BENCHMARK EXECUTION RESULTS
----------------------------------------------------------------------
Total Ingested Documents :     1000
Accepted Clean Documents :      248 ( 24.8%)
Deduplicated (Semantic)  :      664 ( 66.4%)
Rejected (Low Quality)   :       88 (  8.8%)
PII Entities Redacted    :      162
Toxic Entities Redacted  :       11
Total Processing Time    :    0.725 seconds
Throughput               :   1379.3 docs/sec
----------------------------------------------------------------------
```

---

## Test Specifications

The test suite in [test_engine.py](file:///c:/Users/4/Music/backups/shivanshdimri-main/projects/training_data_intelligence_platform/test_engine.py) validates the core pipeline components:

1. **`test_simhash_identical_and_near_duplicates`**: Verifies exact zero distance on identical text, small Hamming distance ($\le 4$) on near duplicates, and large divergence on disjoint topics.
2. **`test_minhash_jaccard_similarity`**: Validates signature alignment with 1.0 Jaccard for identical text and $\ge 0.70$ for minor modifications.
3. **`test_quality_scoring_rejection_rules`**: Validates rejections for empty strings, sub-threshold word lengths, symbol-heavy noise, and repetition loops.
4. **`test_pii_and_toxicity_redaction`**: Confirms regex masking of emails, phone numbers, SSNs, API credentials, and toxic rhetoric.
5. **`test_chunked_streaming`**: Ensures exact partitioning across chunk boundaries.
6. **`test_pipeline_deduplication_and_audit_ledger`**: End-to-end integration test verifying document acceptance, near-duplicate pruning, and ledger accounting.
7. **`test_edge_cases_empty_and_minimal`**: Handles zero-length streams, empty inputs, and boundary cases without unhandled exceptions.
8. **`test_minhash_deduplication_pipeline`**: Verifies MinHash duplicate detection inside streaming pipeline executions.
9. **`test_multi_pii_and_metadata_retention`**: Confirms metadata pass-through and multi-token redactions.

Run the test suite:

```bash
pytest test_engine.py -v
```

---

## Author

**Shivansh Dimri**  
Associate AI Engineer — Python Engineering · AI Evaluation · Data Systems · Automation  
Website: [https://shivanshdimri.in](https://shivanshdimri.in)  
GitHub: [@shiv677](https://github.com/shiv677)  
Email: [dimrishiv933@gmail.com](mailto:dimrishiv933@gmail.com)
