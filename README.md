# BPClassifier — Boilerplate vs Substantive Sentence Classifier

**NLP for Finance — Spring 2026 | Assignment 2**
**Author:** Yueqi Lin

---

## Overview

A binary sentence classifier for earnings-call transcripts that distinguishes **boilerplate** (scripted, generic, non-material) from **substantive** (financials, guidance, strategy, risks) sentences. Built on 131 transcripts covering 15 tickers (AMD, AVGO, BLK, C, FAST, FDX, GS, INTC, JNJ, JPM, NKE, NVDA, PLTR, WFC).

**Hard constraint:** substantive recall ≥ 0.96 on the held-out test set.

---

## Repository Structure

```
├── Assignment_2_BPClassifier.ipynb   # Main pipeline notebook
├── run_gold_judges.py                # Standalone script to reproduce gold labels
├── gui.py                            # Streamlit inline tagging app
├── cache/
│   ├── sentence_pool.parquet         # 53,236 unique sentences (≥40 chars)
│   ├── splits.pkl                    # Train/val/test splits (60/20/20, seed=42)
│   ├── embeddings_gold.pkl           # all-MiniLM-L6-v2 embeddings (384-dim)
│   └── gold/
│       ├── gold_labels.parquet       # 2,500-sentence gold set (7-judge MV + human)
│       ├── human_review.csv          # Round 1 human audit (299 labels)
│       ├── human_review_round2.csv   # Round 2 human audit (216 sentences)
│       ├── judge1_qwen3.parquet
│       ├── judge2_gemma3.parquet
│       ├── judge3_cogito.parquet
│       ├── judge4_qwen314b.parquet
│       ├── judge5_gemma12b.parquet
│       ├── judge6_ministral3.parquet
│       └── judge7_cogito14b.parquet
├── saved_model/                      # Saved best model (best_model.pkl)
└── ECT/                              # Raw earnings-call transcripts (131 files)
```

---

## Gold Labeling Methodology

Gold labels were generated via **5-LLM majority vote (≥ 3/5)** on a stratified sample of 2,500 sentences, followed by a **2-round human audit** of close-call sentences:

| Judge | Model | BP% | Failures |
|---|---|---|---|
| j3 | cogito:8b | 8.2% | 0% |
| j4 | qwen3:14b | 11.2% | 0% |
| j5 | gemma3:12b | 19.4% | 0% |
| j6 | ministral-3:8b | 16.5% | 0% |
| j7 | cogito:14b | 23.6% | 0% |

**Removed judges (manual review — systematic disagreement with ground truth):**

| Judge | Model | BP% | Reason |
|---|---|---|---|
| ~~j1~~ | ~~qwen3:8b~~ | ~~29.3%~~ | Over-flagged boilerplate; manual audit disagreed |
| ~~j2~~ | ~~gemma3:4b~~ | ~~47.5%~~ | Severe BP bias; overridden 746/2,500 times by majority |

- All 5 active judges responded on all 2,500 sentences (0% failure rate)
- Final LLM label = majority vote (≥ 3 of 5 agree)
- **Human audit round 1:** 299 close-call sentences reviewed; human label overrides LLM vote
- **Human audit round 2:** 216 additional close calls (`human_review_round2.csv`, in progress)
- Unanimous agreement (5-0): 1,921 sentences (76.8%)

**Label prompt** enforces strict single-word output (`boilerplate` / `substantive`) with edge-case anchors (analyst intros → boilerplate, safe-harbor → boilerplate, sentences with dollar amounts + context → substantive).

**Final gold set:** 2,500 sentences | BP = 334 (13.4%) | SB = 2,166 (86.6%)

---

## Pipeline

The notebook (`Assignment_2_BPClassifier.ipynb`) is organized in sections:

| Section | Description |
|---|---|
| §1 | Environment setup, imports, paths |
| §2 | Sentence extraction from 131 transcripts → `sentence_pool.parquet` |
| §3 | Gold labeling: 5 LLM judges (j3–j7) + 2-round human audit → `gold_labels.parquet` |
| §4 | Stratified train/val/test split (60/20/20) |
| §5 | Feature engineering: 384-dim embeddings + 25 regex flags = 409 features |
| §6 | Classifier zoo: Rules, LogReg, HistGBM, FastText, FinBERT, SetFit |
| §7 | OOF threshold tuning (recall floor ≥ 0.96) |
| §8 | Ensemble + final test evaluation + save best model |
| §9 | Error analysis |

---

## Classifier Results (Validation Set)

| Model | Macro-F1 | BP F1 | SB Recall | Meets Floor |
|---|---|---|---|---|
| 1 — Rules+Regex | 0.600 | 0.302 | 0.875 | ✗ |
| 2 — LogReg (emb+regex) | 0.747 | 0.539 | 0.969 | ✓ |
| 3 — HistGBM (emb+regex) | 0.824 | 0.681 | 0.976 | ✓ |
| 4 — FastText | 0.731 | 0.506 | 0.976 | ✓ |
| 5 — FinBERT-FT | 0.847 | 0.725 | 0.969 | ✓ |
| 6 — SetFit (MiniLM) | **0.853** | **0.735** | 0.976 | ✓ |
| 7a — Ensemble(mean-prob) | 0.844 | 0.717 | 0.982 | ✓ |
| 7b — Ensemble(rank-avg) | 0.818 | 0.673 | 0.964 | ✓ |

*Gold: BP=256 (10.2%) / SB=2244 (89.8%). SetFit best on val; Ensemble second.*

## Final Test Set Results

| Model | Macro-F1 | BP F1 | SB Recall | Meets Floor |
|---|---|---|---|---|
| HistGBM (t=0.815) | 0.764 | 0.571 | 0.969 | ✓ |
| **Ensemble-mean (t=0.670)** | **0.816** | **0.667** | **0.976** | **✓** |

*Saved artifact: `best_model.pkl` = HistGBM retrained on train+val, threshold=0.815.*

---

## Setup

### Requirements

```bash
pip install pandas numpy scikit-learn sentence-transformers tqdm \
            streamlit fasttext-wheel transformers accelerate \
            setfit pyarrow datasets
```

### Ollama models (for gold labeling only)

```bash
ollama pull cogito:8b
ollama pull qwen3:14b
ollama pull gemma3:12b
ollama pull ministral-3:8b
ollama pull cogito:14b
```

### Reproduce gold labels

```bash
python run_gold_judges.py --smoke   # connectivity test
python run_gold_judges.py           # full 2,500-sentence run (~60 min)
```

### Run the notebook

Open `Assignment_2_BPClassifier.ipynb` in Jupyter and run cells top-to-bottom. Most expensive steps are cached — re-runs skip sentence extraction, embeddings, and completed judge passes automatically.

### Run the GUI

```bash
streamlit run gui.py
```

Requires `saved_model/best_model.pkl` (saved after §8 completes). Upload or paste any earnings-call transcript; boilerplate sentences are highlighted in red with confidence scores.

---

## Data

- **131 transcripts**, 15 tickers, multiple quarters (2022–2024)
- **53,236 unique sentences** after deduplication (minimum 40 characters)
- **2,500-sentence gold sample** stratified by speaker type (analyst / executive / IR / operator)
- **Splits:** train=1,500 / val=500 / test=500 (seed=42, stratified)

Raw transcripts are in `ECT/`. The `cache/` directory stores all intermediate artifacts so expensive steps only run once.
