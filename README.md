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
│       ├── gold_labels.parquet       # 2,500-sentence gold set (5-judge MV)
│       ├── judge1_qwen3.parquet      # Individual judge caches
│       ├── judge2_gemma3.parquet
│       ├── judge3_cogito.parquet
│       ├── judge4_qwen314b.parquet
│       └── judge5_gemma12b.parquet
├── saved_model/                      # Saved best model (best_model.pkl)
└── ECT/                              # Raw earnings-call transcripts (131 files)
```

---

## Gold Labeling Methodology

Gold labels were generated via **5-LLM majority vote** on a stratified sample of 2,500 sentences:

| Judge | Model | Source | BP% | Failures |
|---|---|---|---|---|
| j1 | qwen3:8b | Ollama (local) | 29.3% | 0% |
| j2 | gemma3:4b | Ollama (local) | 47.5% | 0% |
| j3 | cogito:latest | Ollama (local) | 8.2% | 0% |
| j4 | qwen3:14b | Ollama (local) | 11.2% | 0% |
| j5 | gemma3:12b | Ollama (local) | 19.4% | 0% |

- All 5 judges responded on all 2,500 sentences (0% failure rate)
- Final label = majority vote (≥3 of 5 agree)
- Unanimous agreement (5-0): 1,408 sentences (56.3%)
- Close calls (2-3 split): 481 sentences → hand-audited

**Label prompt** enforces strict single-word output (`boilerplate` / `substantive`) with edge-case anchors (analyst intros → boilerplate, safe-harbor → boilerplate, sentences with dollar amounts + context → substantive).

**Final gold set:** 2,500 sentences | BP=458 (18.3%) | SB=2,042 (81.7%)

---

## Pipeline

The notebook (`Assignment_2_BPClassifier.ipynb`) is organized in sections:

| Section | Description |
|---|---|
| §1 | Environment setup, imports, paths |
| §2 | Sentence extraction from 131 transcripts → `sentence_pool.parquet` |
| §3 | Gold labeling: 5 LLM judges + majority vote → `gold_labels.parquet` |
| §4 | Stratified train/val/test split (60/20/20) |
| §5 | Feature engineering: 384-dim embeddings + 25 regex flags = 409 features |
| §6 | Classifier zoo: Rules, LogReg, HistGBM, FastText, FinBERT, SetFit |
| §7 | OOF threshold tuning (recall floor ≥ 0.96) |
| §8 | Ensemble + final test evaluation + save best model |
| §9 | Error analysis |

---

## Classifier Results (Validation Set)

| Model | macro-F1 | SB Recall | Meets Floor |
|---|---|---|---|
| 1 — Rules+Regex | 0.566 | 0.890 | ✗ |
| 2 — LogReg (emb+reg) | 0.655 | 0.955 | ✗ |
| 3 — HistGBM (emb+reg) | 0.695 | 0.960 | ✓ |
| 4 — FastText | TBD | | |
| 5 — FinBERT-FT | TBD | | |
| 6 — SetFit | TBD | | |
| 7 — Ensemble | TBD | | |

---

## Setup

### Requirements

```bash
pip install pandas numpy scikit-learn sentence-transformers tqdm \
            streamlit fasttext-wheel groq google-genai transformers \
            setfit pyarrow
```

### Ollama models (for gold labeling only)

```bash
ollama pull qwen3:8b
ollama pull gemma3:4b
ollama pull cogito:latest
ollama pull qwen3:14b
ollama pull gemma3:12b
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
