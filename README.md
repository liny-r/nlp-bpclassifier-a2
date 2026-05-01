# BPClassifier — Boilerplate vs. Substantive Sentence Classifier

**NLP for Finance — Spring 2026 | Assignment 2**
**Author:** Yueqi Lin

---

## Overview

A binary sentence classifier for earnings-call transcripts that distinguishes **boilerplate** (scripted, generic, non-material) from **substantive** (financials, guidance, strategy, risks) sentences. Built on 131 transcripts covering 15 tickers (AMD, AVGO, BLK, C, FAST, FDX, GS, INTC, JNJ, JPM, NKE, NVDA, PLTR, WFC).

**Hard constraint:** substantive recall ≥ 0.96 on the held-out test set.

---

## Repository Structure

```
├── Assignment_2_BPClassifier.ipynb   # Main pipeline notebook (§1–§9)
├── report.md                         # Full write-up
├── run_gold_judges.py                # Reproduce gold labels via Ollama
├── gui.py                            # Streamlit tagging app
├── figures/
│   ├── leaderboard.png                   # Test leaderboard chart
│   └── confusion_matrix.png              # HistGBM test confusion matrix
├── cache/
│   ├── sentence_pool.parquet         # 53,236 unique sentences (≥40 chars)
│   ├── splits.pkl                    # Train/val/test splits (60/20/20, seed=42)
│   ├── embeddings_gold.pkl           # all-MiniLM-L6-v2 embeddings (384-dim)
│   ├── error_analysis_val.csv        # Misclassification examples
│   └── gold/
│       ├── gold_labels.parquet       # 2,500-sentence gold set (5-judge MV + human round 3)
│       ├── human_review_v3.csv       # Round 3 human audit (255 labels — only round used)
│       ├── human_review.csv          # Round 1 audit (archived — contained errors, not used)
│       ├── human_review_round2.csv   # Round 2 audit (archived — contained errors, not used)
│       ├── judge3_cogito.parquet
│       ├── judge4_qwen314b.parquet
│       ├── judge5_gemma12b.parquet
│       ├── judge6_ministral3.parquet
│       ├── judge7_cogito14b.parquet
│       ├── judge1_qwen3.parquet      # Archived — removed judge (over-flagged BP)
│       └── judge2_gemma3.parquet     # Archived — removed judge (severe BP bias)
├── saved_model/
│   └── best_model.pkl                # HistGBM + threshold=0.810 (deployment artifact)
└── ECT/                              # Raw earnings-call transcripts (131 files)
```

---

## Gold Labeling Methodology

2,500 sentences sampled from the pool (stratified by `speaker_type`) are labeled by a **5-judge LLM majority vote (≥ 3/5)**, followed by a **human audit (round 3)** of close-call sentences.

**Active judges:**

| Judge | Model | BP% |
|-------|-------|-----|
| j3 | cogito:8b | 8.2% |
| j4 | qwen3:14b | 11.2% |
| j5 | gemma3:12b | 19.4% |
| j6 | ministral-3:8b | 16.5% |
| j7 | cogito:14b | 23.6% |

**Removed judges (systematic disagreement with human ground truth):**

| Judge | Model | BP% | Reason |
|-------|-------|-----|--------|
| ~~j1~~ | ~~qwen3:8b~~ | ~~29.3%~~ | Over-flagged boilerplate; manual audit disagreed |
| ~~j2~~ | ~~gemma3:4b~~ | ~~47.5%~~ | Severe BP bias; overrode majority in 746/2,500 sentences |

**Final gold set:** 2,500 sentences | BP = 257 (10.3%) | SB = 2,243 (89.7%)

---

## Pipeline

| Section | Description |
|---------|-------------|
| §1 | Environment setup, imports, paths |
| §2 | Sentence extraction from 131 transcripts → `sentence_pool.parquet` |
| §3 | Gold labeling: 5 LLM judges (j3–j7) + human audit (round 3) → `gold_labels.parquet` |
| §4 | Stratified train/val/test split (60/20/20, seed=42) |
| §5 | Feature engineering: 384-dim embeddings + 25 regex flags = 409-dim features |
| §6 | Classifier zoo: Rules, LogReg, HistGBM, FastText, FinBERT, SetFit + 2 ensembles |
| §7 | OOF threshold tuning (TUNE_FLOOR = 0.97; safety margin above 0.96 constraint) |
| §8 | Ensemble + final test evaluation + leaderboard |
| §9 | Error analysis |

---

## Final Test Set Results

| Rank | Model | Macro-F1 | BP F1 | SB Recall | Meets Floor |
|------|-------|----------|-------|-----------|-------------|
| 1 | **FinBERT-FT** | **0.923** | 0.862 | 0.976 | ✓ |
| 2 | Ensemble(mean-prob) | 0.889 | 0.800 | 0.980 | ✓ |
| 3 | SetFit | 0.846 | 0.719 | 0.987 | ✓ |
| 4 | HistGBM | 0.831 | 0.695 | 0.976 | ✓ |
| 5 | Ensemble(rank-avg) | 0.828 | 0.688 | 0.978 | ✓ |
| 6 | LogReg | 0.813 | 0.659 | 0.978 | ✓ |
| 7 | FastText | 0.715 | 0.475 | 0.978 | ✓ |
| 8 | Rules+Regex | 0.664 | 0.410 | 0.898 | **✗** |

*Deployed artifact: `best_model.pkl` = HistGBM retrained on train+val, threshold=0.810.*

---

## Setup

### Requirements

```bash
pip install pandas numpy scikit-learn sentence-transformers tqdm \
            streamlit fasttext-wheel transformers accelerate \
            setfit pyarrow datasets nltk
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

Open `Assignment_2_BPClassifier.ipynb` in Jupyter and run cells top-to-bottom. All expensive steps (sentence extraction, embeddings, FinBERT weights, judge passes) are cached — re-runs skip completed steps automatically.

### Run the GUI

```bash
/Users/yueqilin/anaconda3/bin/python -m streamlit run gui.py
```

Select a transcript from the ECT library tab, upload a `.txt` file, or paste text directly. Boilerplate sentences are highlighted in red with confidence scores.

---

## Data

- **131 transcripts**, 15 tickers, 2022–2025
- **53,236 unique sentences** after deduplication (minimum 40 characters)
- **2,500-sentence gold sample** stratified by speaker type (analyst / executive / IR / operator)
- **Splits:** train=1,500 / val=500 / test=500 (seed=42, stratified by label)
