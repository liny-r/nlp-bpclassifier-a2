# BPClassifier — Boilerplate vs. Substantive Sentence Classifier

**NLP for Finance — Spring 2026 | Assignment 2**
**Author:** Yueqi Lin

---

## Overview

A binary sentence classifier for earnings-call transcripts that distinguishes **boilerplate** (scripted, generic, non-material) from **substantive** (financials, guidance, strategy, risks) sentences. Built on 131 transcripts covering 14 tickers (AMD, AVGO, BLK, C, FAST, FDX, GS, INTC, JNJ, JPM, NKE, NVDA, PLTR, WFC).

**Winner:** SetFit (`sentence-transformers/all-MiniLM-L6-v2`, contrastive fine-tuned) — test macro-F1 = **0.9308**, SB recall = **0.9822**  
**Hard constraint:** substantive recall ≥ 0.96 on the held-out test set.

---

## Repository Structure

```
├── Assignment_2_BPClassifier.ipynb       # Main pipeline notebook (end-to-end)
├── Assignment_2_writeup_YueqiLin.pdf     # Project write-up (PDF)
├── Assignment_2_writeup_YueqiLin.md      # Write-up source (Markdown)
├── run_gold_judges.py                    # Reproduce gold labels via Ollama
├── gui.py                                # Streamlit tagging app
├── requirements.txt                      # Python dependencies
├── figures/
│   ├── leaderboard.png                   # Test leaderboard chart
│   ├── confusion_matrix.png              # SetFit test confusion matrix (winner)
│   └── GUI_screenshot_*.png              # GUI screenshots (seen + unseen transcripts)
├── cache/
│   ├── sentence_pool.parquet             # 53,236 unique sentences (≥40 chars)
│   ├── splits.pkl                        # Train/val/test splits (60/20/20, seed=42)
│   ├── embeddings_gold.pkl               # all-MiniLM-L6-v2 embeddings (384-dim)
│   ├── error_analysis_val.csv            # Val-set HistGBM/ensemble error analysis (development artifact)
│   └── gold/
│       ├── gold_labels.parquet           # 2,500-sentence gold set (5-judge MV + human audit)
│       ├── human_review_final.csv        # Round 3 human audit (255 close-call sentences; 11 BP / 244 SB)
│       ├── judge1_qwen3.parquet          # Removed judge (over-flagged boilerplate)
│       ├── judge2_gemma3.parquet         # Removed judge (severe BP bias)
│       ├── judge3_cogito.parquet
│       ├── judge4_qwen314b.parquet
│       ├── judge5_gemma12b.parquet
│       ├── judge6_ministral3.parquet
│       └── judge7_cogito14b.parquet
├── saved_model/
│   ├── setfit_model/                     # SetFit winner checkpoint (~90 MB, included in zip)
│   ├── finbert_finetuned/                # FinBERT runner-up checkpoint
│   │   ├── config.json
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   ├── training_args.bin
│   │   └── model.safetensors             # ~418 MB — included in zip, gitignored on GitHub
│   ├── best_model.pkl                    # HistGBM fallback artifact (~1.6 MB)
│   ├── fasttext_model.bin                # FastText model (~764 MB) — included in zip, gitignored on GitHub
│   └── winner.json                       # {"winner_model": "SetFit ...", "threshold": 0.955}
├── ECT/                                  # Raw earnings-call transcripts (131 files, from ECT.zip)
└── ECT_unseen/                           # Two unseen transcripts for §9 verification
    ├── AAPL_Q2-2026.txt                  # Sourced from Seeking Alpha (not in ECT.zip)
    └── MSFT_Q3-2026.txt                  # Sourced from Seeking Alpha (not in ECT.zip)
```

> **GitHub vs. zip:** `model.safetensors` (~418 MB) and `fasttext_model.bin` (~764 MB) exceed GitHub's 100 MB file-size limit and are excluded from this repository. The SetFit winner checkpoint (`saved_model/setfit_model/`, ~90 MB) is included in both. The submission zip includes all model weights — if you are working from the zip, skip to **Quick start** below.

---

## Quick Start (from submission zip)

Model weights are included in the zip. Just install dependencies and run:

```bash
pip install -r requirements.txt
streamlit run gui.py
```

To re-run the full pipeline (retrains all classifiers including SetFit and FinBERT):

```bash
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=7200 Assignment_2_BPClassifier.ipynb
```

---

## Setup from Git Clone

The large model weights are not in the repository. You must run the notebook to regenerate them before the GUI will work.

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Add the ECT corpus** (provided separately as `ECT.zip`):
```bash
unzip ECT.zip -d ECT/
```

**3. (Optional) Reproduce gold labels** — skip if using the cached labels in `cache/gold/`. Requires Ollama:
```bash
ollama pull cogito:8b && ollama pull qwen3:14b && ollama pull gemma3:12b \
    && ollama pull ministral-3:8b && ollama pull cogito:14b
python run_gold_judges.py --smoke   # connectivity check
python run_gold_judges.py           # full run (~60 min)
```

**4. Run the notebook** (trains all classifiers, saves SetFit and FinBERT weights, writes `winner.json`):
```bash
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=7200 Assignment_2_BPClassifier.ipynb
```
SetFit contrastive fine-tuning takes ~5 min on CPU. FinBERT fine-tuning takes ~15 min on GPU or ~40 min on CPU. All steps are cached — re-runs skip completed work.

**5. Start the GUI:**
```bash
streamlit run gui.py
```

---

## Gold Labeling Methodology

2,500 sentences sampled from the pool (stratified by `speaker_type`) are labeled by a **5-judge LLM majority vote (≥ 3/5)**, followed by a **human audit** of 255 close-call (3–2 split) sentences.

**Active judges:**

| Judge | Model | BP% |
|-------|-------|-----|
| j3 | cogito:8b | 8.2% |
| j4 | qwen3:14b | 11.2% |
| j5 | gemma3:12b | 19.4% |
| j6 | ministral-3:8b | 16.5% |
| j7 | cogito:14b | 23.6% |

**Removed judges:**

| Judge | Model | BP% | Reason |
|-------|-------|-----|--------|
| ~~j1~~ | ~~qwen3:8b~~ | ~~29.3%~~ | Over-flagged boilerplate; manual audit disagreed systematically |
| ~~j2~~ | ~~gemma3:4b~~ | ~~47.5%~~ | Severe BP bias; overrode majority in 746/2,500 sentences |

**Final gold set:** 2,500 sentences | BP = 257 (10.3%) | SB = 2,243 (89.7%)

---

## Final Test Set Results

| Rank | Model | Macro-F1 | BP F1 | SB Recall | Meets Floor |
|------|-------|----------|-------|-----------|-------------|
| 1 | **SetFit** | **0.9308** | 0.8762 | 0.9822 | ✓ |
| 2 | FinBERT-FT | 0.9228 | 0.8624 | 0.9755 | ✓ |
| 3 | Ensemble (mean-prob) | 0.9047 | 0.8283 | 0.9844 | ✓ |
| 4 | Ensemble (rank-avg) | 0.8518 | 0.7312 | 0.9822 | ✓ |
| 5 | HistGBM | 0.8313 | 0.6947 | 0.9755 | ✓ |
| 6 | LogReg | 0.8046 | 0.6444 | 0.9777 | ✓ |
| 7 | FastText | 0.7436 | 0.5333 | 0.9666 | ✓ |
| 8 | Rules+Regex | 0.6639 | 0.4098 | 0.8976 | **✗** |

*Winner: SetFit (contrastive fine-tuned). Checkpoint at `saved_model/setfit_model/`, threshold = 0.955 (recorded in `saved_model/winner.json`).*

---

## Data

- **131 transcripts**, 14 tickers, 2022–2025
- **53,236 unique sentences** after deduplication (minimum 40 characters)
- **2,500-sentence gold sample** stratified by speaker type (analyst / executive / IR / operator)
- **Splits:** train = 1,500 / val = 500 / test = 500 (seed = 42, stratified by label)
- **ECT corpus:** provided as `ECT.zip` (not in this repository)
- **Unseen transcripts:** `ECT_unseen/` — AAPL Q2-2026 and MSFT Q3-2026 sourced from Seeking Alpha
