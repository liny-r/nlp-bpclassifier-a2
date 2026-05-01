import sys
sys.stdout.reconfigure(line_buffering=True)
import time as _time
_t0_total = _time.time()

def _section(name):
    elapsed = (_time.time() - _t0_total) / 60
    print(f'\n{"="*60}', flush=True)
    print(f'  {name}  [{elapsed:.1f} min elapsed]', flush=True)
    print(f'{"="*60}', flush=True)

# ═══ CELL [02] ═══
# Uncomment to install in a fresh environment
# !pip install -q nltk sentence-transformers scikit-learn fasttext-wheel setfit \
#              transformers datasets accelerate pandas numpy pyarrow tqdm matplotlib plotly streamlit

import os, re, json, pickle, time, warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 160)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path('/Users/yueqilin/Desktop/9796 NLP/Transcript Assignment2')
TRANSCRIPTS = ROOT / 'ECT'
CACHE       = ROOT / 'cache'
GOLD_DIR    = CACHE / 'gold'
EMBED_CACHE = CACHE / 'embeddings.pkl'
MODEL_DIR   = ROOT / 'saved_model'

for d in (CACHE, GOLD_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

txts = sorted(TRANSCRIPTS.glob('*.txt'))
print(f'{len(txts)} transcripts found')

# ═══ CELL [04] ═══
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize

# ── Transcript parser for Assignment 2 pipeline ───────────────────────────────
SECTION_HEADERS = {
    'Presentation Operator Message',
    'Presenter Speech',
    'Question and Answer Operator Message',
    'Question',
    'Answer',
}
HEADER_RE = re.compile(
    r'^(?P<company>.+?),\s*Q(?P<q>\d)\s*(?P<y>\d{4}).*?Earnings Call.*?'
    r'(?P<date>[A-Z][a-z]+ \d{1,2},\s*\d{4})'
)

def _blocks(text):
    lines = text.splitlines()
    i, n = 0, len(lines)
    while i < n:
        line = lines[i].strip()
        if line in SECTION_HEADERS:
            section = line; i += 1
            role = lines[i].strip() if i < n else ''; i += 1
            buf = []
            while i < n and lines[i].strip() not in SECTION_HEADERS:
                buf.append(lines[i]); i += 1
            yield section, role, '\n'.join(buf).strip()
        else:
            i += 1

def _speaker_type(role: str) -> str:
    r = role.lower()
    if 'operator' in r:              return 'operator'
    if 'investor relations' in r or r.startswith('executives') and any(
            x in r for x in ['investor', 'ir ', ' ir', 'relations']): return 'ir'
    if r.startswith('executives'):   return 'executive'
    if r.startswith('analysts'):     return 'analyst'
    if r == 'operator':              return 'operator'
    return 'unknown'

def _parse_meta(path):
    text = path.read_text(errors='ignore')
    stem = path.stem
    ticker, _, quarter = stem.partition('_')
    m = HEADER_RE.match(text.splitlines()[0])
    call_date = None
    if m:
        from datetime import datetime
        try:
            call_date = datetime.strptime(m.group('date').replace('  ',' '), '%b %d, %Y').strftime('%Y-%m-%d')
        except ValueError:
            pass
    return ticker, quarter, call_date, text

# ── Sentence extraction ────────────────────────────────────────────────────────
MIN_CHARS = 40

def extract_sentences(path: Path) -> List[Dict]:
    ticker, quarter, call_date, text = _parse_meta(path)
    rows = []
    in_qa = False
    for section, role, body in _blocks(text):
        if section == 'Question and Answer Operator Message':
            in_qa = True; continue
        if not body.strip():
            continue
        sec_label = 'qa' if in_qa else 'prepared'
        spk = _speaker_type(role)
        for sent in sent_tokenize(body):
            sent = sent.strip()
            if len(sent) >= MIN_CHARS:
                rows.append({
                    'ticker': ticker, 'quarter': quarter, 'call_date': call_date,
                    'section': sec_label, 'speaker_type': spk, 'text': sent,
                })
    return rows

# Cache check — skip re-extraction if sentence_pool already saved
_pool_cache = CACHE / 'sentence_pool.parquet'
if _pool_cache.exists():
    sentence_pool = pd.read_parquet(_pool_cache)
    print(f'Loaded sentence_pool from cache: {len(sentence_pool):,} sentences')
    print(sentence_pool['speaker_type'].value_counts().to_string())
else:
    all_rows = []
    for p in tqdm(txts, desc='extracting sentences'):
        all_rows.extend(extract_sentences(p))

    sentence_pool = pd.DataFrame(all_rows)

    # Drop exact duplicates on text
    before = len(sentence_pool)
    sentence_pool = sentence_pool.drop_duplicates(subset='text').reset_index(drop=True)
    print(f'Extracted {before:,} sentences → {len(sentence_pool):,} after dedup')
    print(sentence_pool['speaker_type'].value_counts().to_string())
    print(sentence_pool['section'].value_counts().to_string())

    # Save full pool
    sentence_pool.to_parquet(CACHE / 'sentence_pool.parquet', index=False)
    print('\nSaved sentence_pool.parquet')

# ═══ CELL [14] ═══
from sklearn.model_selection import train_test_split

gold_final = pd.read_parquet(GOLD_DIR / 'gold_labels.parquet')

X_all = gold_final['text'].values
y_all = gold_final['label'].values

# 60 / 20 / 20
X_tmp, X_test, y_tmp, y_test = train_test_split(
    X_all, y_all, test_size=0.20, stratify=y_all, random_state=RANDOM_SEED)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=RANDOM_SEED)  # 0.25 × 0.80 = 0.20

print(f'train: {len(X_train)}  val: {len(X_val)}  test: {len(X_test)}')
for split_name, y in [('train', y_train), ('val', y_val), ('test', y_test)]:
    bp = (y == 0).sum(); sb = (y == 1).sum()
    print(f'  {split_name}: boilerplate={bp} ({bp/len(y)*100:.1f}%)  substantive={sb} ({sb/len(y)*100:.1f}%)')

# Save splits (indices into gold_final)
splits = {'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
          'y_train': y_train, 'y_val': y_val, 'y_test': y_test}
with open(CACHE / 'splits.pkl', 'wb') as f:
    pickle.dump(splits, f)
print('Saved splits.pkl')

# ═══ CELL [16] ═══
# ── 25 regex feature flags ────────────────────────────────────────────────────

REGEX_FEATURES = [
    # --- Boilerplate signals ---
    ('f_operator_phrase',   r'(?i)(my name is|conference operator|welcome everyone|welcome to nvidia|welcome to the)'),
    ('f_safe_harbor',       r'(?i)(forward[- ]looking|safe[- ]harbor|actual results may differ|risks and uncertainties)'),
    ('f_sec_filing',        r'(?i)(form 10-[kq]|sec filing|securities and exchange|8-k|annual report)'),
    ('f_webcast',           r'(?i)(webcast|replay until|investor relations website|ir website)'),
    ('f_generic_thanks',    r'(?i)\b(thank you|thanks)\b(?!.{0,30}(revenue|guidance|earnings|results|growth))'),
    ('f_question_intro',    r'(?i)(our next question|your line is (open|now)|goes to the line of|you may (begin|proceed|go ahead))'),
    ('f_analyst_firm',      r'(?i)\b(Goldman|Morgan Stanley|JPMorgan|Citi(group)?|UBS|Wells Fargo|Deutsche Bank|Barclays|BofA|Bernstein|Cowen|Jefferies|Piper Sandler|Evercore|Oppenheimer|Mizuho)\b'),
    ('f_call_close',        r'(?i)(no further questions|this concludes|thank you for (your time|participating|joining))'),
    ('f_nongaap',           r'(?i)(non-gaap|reconciliation|gaap (to|and) non-gaap)'),
    ('f_short_affirm',      r'(?i)^(sure\.?|great\.?|okay\.?|yes\.?|absolutely\.?|of course\.?|thank you\.?)\s*$'),
    ('f_operator_instr',    r'(?i)\[operator instructions\]'),
    ('f_turn_over',         r'(?i)(turn (the call|it) over|let me (now )?turn|i.d like to (now )?turn)'),
    # --- Substantive signals ---
    ('f_dollar_amount',     r'\$[\d,\.]+\s*(billion|million|thousand|[BbMmKk])?'),
    ('f_percentage',        r'\b\d+(\.\d+)?\s*%'),
    ('f_revenue_mention',   r'(?i)\b(revenue|net (income|loss|sales)|total (sales|revenue))\b'),
    ('f_margin_mention',    r'(?i)\b(gross margin|operating margin|ebitda margin|profit margin)\b'),
    ('f_eps_mention',       r'(?i)\b(earnings per share|eps|diluted eps|non-gaap eps)\b'),
    ('f_guidance_word',     r'(?i)\b(guidance|outlook|forecast|expect(s|ed)?|anticipate[sd]?|project(s|ed)?|full[- ]year)\b'),
    ('f_raised_lowered',    r'(?i)\b(raised?|lowered?|increased?|decreased?|reaffirmed?)\s+(guidance|outlook|revenue|earnings|estimate)'),
    ('f_yoy_qoq',           r'(?i)\b(year[- ]over[- ]year|sequentially|quarter[- ]over[- ]quarter|yoy|qoq|vs\. prior)\b'),
    ('f_record_quarter',    r'(?i)\b(record (revenue|quarter|sales|profit|high)|all[- ]time (high|record))\b'),
    ('f_product_launch',    r'(?i)\b(launched?|announced?|introduced?|released?|shipped?|ramped?)\s+(new |our |the )?\w+ (platform|product|system|service|model|chip|GPU|CPU)'),
    ('f_customer_mention',  r'(?i)\b(customer(s)?|client(s)?|partner(s)?)\s+(include|such as|like|across|with)'),
    # --- Structural signals ---
    ('f_sentence_short',    None),   # computed below (len < 10 words)
    ('f_has_digits',        r'\b\d+\b'),
]

COMPILED = [(name, re.compile(pat) if pat else None) for name, pat in REGEX_FEATURES]

def build_regex_features(texts):
    rows = []
    for text in texts:
        row = []
        for name, pattern in COMPILED:
            if name == 'f_sentence_short':
                row.append(1 if len(text.split()) < 10 else 0)
            else:
                row.append(1 if pattern.search(text) else 0)
        rows.append(row)
    return np.array(rows, dtype=np.float32)

feat_names = [name for name, _ in REGEX_FEATURES]
print(f'{len(feat_names)} regex features defined')

# Quick sanity
sample_texts = [
    "Revenue of $26 billion was up 18% sequentially.",
    "My name is Regina and I will be your conference operator.",
]
sample_feats = build_regex_features(sample_texts)
for t, feats in zip(sample_texts, sample_feats):
    active = [feat_names[i] for i, v in enumerate(feats) if v > 0]
    print(f'  "{t[:60]}"')
    print(f'    → {active}')

# ═══ CELL [17] ═══
# ── Sentence embeddings (cached) ──────────────────────────────────────────────
# Uses all-MiniLM-L6-v2 (384-dim). Runs on Apple MPS if available.

from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

def get_embeddings(texts, model_name=EMBED_MODEL_NAME, batch_size=256, cache_key='all'):
    """Embed texts, using disk cache keyed by cache_key."""
    cache = CACHE / f'embeddings_{cache_key}.pkl'
    if cache.exists():
        with open(cache, 'rb') as f:
            data = pickle.load(f)
        if data.get('model') == model_name and data.get('n') == len(texts):
            print(f'  Embeddings loaded from cache ({cache_key}): {data["emb"].shape}')
            return data['emb']
    print(f'  Embedding {len(texts)} sentences with {model_name}...')
    model = SentenceTransformer(model_name)
    emb = model.encode(list(texts), batch_size=batch_size, show_progress_bar=True,
                       convert_to_numpy=True, normalize_embeddings=True)
    with open(cache, 'wb') as f:
        pickle.dump({'model': model_name, 'n': len(texts), 'emb': emb}, f)
    print(f'  Saved embeddings cache: {emb.shape}')
    return emb

# Embed all gold sentences at once, then split by index
all_texts = gold_final['text'].values
all_emb = get_embeddings(all_texts, cache_key='gold')

# Recover split indices
with open(CACHE / 'splits.pkl', 'rb') as f:
    splits = pickle.load(f)

# We need to match texts back to indices in gold_final
text_to_idx = {t: i for i, t in enumerate(all_texts)}

def texts_to_emb(texts):
    idxs = [text_to_idx[t] for t in texts]
    return all_emb[idxs]

# Build feature matrices
def build_full_features(texts):
    emb = texts_to_emb(texts)
    reg = build_regex_features(texts)
    return np.hstack([emb, reg])

print('\nBuilding feature matrices...')
F_train = build_full_features(splits['X_train'])
F_val   = build_full_features(splits['X_val'])
F_test  = build_full_features(splits['X_test'])
print(f'Feature dimensions: train={F_train.shape}  val={F_val.shape}  test={F_test.shape}')

y_train, y_val, y_test = splits['y_train'], splits['y_val'], splits['y_test']
X_train, X_val, X_test = splits['X_train'], splits['X_val'], splits['X_test']

# ═══ CELL [19] ═══
# ── Classifier 1 helper: Rules-based classifier ──────────────────────────────
# NOTE: (?i) inline flags stripped; re.IGNORECASE used instead
# (joining with | breaks inline flags in Python 3.11+).

_BP_PATTERNS = [
    r'\b(good\s+(morning|afternoon|evening)|welcome\s+to|thank\s+you\s+for\s+(joining|calling|participating))\b',
    r'\b(my\s+name\s+is|i\s+will\s+be\s+your\s+(conference\s+)?operator)\b',
    r'\b(forward[- ]looking\s+statement|safe[- ]harbor|actual\s+results\s+may\s+differ)\b',
    r'\b(please\s+(press|hold|stand\s+by)|operator\s+instructions)\b',
    r'(\[operator\s+instructions\])',
    r'\b(our\s+next\s+question|your\s+line\s+is\s+(open|now\s+open)|goes\s+to\s+(the\s+line\s+of|[A-Z]))\b',
    r'\b(you\s+may\s+(begin|proceed|go\s+ahead))\b',
    r'\b(no\s+further\s+questions|this\s+concludes\s+today.s\s+(call|conference|question)|thank\s+you\s+for\s+your\s+(time|participation|attending))\b',
    r'\b(webcast\s+(will\s+be\s+)?available|replay\s+until|investor\s+relations\s+website)\b',
    r'\b(non[-\s]gaap|reconciliation|sec\s+filings|form\s+10-[kq]|securities\s+and\s+exchange)\b',
    r'^(sure\.?|great\.?|okay\.?|yes\.?|absolutely\.?|of\s+course\.?)\s*$',
    r'\b(i\s+would\s+like\s+to\s+(introduce|turn\s+the\s+call\s+over|welcome))\b',
    r'\b(from\s+the\s+line\s+of|with\s+(Goldman|Morgan|JPMorgan|Citi|UBS|Wells|Deutsche|Barclays|BofA|Bank\s+of\s+America|Bernstein|Cowen|Raymond|Jefferies|Piper|Evercore|Oppenheimer|Mizuho))\b',
]
_BP_RE = re.compile('|'.join(_BP_PATTERNS), re.IGNORECASE)

_SUBST_PATTERNS = [
    r'\$[\d,\.]+\s*(billion|million|thousand|b|m)?',
    r'\b\d+(\.\d+)?\s*%',
    r'\b(revenue|eps|earnings\s+per\s+share|gross\s+margin|operating\s+(income|margin)|net\s+(income|loss)|ebitda|guidance|outlook|forecast)\b',
    r'\b(year[- ]over[- ]year|sequentially|quarter[- ]over[- ]quarter|yoy|qoq)\b',
    r'\b(raised?|lowered?|reaffirmed?|expect(s|ed)?|anticipate[sd]?)\s+(guidance|outlook|revenue|earnings)\b',
    r'\b(record\s+(revenue|quarter|sales|profit)|all[- ]time\s+high)\b',
]
_SUBST_RE = re.compile('|'.join(_SUBST_PATTERNS), re.IGNORECASE)

def rules_judge(text: str) -> int:
    """Returns 0 (boilerplate) or 1 (substantive)."""
    if _BP_RE.search(text):
        # Override if strong substantive signal also present
        if _SUBST_RE.search(text) and len(text) > 80:
            return 1
        return 0
    if _SUBST_RE.search(text):
        return 1
    # Short sentences with no signal → boilerplate
    if len(text.split()) < 12:
        return 0
    return 1  # default: substantive for longer sentences without BP markers

# Test rules judge
test_cases = [
    ("My name is Regina and I will be your conference operator today.", 0),
    ("Revenue of $26 billion was up 18% sequentially and up 262% year-on-year.", 1),
    ("These are subject to a number of significant risks and uncertainties.", 0),
    ("We expect full-year revenue growth of approximately 15% to 20%.", 1),
    ("Thank you.", 0),
    ("Your line is open.", 0),
]
print('Rules judge sanity check:')
all_pass = True
for text, expected in test_cases:
    pred = rules_judge(text)
    status = '✓' if pred == expected else '✗'
    if pred != expected: all_pass = False
    print(f'  {status} [{"BP" if pred==0 else "SB"}] {text[:70]}')
print('All pass!' if all_pass else 'Some mismatches — review patterns.')


# ═══ CELL [20] ═══
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_recall_curve, roc_auc_score
)
import time

SUBSTANTIVE_CLASS = 1
RECALL_FLOOR = 0.96

leaderboard = []

def evaluate(name, y_true, y_pred, y_proba=None, train_time=None, infer_time=None, n_infer=None):
    acc    = (y_true == y_pred).mean()
    mf1    = f1_score(y_true, y_pred, average='macro')
    bp_f1  = f1_score(y_true, y_pred, pos_label=0)
    sb_f1  = f1_score(y_true, y_pred, pos_label=1)
    rep    = classification_report(y_true, y_pred,
                                   target_names=['boilerplate','substantive'], digits=3)
    cm     = confusion_matrix(y_true, y_pred)
    sb_rec = cm[1,1] / max(1, cm[1,:].sum())
    throughput = (n_infer / infer_time) if (infer_time and n_infer) else None
    row = {
        'model': name, 'accuracy': acc, 'macro_f1': mf1,
        'bp_f1': bp_f1, 'sb_f1': sb_f1, 'sb_recall': sb_rec,
        'meets_floor': sb_rec >= RECALL_FLOOR,
        'train_sec': train_time, 'throughput_sps': throughput,
    }
    leaderboard.append(row)
    print(f'\n=== {name} ===')
    print(rep)
    print(f'Confusion matrix:\n{cm}')
    if train_time  is not None: print(f'Train time: {train_time:.1f}s')
    if throughput  is not None: print(f'Throughput: {throughput:.0f} sentences/sec')
    return row

def threshold_sweep(y_true, probas_subst, recall_floor=RECALL_FLOOR):
    """
    Sweep thresholds [0.01, 0.99].
    Primary: highest macro-F1 subject to substantive recall >= recall_floor.
    Fallback (floor unachievable): threshold with highest substantive recall,
    then highest macro-F1 among ties. Always returns a float, never None.
    """
    thresholds = np.linspace(0.01, 0.99, 197)
    best_t, best_f1 = None, -1
    # fallback tracking
    fb_t, fb_rec, fb_f1 = thresholds[0], -1.0, -1.0

    for t in thresholds:
        preds = (probas_subst >= t).astype(int)
        if preds.sum() == 0:
            continue
        sb_rec = ((preds == 1) & (y_true == 1)).sum() / max(1, (y_true == 1).sum())
        mf1    = f1_score(y_true, preds, average='macro')
        # primary
        if sb_rec >= recall_floor and mf1 > best_f1:
            best_f1, best_t = mf1, t
        # fallback: prefer highest recall, break ties with macro-F1
        if sb_rec > fb_rec or (sb_rec == fb_rec and mf1 > fb_f1):
            fb_rec, fb_f1, fb_t = sb_rec, mf1, t

    if best_t is None:
        print(f'  WARNING: no threshold achieves recall >= {recall_floor:.2f}. '
              f'Falling back to best-recall threshold {fb_t:.3f} '
              f'(recall={fb_rec:.3f}, macro-F1={fb_f1:.3f}).')
        return fb_t, fb_f1
    return best_t, best_f1

def apply_threshold(probas, threshold):
    return (probas >= threshold).astype(int)

print('Evaluation helpers ready.')


# ═══ CELL [21] ═══
# ── Classifier 1: Rules + regex ───────────────────────────────────────────────

t0 = time.time()
y_pred_rules = np.array([rules_judge(t) for t in X_val])
infer_time = time.time() - t0

evaluate('1-Rules+Regex', y_val, y_pred_rules,
         train_time=0, infer_time=infer_time, n_infer=len(X_val))

# ═══ CELL [22] ═══
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

t0 = time.time()
lr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED,
                               class_weight='balanced')),
])
lr_pipe.fit(F_train, y_train)
train_time = time.time() - t0

t0 = time.time()
y_pred_lr = lr_pipe.predict(F_val)
proba_lr  = lr_pipe.predict_proba(F_val)[:, 1]
infer_time = time.time() - t0

# Diagnostic: probability distribution
print(f'P(substantive) stats — min={proba_lr.min():.3f}  max={proba_lr.max():.3f}  '
      f'mean={proba_lr.mean():.3f}  median={np.median(proba_lr):.3f}')

best_t_lr, best_f1_lr = threshold_sweep(y_val, proba_lr)
y_pred_lr_t = apply_threshold(proba_lr, best_t_lr)
print(f'LogReg best threshold: {best_t_lr:.3f}  (macro-F1={best_f1_lr:.4f})')

evaluate('2-LogReg(emb+regex)', y_val, y_pred_lr_t, proba_lr,
         train_time=train_time, infer_time=infer_time, n_infer=len(X_val))


# ═══ CELL [23] ═══
from sklearn.ensemble import HistGradientBoostingClassifier

t0 = time.time()
gbm = HistGradientBoostingClassifier(
    max_iter=300, learning_rate=0.05, max_depth=6,
    min_samples_leaf=20, random_state=RANDOM_SEED,
    class_weight='balanced',
)
gbm.fit(F_train, y_train)
train_time = time.time() - t0

t0 = time.time()
proba_gbm  = gbm.predict_proba(F_val)[:, 1]
infer_time = time.time() - t0

print(f'P(substantive) stats — min={proba_gbm.min():.3f}  max={proba_gbm.max():.3f}  '
      f'mean={proba_gbm.mean():.3f}  median={np.median(proba_gbm):.3f}')

best_t_gbm, best_f1_gbm = threshold_sweep(y_val, proba_gbm)
y_pred_gbm_t = apply_threshold(proba_gbm, best_t_gbm)
print(f'HistGBM best threshold: {best_t_gbm:.3f}  (macro-F1={best_f1_gbm:.4f})')

evaluate('3-HistGBM(emb+regex)', y_val, y_pred_gbm_t, proba_gbm,
         train_time=train_time, infer_time=infer_time, n_infer=len(X_val))


# ═══ CELL [24] ═══
import subprocess, sys, tempfile

try:
    import fasttext
except ModuleNotFoundError:
    print('Installing fasttext-wheel...')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'fasttext-wheel'])
    import fasttext
    print('fasttext installed.')

def write_fasttext_data(texts, labels, path):
    with open(path, 'w', encoding='utf-8') as f:
        for t, l in zip(texts, labels):
            clean = t.replace('\n', ' ').strip()
            f.write(f'__label__{l} {clean}\n')

with tempfile.TemporaryDirectory() as tmpdir:
    train_file = os.path.join(tmpdir, 'train.txt')
    write_fasttext_data(X_train, y_train, train_file)

    t0 = time.time()
    ft_model = fasttext.train_supervised(
        input=train_file,
        epoch=25, lr=0.5, wordNgrams=2, dim=100, minCount=2,
        loss='softmax', verbose=0,
    )
    train_time = time.time() - t0

t0 = time.time()
ft_preds_raw = ft_model.predict([t.replace('\n', ' ') for t in X_val])
infer_time   = time.time() - t0

ft_probas = np.array([
    (p[0] if l[0] == '__label__1' else 1 - p[0])
    for l, p in zip(ft_preds_raw[0], ft_preds_raw[1])
])

print(f'P(substantive) stats — min={ft_probas.min():.3f}  max={ft_probas.max():.3f}  '
      f'mean={ft_probas.mean():.3f}  median={np.median(ft_probas):.3f}')

best_t_ft, best_f1_ft = threshold_sweep(y_val, ft_probas)
y_pred_ft_t = apply_threshold(ft_probas, best_t_ft)
print(f'FastText best threshold: {best_t_ft:.3f}  (macro-F1={best_f1_ft:.4f})')

evaluate('4-FastText', y_val, y_pred_ft_t, ft_probas,
         train_time=train_time, infer_time=infer_time, n_infer=len(X_val))

ft_model.save_model(str(MODEL_DIR / 'fasttext_model.bin'))
print('FastText model saved.')


# ═══ CELL [25] ═══
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
)
from datasets import Dataset as HFDataset

FINBERT_MODEL = 'ProsusAI/finbert'
FINBERT_DIR   = MODEL_DIR / 'finbert_finetuned'

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Device: {device}')

def make_hf_dataset(texts, labels):
    return HFDataset.from_dict({'text': list(texts), 'label': list(labels.astype(int))})

tokenizer_fb = AutoTokenizer.from_pretrained(FINBERT_MODEL)

def tokenize_fn(batch):
    return tokenizer_fb(batch['text'], truncation=True, max_length=128)

ds_val = make_hf_dataset(X_val, y_val).map(tokenize_fn, batched=True)
collator = DataCollatorWithPadding(tokenizer_fb)

_fb_cached = (FINBERT_DIR / 'config.json').exists()
if _fb_cached:
    print('FinBERT: loading saved model from cache...')
    model_fb = AutoModelForSequenceClassification.from_pretrained(str(FINBERT_DIR))
    model_fb.to(device)
    trainer_fb = Trainer(model=model_fb, args=TrainingArguments(
        output_dir=str(FINBERT_DIR), report_to='none', dataloader_num_workers=0,
        per_device_eval_batch_size=32,
    ), data_collator=collator)
    train_time = 0.0
else:
    ds_train = make_hf_dataset(X_train, y_train).map(tokenize_fn, batched=True)
    model_fb = AutoModelForSequenceClassification.from_pretrained(
        FINBERT_MODEL, num_labels=2, ignore_mismatched_sizes=True)
    model_fb.to(device)
    training_args = TrainingArguments(
        output_dir=str(FINBERT_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        logging_steps=50,
        report_to='none',
        dataloader_num_workers=0,
    )
    trainer_fb = Trainer(
        model=model_fb, args=training_args,
        train_dataset=ds_train, eval_dataset=ds_val,
        data_collator=collator,
    )
    t0 = time.time()
    trainer_fb.train()
    train_time = time.time() - t0
    print(f'FinBERT fine-tuning done in {train_time/60:.1f} min')
    trainer_fb.save_model(str(FINBERT_DIR))
    print('FinBERT model saved.')

t0 = time.time()
fb_out     = trainer_fb.predict(ds_val)
infer_time = time.time() - t0
proba_fb   = torch.softmax(torch.tensor(fb_out.predictions), dim=-1).numpy()[:, 1]

print(f'P(substantive) stats — min={proba_fb.min():.3f}  max={proba_fb.max():.3f}  '
      f'mean={proba_fb.mean():.3f}  median={np.median(proba_fb):.3f}')

best_t_fb, best_f1_fb = threshold_sweep(y_val, proba_fb)
y_pred_fb_t = apply_threshold(proba_fb, best_t_fb)
print(f'FinBERT best threshold: {best_t_fb:.3f}  (macro-F1={best_f1_fb:.4f})')

evaluate('5-FinBERT-FT', y_val, y_pred_fb_t, proba_fb,
         train_time=train_time, infer_time=infer_time, n_infer=len(X_val))


# ═══ CELL [26] ═══
import subprocess, sys

try:
    import setfit
except ModuleNotFoundError:
    print('Installing setfit...')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'setfit'])
    print('setfit installed.')

from setfit import SetFitModel, Trainer as SetFitTrainer, TrainingArguments as SetFitArgs
from datasets import Dataset as HFDataset

SETFIT_BASE = 'sentence-transformers/all-MiniLM-L6-v2'
SETFIT_DIR  = MODEL_DIR / 'setfit_model'

sf_train_ds = HFDataset.from_dict({'text': list(X_train), 'label': list(y_train.astype(int))})
sf_val_ds   = HFDataset.from_dict({'text': list(X_val),   'label': list(y_val.astype(int))})

sf_model = SetFitModel.from_pretrained(SETFIT_BASE, num_labels=2)

# Build args — omit report_to (not valid in all setfit versions)
_sf_arg_kwargs = dict(
    output_dir=str(SETFIT_DIR),
    num_epochs=1,
    batch_size=32,
    num_iterations=5,
    seed=RANDOM_SEED,
)
try:
    sf_args = SetFitArgs(**_sf_arg_kwargs)
except TypeError:
    # Older setfit API: TrainingArguments may not exist; fall back to Trainer kwargs
    sf_args = None

if sf_args is not None:
    sf_trainer = SetFitTrainer(
        model=sf_model, args=sf_args,
        train_dataset=sf_train_ds, eval_dataset=sf_val_ds,
    )
else:
    sf_trainer = SetFitTrainer(
        model=sf_model,
        train_dataset=sf_train_ds, eval_dataset=sf_val_ds,
        num_iterations=5, num_epochs=1,
    )

t0 = time.time()
sf_trainer.train()
train_time = time.time() - t0
print(f'SetFit training done in {train_time/60:.1f} min')

t0 = time.time()
sf_probas_raw = sf_model.predict_proba(list(X_val))
infer_time = time.time() - t0
proba_sf = np.array(sf_probas_raw)[:, 1] if np.array(sf_probas_raw).ndim > 1 else np.array(sf_probas_raw)

print(f'P(substantive) stats — min={proba_sf.min():.3f}  max={proba_sf.max():.3f}  '
      f'mean={proba_sf.mean():.3f}  median={np.median(proba_sf):.3f}')

best_t_sf, best_f1_sf = threshold_sweep(y_val, proba_sf)
y_pred_sf_t = apply_threshold(proba_sf, best_t_sf)
print(f'SetFit best threshold: {best_t_sf:.3f}  (macro-F1={best_f1_sf:.4f})')

evaluate('6-SetFit', y_val, y_pred_sf_t, proba_sf,
         train_time=train_time, infer_time=infer_time, n_infer=len(X_val))

sf_model.save_pretrained(str(SETFIT_DIR))
print('SetFit model saved.')


# ═══ CELL [28] ═══
from sklearn.model_selection import StratifiedKFold

# Train+val pool for OOF threshold tuning
X_tv = np.concatenate([X_train, X_val])
y_tv = np.concatenate([y_train, y_val])
F_tv = np.vstack([F_train, F_val])

K = 5
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=RANDOM_SEED)

# --- OOF probabilities for the best non-transformer model
# We'll run HistGBM and LogReg; pick the winner

def oof_probas(estimator_factory, X_feats, y, cv):
    oof = np.zeros(len(y))
    for fold_i, (tr_idx, va_idx) in enumerate(cv.split(X_feats, y)):
        clf = estimator_factory()
        clf.fit(X_feats[tr_idx], y[tr_idx])
        oof[va_idx] = clf.predict_proba(X_feats[va_idx])[:, 1]
        print(f'  fold {fold_i+1}/{K} done')
    return oof

print('Computing OOF probabilities (HistGBM, 5-fold)...')
oof_gbm = oof_probas(
    lambda: HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=6,
        min_samples_leaf=20, random_state=RANDOM_SEED, class_weight='balanced'),
    F_tv, y_tv, skf
)

print('Computing OOF probabilities (LogReg, 5-fold)...')
from sklearn.pipeline import Pipeline
oof_lr = oof_probas(
    lambda: Pipeline([('s', StandardScaler()),
                      ('c', LogisticRegression(C=1.0, max_iter=1000,
                                               class_weight='balanced',
                                               random_state=RANDOM_SEED))]),
    F_tv, y_tv, skf
)

# Threshold sweep on OOF
t_gbm_oof, mf1_gbm_oof = threshold_sweep(y_tv, oof_gbm)
t_lr_oof,  mf1_lr_oof  = threshold_sweep(y_tv, oof_lr)
print(f'\nOOF results:')
print(f'  HistGBM  threshold={t_gbm_oof:.3f}  OOF macro-F1={mf1_gbm_oof:.4f}')
print(f'  LogReg   threshold={t_lr_oof:.3f}   OOF macro-F1={mf1_lr_oof:.4f}')

# Per-fold threshold variance
fold_thresholds_gbm = []
for tr_idx, va_idx in skf.split(F_tv, y_tv):
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=6,
        min_samples_leaf=20, random_state=RANDOM_SEED, class_weight='balanced')
    clf.fit(F_tv[tr_idx], y_tv[tr_idx])
    p = clf.predict_proba(F_tv[va_idx])[:,1]
    t, _ = threshold_sweep(y_tv[va_idx], p)
    fold_thresholds_gbm.append(t)

print(f'\nHistGBM per-fold thresholds: {[f"{t:.3f}" for t in fold_thresholds_gbm]}')
print(f'  mean={np.mean(fold_thresholds_gbm):.3f}  std={np.std(fold_thresholds_gbm):.4f}')

# ═══ CELL [30] ═══
# ── Ensemble: mean probability of top 5 classifiers ──────────────────────────
# Collect val-set probabilities from classifiers 2–6

# Re-score all models on val to get probas (some already computed above)
# Classifier 1 (rules) has no probas — use 0.0 or skip.
# Use: LR, GBM, FastText, FinBERT, SetFit

ensemble_probas_val = np.column_stack([
    proba_lr,   # LogReg
    proba_gbm,  # HistGBM
    ft_probas,  # FastText
    proba_fb,   # FinBERT
    proba_sf,   # SetFit
])

mean_proba_val  = ensemble_probas_val.mean(axis=1)

# Rank-averaged ensemble
from scipy.stats import rankdata
ranks = np.column_stack([rankdata(p) for p in ensemble_probas_val.T])
rank_proba_val = ranks.mean(axis=1) / len(mean_proba_val)

for ens_name, ens_proba in [('7a-Ensemble(mean-prob)', mean_proba_val),
                             ('7b-Ensemble(rank-avg)',  rank_proba_val)]:
    best_t, _ = threshold_sweep(y_val, ens_proba)
    y_pred_ens = (ens_proba >= best_t).astype(int) if best_t else (ens_proba >= 0.5).astype(int)
    print(f'{ens_name} best threshold: {best_t:.3f}')
    evaluate(ens_name, y_val, y_pred_ens, ens_proba, train_time=0)

# ═══ CELL [31] ═══
# ── Leaderboard (validation set) ─────────────────────────────────────────────

lb_df = pd.DataFrame(leaderboard).sort_values('macro_f1', ascending=False).reset_index(drop=True)
lb_df['rank'] = lb_df.index + 1
print('\n=== LEADERBOARD (validation set, sorted by macro-F1) ===')
print(lb_df[['rank','model','accuracy','macro_f1','bp_f1','sb_f1','sb_recall',
             'meets_floor','train_sec','throughput_sps']].to_string(index=False))

# ═══ CELL [32] ═══
# ── Final evaluation on HELD-OUT TEST SET ─────────────────────────────────────

# Pick best val model that meets recall floor
eligible = lb_df[lb_df['meets_floor']]
if eligible.empty:
    print('WARNING: No model meets the recall floor on val!')
else:
    best_name = eligible.iloc[0]['model']
    print(f'Best val model: {best_name}')

F_test = build_full_features(X_test)

# ── Re-train final HistGBM on train+val ──────────────────────────────────────
print('\nRe-training HistGBM on train+val pool...')
final_gbm = HistGradientBoostingClassifier(
    max_iter=300, learning_rate=0.05, max_depth=6,
    min_samples_leaf=20, random_state=RANDOM_SEED, class_weight='balanced')
final_gbm.fit(F_tv, y_tv)
proba_test_gbm = final_gbm.predict_proba(F_test)[:, 1]

# Use val-optimized threshold (meets recall floor on val); OOF threshold
# is tuned for macro-F1 and may fail the floor on test due to distribution shift.
final_gbm_threshold = best_t_gbm
print(f'Using val threshold: {final_gbm_threshold:.3f}  (OOF was {t_gbm_oof:.3f})')

# ── Ensemble test probas (LR + GBM + FastText + FinBERT + SetFit) ─────────────
print('\nComputing test probabilities for ensemble...')

# LogReg
proba_test_lr = lr_pipe.predict_proba(F_test)[:, 1]

# FastText
ft_preds_test_raw = ft_model.predict([t.replace('\n', ' ') for t in X_test])
proba_test_ft = np.array([
    (p[0] if l[0] == '__label__1' else 1 - p[0])
    for l, p in zip(ft_preds_test_raw[0], ft_preds_test_raw[1])
])

# FinBERT
ds_test_fb = make_hf_dataset(X_test, y_test).map(tokenize_fn, batched=True)
fb_out_test = trainer_fb.predict(ds_test_fb)
proba_test_fb = torch.softmax(torch.tensor(fb_out_test.predictions), dim=-1).numpy()[:, 1]

# SetFit
proba_test_sf_raw = sf_model.predict_proba(list(X_test))
proba_test_sf = np.array(proba_test_sf_raw)[:, 1] if np.array(proba_test_sf_raw).ndim > 1 else np.array(proba_test_sf_raw)

# Ensemble (mean prob)
ensemble_probas_test = np.column_stack([
    proba_test_lr, proba_test_gbm, proba_test_ft, proba_test_fb, proba_test_sf
])
mean_proba_test = ensemble_probas_test.mean(axis=1)

# Best ensemble threshold: from val set (already computed as mean_proba_val)
best_t_ens_val, _ = threshold_sweep(y_val, mean_proba_val)

# ── Test results ──────────────────────────────────────────────────────────────
from sklearn.metrics import classification_report, confusion_matrix

def report_test(name, y_true, y_pred):
    print(f'\n=== TEST: {name} ===')
    print(classification_report(y_true, y_pred,
                                target_names=['boilerplate','substantive'], digits=4))
    cm = confusion_matrix(y_true, y_pred)
    print(f'Confusion matrix:\n{cm}')
    sb_rec = cm[1,1] / max(1, cm[1,:].sum())
    mf1 = f1_score(y_true, y_pred, average='macro')
    bp_f1 = f1_score(y_true, y_pred, pos_label=0)
    print(f'Substantive recall: {sb_rec:.4f}  (floor={RECALL_FLOOR})')
    print(f'Meets floor: {sb_rec >= RECALL_FLOOR}')
    return sb_rec, mf1, bp_f1

print('\n' + '='*60)
print('  FINAL TEST SET RESULTS')
print('='*60)

# Individual models
sbr_gbm, mf1_gbm, bp_gbm = report_test(
    f'HistGBM (t={final_gbm_threshold:.3f})',
    y_test, apply_threshold(proba_test_gbm, final_gbm_threshold))

sbr_ens, mf1_ens, bp_ens = report_test(
    f'Ensemble-mean (t={best_t_ens_val:.3f})',
    y_test, apply_threshold(mean_proba_test, best_t_ens_val))

print('\n--- Test Summary ---')
print(f'  HistGBM   macro-F1={mf1_gbm:.4f}  BP-F1={bp_gbm:.4f}  SB-recall={sbr_gbm:.4f}  {"✓" if sbr_gbm >= RECALL_FLOOR else "✗"}')
print(f'  Ensemble  macro-F1={mf1_ens:.4f}  BP-F1={bp_ens:.4f}  SB-recall={sbr_ens:.4f}  {"✓" if sbr_ens >= RECALL_FLOOR else "✗"}')

# ═══ CELL [33] ═══
# ── Save best model to disk ───────────────────────────────────────────────────

best_model_payload = {
    'model': final_gbm,
    'threshold': float(final_gbm_threshold),
    'embed_model': EMBED_MODEL_NAME,
    'feat_names': feat_names,
}

with open(MODEL_DIR / 'best_model.pkl', 'wb') as f:
    pickle.dump(best_model_payload, f)
print('\nSaved best_model.pkl')
print(f'  Model:     HistGBM (retrained on train+val)')
print(f'  Threshold: {final_gbm_threshold:.3f}  (val-optimized, recall-floor-constrained)')
print(f'  Embedding: {EMBED_MODEL_NAME}')


# ═══ CELL [34] — §9 Error Analysis ═══
_section('§9 Error Analysis')

import textwrap

# Build val DataFrame from in-memory arrays, then join on text for metadata
gold_all = pd.read_parquet(CACHE / 'gold' / 'gold_labels.parquet')
df_val   = pd.DataFrame({'text': list(X_val), 'true_label': y_val.astype(int)})
df_val   = df_val.merge(
    gold_all[['text','ticker','quarter','speaker_type']].drop_duplicates('text'),
    on='text', how='left'
)

df_val['proba_gbm']   = proba_gbm
df_val['proba_ens']   = mean_proba_val
df_val['pred_gbm']    = apply_threshold(proba_gbm,   final_gbm_threshold)
df_val['pred_ens']    = apply_threshold(mean_proba_val, best_t_ens_val)
# true_label already set from merge above

# Error type labels
def error_type(row):
    t, p = int(row['true_label']), int(row['pred_gbm'])
    if t == p:   return 'TP' if t == 1 else 'TN'
    if t == 1:   return 'FN'  # substantive predicted boilerplate
    return 'FP'               # boilerplate predicted substantive

df_val['error_gbm'] = df_val.apply(error_type, axis=1)

fn_df = df_val[df_val['error_gbm'] == 'FN'].sort_values('proba_gbm')   # lowest SB score first
fp_df = df_val[df_val['error_gbm'] == 'FP'].sort_values('proba_gbm', ascending=False)  # highest SB score first

print(f'\nVal error breakdown (HistGBM t={final_gbm_threshold:.3f}):')
for et, cnt in df_val['error_gbm'].value_counts().items():
    print(f'  {et}: {cnt}')

# ── Feature contribution analysis ──────────────────────────────────────────
print(f'\nRegex feature rates in FN vs FP vs all val sentences:')
regex_cols = [c for c in feat_names if c.startswith('f_')]
fn_feat_idx = [feat_names.index(c) for c in regex_cols if c in feat_names]

fn_mask = (df_val['error_gbm'] == 'FN').values
fp_mask = (df_val['error_gbm'] == 'FP').values

fn_feats = F_val[fn_mask, :][:, fn_feat_idx].mean(axis=0) if fn_mask.sum() > 0 else np.zeros(len(fn_feat_idx))
fp_feats = F_val[fp_mask, :][:, fn_feat_idx].mean(axis=0) if fp_mask.sum() > 0 else np.zeros(len(fn_feat_idx))
all_feats= F_val[:, fn_feat_idx].mean(axis=0)

print(f'  {"Feature":<30} {"FN avg":>8} {"FP avg":>8} {"all avg":>8}')
for i, c in enumerate(regex_cols):
    if c not in feat_names: continue
    if fn_feats[i] > 0.01 or fp_feats[i] > 0.01:  # only show active features
        print(f'  {c:<30} {fn_feats[i]:>8.3f} {fp_feats[i]:>8.3f} {all_feats[i]:>8.3f}')

# ── Speaker-type error breakdown ───────────────────────────────────────────
print(f'\nError rate by speaker_type (HistGBM):')
for stype, grp in df_val.groupby('speaker_type'):
    n = len(grp)
    fn = (grp['error_gbm'] == 'FN').sum()
    fp = (grp['error_gbm'] == 'FP').sum()
    print(f'  {stype:<12}  n={n:4d}  FN={fn:3d} ({fn/n*100:4.1f}%)  FP={fp:3d} ({fp/n*100:4.1f}%)')

# ── FN examples (missed substantive) ──────────────────────────────────────
print(f'\n— False Negatives: TOP 10 most-missed substantive sentences —')
print(f'  (model incorrectly labeled as boilerplate, sorted by lowest SB confidence)\n')
for _, row in fn_df.head(10).iterrows():
    print(f'  P(SB)={row["proba_gbm"]:.3f}  speaker={row.get("speaker_type","?")}')
    print(f'  "{textwrap.shorten(row["text"], 120, placeholder="...")}"')
    print()

# ── FP examples (boilerplate called substantive) ──────────────────────────
print(f'— False Positives: TOP 10 boilerplate sentences called substantive —')
print(f'  (sorted by highest SB confidence)\n')
for _, row in fp_df.head(10).iterrows():
    print(f'  P(SB)={row["proba_gbm"]:.3f}  speaker={row.get("speaker_type","?")}')
    print(f'  "{textwrap.shorten(row["text"], 120, placeholder="...")}"')
    print()

# ── Save error analysis CSV ─────────────────────────────────────────────
err_path = CACHE / 'error_analysis_val.csv'
df_val[['ticker','quarter','speaker_type','text','true_label',
        'proba_gbm','pred_gbm','error_gbm','proba_ens','pred_ens']].to_csv(err_path, index=False)
print(f'Error analysis saved: {err_path}  ({len(df_val)} rows)')

# ── Calibration / confidence distribution ─────────────────────────────────
print(f'\nConfidence distribution on val (HistGBM):')
bins = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.01]
labels = ['0–0.2','0.2–0.4','0.4–0.6','0.6–0.8','0.8–0.9','0.9–0.95','0.95–1.0']
df_val['conf_bin'] = pd.cut(df_val['proba_gbm'], bins=bins, labels=labels, right=False)
for b, grp in df_val.groupby('conf_bin', observed=True):
    n = len(grp); acc = (grp['true_label']==grp['pred_gbm']).mean()
    act_sb = grp['true_label'].mean()
    print(f'  P={b}  n={n:3d}  accuracy={acc:.3f}  frac_SB={act_sb:.3f}')

_section('Pipeline complete')

