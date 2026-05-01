"""
BPClassifier GUI — Streamlit app for inline boilerplate tagging.

Run:
    streamlit run gui.py

Loads saved_model/best_model.pkl at startup. Accepts a .txt earnings-call
transcript via file upload or text paste; displays every sentence in its
original position with boilerplate highlighted in red.
"""

import pickle, re, html as _html
from pathlib import Path

import numpy as np
import streamlit as st

# ── Load model once ───────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / 'saved_model' / 'best_model.pkl'

@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)

# ── Feature helpers (must match notebook §5) ──────────────────────────────────
REGEX_FEATURES = [
    ('f_operator_phrase',   r'(?i)(\bmy name is\b|conference operator|welcome everyone|welcome to nvidia|welcome to the)'),
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
    ('f_sentence_short',    None),
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

def predict_sentences(sentences, payload):
    if not sentences:
        return np.array([], dtype=int), np.array([], dtype=float)
    embedder = load_embedder(payload['embed_model'])
    emb = embedder.encode(sentences, batch_size=128, show_progress_bar=False,
                          convert_to_numpy=True, normalize_embeddings=True)
    reg = build_regex_features(sentences)
    X   = np.hstack([emb, reg])
    probas = payload['model'].predict_proba(X)[:, 1]
    labels = (probas >= payload['threshold']).astype(int)
    return labels, probas

# ── Sentence splitter ─────────────────────────────────────────────────────────
def split_sentences(text: str, min_chars: int = 40):
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    from nltk.tokenize import sent_tokenize
    sentences, kept = [], []
    for sent in sent_tokenize(text):
        s = sent.strip()
        if len(s) >= min_chars:
            kept.append(s)
        sentences.append((s, len(s) >= min_chars))
    return sentences, kept

# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title='BPClassifier', layout='wide')
st.title('Boilerplate vs. Substantive Sentence Classifier')
st.caption('NLP for Finance — Spring 2026 | Assignment 2')

# Sidebar
with st.sidebar:
    st.header('About')
    st.markdown("""
    **Red background** = boilerplate
    (scripted intros, safe-harbor, housekeeping)

    **No highlight** = substantive
    (financials, guidance, strategy, Q&A specifics)

    **Recall constraint:** substantive recall ≥ 0.96
    **Model:** HistGBM + sentence embeddings + 25 regex flags
    """)
    st.divider()
    if MODEL_PATH.exists():
        st.success('Model loaded ✓')
    else:
        st.error(f'Model not found at {MODEL_PATH}')
        st.stop()

# Input
ECT_DIR = Path(__file__).parent / 'ECT'
ect_files = sorted(ECT_DIR.glob('*.txt')) if ECT_DIR.exists() else []

st.subheader('Load transcript')
tab_ect, tab_upload, tab_paste = st.tabs(['ECT library', 'Upload .txt file', 'Paste text'])

raw_text = ''
with tab_ect:
    if ect_files:
        labels = [f.name for f in ect_files]
        choice = st.selectbox('Select transcript', ['— choose —'] + labels)
        if choice != '— choose —':
            raw_text = (ECT_DIR / choice).read_text(encoding='utf-8', errors='ignore')
            st.caption(f'Loaded: {choice}  ({len(raw_text):,} chars)')
    else:
        st.warning('ECT/ directory not found next to gui.py.')

with tab_upload:
    uploaded = st.file_uploader('Choose a .txt earnings-call transcript', type=['txt'])
    if uploaded:
        raw_text = uploaded.read().decode('utf-8', errors='ignore')

with tab_paste:
    pasted = st.text_area('Paste transcript text here', height=200)
    if pasted.strip():
        raw_text = pasted

if not raw_text.strip():
    st.info('Select a transcript from ECT, upload a file, or paste text to begin.')
    st.stop()

# Run classifier
with st.spinner('Classifying sentences...'):
    payload = load_model()
    all_sents, kept_sents = split_sentences(raw_text)
    if not kept_sents:
        st.warning('No sentences long enough to classify (min 40 characters).')
        st.stop()
    labels, probas = predict_sentences(kept_sents, payload)

# Map labels back to all sentences
label_map = {}
proba_map = {}
idx = 0
for sent, keep in all_sents:
    if keep:
        label_map[sent] = int(labels[idx])
        proba_map[sent]  = float(probas[idx])
        idx += 1

# Statistics panel
n_bp = sum(1 for v in label_map.values() if v == 0)
n_sb = sum(1 for v in label_map.values() if v == 1)
n_total = len(label_map)

st.subheader('Statistics')
col1, col2, col3 = st.columns(3)
col1.metric('Total sentences', n_total)
col2.metric('Boilerplate', f'{n_bp} ({n_bp/n_total*100:.1f}%)')
col3.metric('Substantive', f'{n_sb} ({n_sb/n_total*100:.1f}%)')

# Inline document view
st.divider()
st.subheader('Tagged transcript')

bp_style   = 'background-color: #ffcccc; padding: 2px 4px; border-radius: 2px;'
sb_style   = ''
short_style = 'color: #aaaaaa; font-style: italic;'

import streamlit.components.v1 as _components

html_parts = ['<div style="font-family: sans-serif; line-height: 1.8; font-size: 14px; padding: 4px;">']
for sent, keep in all_sents:
    s = _html.escape(sent.replace('\n', ' ').replace('\r', ' '))
    if not keep:
        html_parts.append(f'<span style="{short_style}">{s}</span> ')
        continue
    lbl = label_map.get(sent, 1)
    prob = proba_map.get(sent, 0.5)
    style = bp_style if lbl == 0 else sb_style
    tag   = 'BP' if lbl == 0 else ''
    title = f'P(substantive)={prob:.3f}'
    if lbl == 0:
        html_parts.append(
            f'<span style="{style}" title="{title}">'
            f'<sup style="font-size:9px;color:#cc0000">{tag}</sup>'
            f'{s}</span> '
        )
    else:
        html_parts.append(f'<span title="{title}">{s}</span> ')
html_parts.append('</div>')

est_height = max(600, len(all_sents) * 28)
_components.html(''.join(html_parts), height=est_height, scrolling=True)

# Download tagged results as CSV
import pandas as pd, io
out_df = pd.DataFrame([
    {'sentence': s, 'label': 'boilerplate' if l == 0 else 'substantive',
     'p_substantive': round(proba_map[s], 4)}
    for s, l in label_map.items()
])
csv_buf = io.StringIO()
out_df.to_csv(csv_buf, index=False)
st.download_button('Download tagged sentences (CSV)', csv_buf.getvalue(),
                   file_name='tagged_transcript.csv', mime='text/csv')
