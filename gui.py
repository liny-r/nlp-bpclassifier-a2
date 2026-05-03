"""
BPClassifier GUI — Streamlit app for inline boilerplate tagging.

Run:
    streamlit run gui.py

If setfit is not found in the active Python, the script automatically
re-launches itself using the Anaconda/Miniconda Streamlit that has it.

Loads the winning model (SetFit, all-MiniLM-L6-v2 contrastive fine-tuned)
from saved_model/setfit_model/ and its threshold from saved_model/winner.json.
"""

# ── Auto-restart with Anaconda Streamlit if setfit is missing ─────────────────
# This block runs before 'import streamlit' so the process is replaced cleanly.
import os as _os, sys as _sys
try:
    import setfit as _sf
    _SETFIT_OK   = True
    _SetFitModel = _sf.SetFitModel
except ImportError:
    _SETFIT_OK   = False
    _SetFitModel = None
    from pathlib import Path as _P
    _conda = _os.environ.get('CONDA_PREFIX', '')
    _candidates = [
        *([_P(_conda) / 'bin' / 'streamlit'] if _conda else []),
        _P.home() / 'anaconda3'          / 'bin' / 'streamlit',
        _P.home() / 'miniconda3'         / 'bin' / 'streamlit',
        _P.home() / 'opt' / 'anaconda3'  / 'bin' / 'streamlit',
        _P.home() / 'opt' / 'miniconda3' / 'bin' / 'streamlit',
        _P('/opt/anaconda3/bin/streamlit'),
        _P('/opt/miniconda3/bin/streamlit'),
    ]
    for _st_bin in _candidates:
        if _st_bin.exists():
            print(f'\n[BPClassifier] setfit not found — re-launching with {_st_bin}\n',
                  flush=True)
            _os.execv(str(_st_bin), [str(_st_bin), 'run', __file__] + _sys.argv[1:])

# ── Regular imports ───────────────────────────────────────────────────────────
import json, html as _html
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as _components

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
WINNER_PATH = ROOT / 'saved_model' / 'winner.json'
SETFIT_PATH = ROOT / 'saved_model' / 'setfit_model'

# ── Load winner model once ────────────────────────────────────────────────────
@st.cache_resource
def load_winner():
    winner = json.loads(WINNER_PATH.read_text())
    threshold = float(winner['threshold'])
    model = _SetFitModel.from_pretrained(str(SETFIT_PATH))
    return model, threshold

# ── SetFit inference ──────────────────────────────────────────────────────────
def predict(sentences, model, threshold):
    probas = model.predict_proba(list(sentences), as_numpy=True)[:, 1]
    return (probas >= threshold).astype(int), probas

# ── Paragraph-aware sentence splitter ────────────────────────────────────────
def split_preserving_lines(text: str, min_chars: int = 40):
    import nltk
    nltk.download('punkt_tab', quiet=True)
    from nltk.tokenize import sent_tokenize

    tokens = []  # ('sentence', keep, text) | ('break',)
    kept   = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            tokens.append(('break',))
            continue
        for s in sent_tokenize(line):
            s = s.strip()
            if not s:
                continue
            keep = len(s) >= min_chars
            tokens.append(('sentence', keep, s))
            if keep:
                kept.append(s)
        tokens.append(('break',))

    return tokens, kept

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title='Boilerplate Detector', layout='wide')

st.markdown(
    '<style>.block-container{padding-top:3rem;padding-bottom:1rem;}</style>',
    unsafe_allow_html=True,
)

# ── Session state ─────────────────────────────────────────────────────────────
if 'raw_text'     not in st.session_state: st.session_state.raw_text     = ''
if 'source_label' not in st.session_state: st.session_state.source_label = ''

# ── Compact header ────────────────────────────────────────────────────────────
hdr_col, clear_col = st.columns([8, 1])
with hdr_col:
    st.markdown(
        '<p style="margin:0;font-size:24px;font-weight:700;">Boilerplate Detector'
        '<span style="font-size:12px;color:#888;font-weight:normal;margin-left:10px;">'
        'SetFit &nbsp;·&nbsp; all-MiniLM-L6-v2 (contrastive fine-tuned) &nbsp;·&nbsp; test macro-F1 = 0.931'
        '</span></p>'
        '<p style="margin:0 0 8px;font-size:12px;color:#888;">'
        'Highlights scripted intros, safe-harbor language, and operator chatter in earnings-call transcripts.</p>',
        unsafe_allow_html=True,
    )
with clear_col:
    st.markdown('<div style="margin-top:6px"></div>', unsafe_allow_html=True)
    if st.button('✕ Clear', use_container_width=True):
        st.session_state.raw_text    = ''
        st.session_state.source_label = ''
        st.rerun()

# ── Input tabs (Upload | Paste | ECT library) ─────────────────────────────────
ECT_DIR   = ROOT / 'ECT'
ect_files = sorted(ECT_DIR.glob('*.txt')) if ECT_DIR.exists() else []

tab_upload, tab_paste, tab_ect = st.tabs(['Upload .txt file', 'Paste text', 'ECT library'])

with tab_upload:
    uploaded = st.file_uploader('Choose a .txt earnings-call transcript', type=['txt'],
                                label_visibility='collapsed')
    if uploaded:
        st.session_state.raw_text    = uploaded.read().decode('utf-8', errors='ignore')
        st.session_state.source_label = uploaded.name

with tab_paste:
    pasted = st.text_area('Paste transcript', height=160, label_visibility='collapsed',
                          placeholder='Paste raw earnings-call transcript here…')
    if pasted.strip():
        st.session_state.raw_text    = pasted
        st.session_state.source_label = 'pasted text'

with tab_ect:
    if ect_files:
        choice = st.selectbox('Select transcript', ['— choose —'] + [f.name for f in ect_files],
                              label_visibility='collapsed')
        if choice != '— choose —':
            st.session_state.raw_text    = (ECT_DIR / choice).read_text(encoding='utf-8', errors='ignore')
            st.session_state.source_label = choice
    else:
        st.warning('ECT/ directory not found next to gui.py.')

raw_text = st.session_state.raw_text

if not raw_text.strip():
    st.info('Select a transcript from the ECT library, upload a .txt file, or paste text above.')
    st.stop()

# ── Classify ──────────────────────────────────────────────────────────────────
with st.spinner('Classifying with SetFit…'):
    model, threshold = load_winner()
    tokens, kept_sents = split_preserving_lines(raw_text)
    if not kept_sents:
        st.warning('No sentences long enough to classify (min 40 chars).')
        st.stop()
    labels_arr, probas_arr = predict(kept_sents, model, threshold)

label_map = {}
proba_map = {}
idx = 0
for tok in tokens:
    if tok[0] == 'sentence' and tok[1]:
        label_map[tok[2]] = int(labels_arr[idx])
        proba_map[tok[2]] = float(probas_arr[idx])
        idx += 1

n_bp    = sum(1 for v in label_map.values() if v == 0)
n_sb    = sum(1 for v in label_map.values() if v == 1)
n_total = len(label_map)

# ── Stats bar ─────────────────────────────────────────────────────────────────
sc1, sc2, sc3 = st.columns([1, 2, 2])
sc1.metric('Total', n_total)
sc2.metric('Boilerplate', f'{n_bp}  ({n_bp/n_total*100:.1f}%)')
sc3.metric('Substantive', f'{n_sb}  ({n_sb/n_total*100:.1f}%)')

bp_pct = n_bp / n_total * 100
st.markdown(
    f'<div style="height:6px;border-radius:3px;overflow:hidden;margin:2px 0 10px;">'
    f'<div style="width:{bp_pct:.1f}%;background:#e05252;height:100%;display:inline-block;"></div>'
    f'<div style="width:{100-bp_pct:.1f}%;background:#52a852;height:100%;display:inline-block;"></div>'
    f'</div>',
    unsafe_allow_html=True,
)

if 'view_mode' not in st.session_state:
    st.session_state.view_mode = 'All'

lc0, lc1, lc2, lc3, lc_src = st.columns([0.6, 1.4, 1.4, 1.6, 4])

with lc0:
    if st.button('All',
                 type='primary' if st.session_state.view_mode == 'All' else 'secondary',
                 use_container_width=True):
        st.session_state.view_mode = 'All'; st.rerun()

with lc1:
    if st.button('🔴 Boilerplate',
                 type='primary' if st.session_state.view_mode == 'BP only' else 'secondary',
                 use_container_width=True):
        st.session_state.view_mode = 'BP only'; st.rerun()

with lc2:
    if st.button('🟢 Substantive',
                 type='primary' if st.session_state.view_mode == 'Sub only' else 'secondary',
                 use_container_width=True):
        st.session_state.view_mode = 'Sub only'; st.rerun()

with lc3:
    if st.button('Short / filtered',
                 type='primary' if st.session_state.view_mode == 'Short only' else 'secondary',
                 use_container_width=True):
        st.session_state.view_mode = 'Short only'; st.rerun()

with lc_src:
    st.markdown(
        f'<p style="margin:8px 0 0;font-size:12px;color:#aaa;text-align:right;">'
        f'{st.session_state.source_label}</p>',
        unsafe_allow_html=True,
    )

view_mode = st.session_state.view_mode

# ── Document view ─────────────────────────────────────────────────────────────
bp_style    = 'background-color:#ffcccc;padding:1px 3px;border-radius:2px;'
short_style = 'color:#aaaaaa;font-style:italic;'

html_parts = ['<div style="font-family:sans-serif;font-size:14px;line-height:1.9;padding:4px;">']
pending_breaks = 0

for tok in tokens:
    if tok[0] == 'break':
        pending_breaks += 1
        continue

    _, keep, s = tok
    se = _html.escape(s)

    if pending_breaks > 0:
        html_parts.append('<br>' * min(pending_breaks, 2))
        pending_breaks = 0

    lbl   = label_map.get(s, 1)
    prob  = proba_map.get(s, 0.5)
    title = f'P(substantive)={prob:.3f}'

    if not keep:
        if view_mode in ('All', 'Short only'):
            html_parts.append(f'<span style="{short_style}" title="too short">{se}</span> ')
        continue

    if view_mode == 'Short only': continue
    if view_mode == 'BP only'  and lbl != 0: continue
    if view_mode == 'Sub only' and lbl != 1: continue

    if lbl == 0:
        html_parts.append(
            f'<span style="{bp_style}" title="{title}">'
            f'<sup style="font-size:9px;color:#c00;">BP</sup>{se}</span> '
        )
    else:
        html_parts.append(f'<span title="{title}">{se}</span> ')

html_parts.append('</div>')

est_height = max(500, len([t for t in tokens if t[0] == 'sentence']) * 26)
_components.html(''.join(html_parts), height=est_height, scrolling=True)

# ── Download ──────────────────────────────────────────────────────────────────
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
