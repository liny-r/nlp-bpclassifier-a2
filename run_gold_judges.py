"""
run_gold_judges.py — Reproduce all 5 LLM gold-label judges.

Usage:
    python run_gold_judges.py           # run all judges
    python run_gold_judges.py --smoke   # smoke-test only (no full run)

Judges (all local Ollama — no API keys needed):
    j3  cogito:8b           judge3_cogito.parquet
    j4  qwen3:14b           judge4_qwen314b.parquet
    j5  gemma3:12b          judge5_gemma12b.parquet
    j6  ministral-3:8b      judge6_ministral3.parquet
    j7  cogito:14b          judge7_cogito14b.parquet

Removed judges (manual review showed systematic disagreement with ground truth):
    j1  qwen3:8b   — removed: over-flagged boilerplate (~29% BP, inconsistent with human audit)
    j2  gemma3:4b  — removed: severe BP bias (~48% BP), overridden 746/2500 times by majority

Majority vote: ≥ 3/5 judges agree.
Checkpoints every CHECKPOINT_EVERY sentences — safe to interrupt and resume.
"""

import os, sys, time, threading, requests
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent
CACHE    = ROOT / 'cache'
GOLD_DIR = CACHE / 'gold'
GOLD_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED      = 42
GOLD_N           = 2500
CHECKPOINT_EVERY = 50
OLLAMA_HOST      = 'http://localhost:11434'
OLLAMA_TIMEOUT   = 45


# ── Shared prompt ──────────────────────────────────────────────────────────────
LABEL_PROMPT = """\
You are labeling sentences from earnings-call transcripts.

Classify the sentence below as EXACTLY one of:
  boilerplate — scripted, generic, no material information (operator phrases,
                safe-harbor disclaimers, generic thanks, analyst name/firm
                introductions, housekeeping, "please hold", etc.)
  substantive — material content: financial figures, guidance, segment results,
                strategy, product commentary, specific risks, analyst questions
                about financials.

Edge cases:
- Analyst name intro lines → boilerplate
- "Thank you" alone or with filler → boilerplate
- Safe-harbor even if it mentions metrics → boilerplate
- Short affirmations ("Sure.", "Great.") → boilerplate
- Sentence with a dollar amount or percentage AND real context → substantive

Respond with ONLY one word: boilerplate or substantive

Sentence: {sentence}"""

SMOKE_CASES = [
    ("Revenue was $26 billion, up 18% sequentially.", 1),
    ("My name is Regina and I will be your conference operator.", 0),
    ("Thank you for joining us today.", 0),
    ("We expect full-year EPS of $12 to $13.", 1),
]


def _parse(raw):
    r = raw.strip().lower()
    if 'boilerplate' in r: return 0
    if 'substantive'  in r: return 1
    return None


# ── Judge functions ────────────────────────────────────────────────────────────

def ollama_judge(sentence, model):
    try:
        r = requests.post(f'{OLLAMA_HOST}/api/generate', timeout=OLLAMA_TIMEOUT,
            json={'model': model,
                  'prompt': LABEL_PROMPT.format(sentence=sentence[:500]),
                  'stream': False, 'think': False,
                  'options': {'think': False, 'temperature': 0.0,
                              'num_predict': 16, 'num_ctx': 2048}})
        r.raise_for_status()
        return _parse(r.json()['response'])
    except Exception:
        return None



# ── Sampling ───────────────────────────────────────────────────────────────────

def stratified_sample(df, n, seed=RANDOM_SEED):
    groups  = [g for _, g in df.groupby('speaker_type')]
    weights = np.array([len(g) for g in groups], dtype=float) / len(df)
    counts  = np.round(weights * n).astype(int)
    counts[-1] += n - counts.sum()
    parts = [g.sample(min(c, len(g)), random_state=seed)
             for g, c in zip(groups, counts)]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


# ── Per-judge runner with checkpointing ───────────────────────────────────────

_print_lock = threading.Lock()

def _log(msg):
    with _print_lock:
        print(msg, flush=True)


def run_judge(gold_sample, judge_fn, name, cache_path, force=False):
    ckpt = GOLD_DIR / f'{name}_checkpoint.csv'

    if cache_path.exists() and not force:
        cached = pd.read_parquet(cache_path)
        n_fail = (cached['label'] == -1).sum()
        _log(f'[{name}] already cached ({len(cached)} rows, {n_fail} failures) — skip')
        return cached['label']

    done = {}
    if ckpt.exists() and not force:
        cp = pd.read_csv(ckpt)
        done = dict(zip(cp['text'], cp['label']))
        _log(f'[{name}] resuming from checkpoint: {len(done)} done')

    texts   = gold_sample['text'].tolist()
    pending = [(i, t) for i, t in enumerate(texts) if t not in done]
    buf     = []

    with tqdm(pending, desc=f'[{name}]', unit='sent', dynamic_ncols=True, leave=True) as pbar:
        for _, text in pbar:
            lbl    = judge_fn(text)
            result = lbl if lbl is not None else -1
            done[text] = result
            buf.append({'text': text, 'label': result})
            if len(buf) >= CHECKPOINT_EVERY:
                pd.DataFrame(buf).to_csv(
                    ckpt, mode='a', header=not ckpt.exists(), index=False)
                buf = []

    if buf:
        pd.DataFrame(buf).to_csv(
            ckpt, mode='a', header=not ckpt.exists(), index=False)

    final = [done[t] for t in texts]
    s = pd.Series(final, name='label')
    pd.DataFrame({'label': s}).to_parquet(cache_path)
    n_fail = (s == -1).sum()
    _log(f'[{name}] DONE — {n_fail}/{len(s)} failures ({n_fail/len(s)*100:.1f}%)')
    if ckpt.exists():
        ckpt.unlink()
    return s


# ── Smoke test ─────────────────────────────────────────────────────────────────

def smoke_test(judges):
    print('\n=== Smoke test (4 calibration sentences) ===')
    header = f"  {'Judge':<16} {'1 SB$26B':>10} {'2 BPoper':>10} {'3 BPthx':>10} {'4 SBeps':>10}  Result"
    print(header)
    print('  ' + '-' * (len(header) - 2))

    all_pass = True
    for name, fn, _, avail in judges:
        if not avail:
            print(f'  — {name:<16} SKIP (unavailable)')
            continue
        preds = []
        for sent, expected in SMOKE_CASES:
            got = fn(sent)
            preds.append(('✓' if got == expected else '✗') +
                         ('SB' if got == 1 else ('BP' if got == 0 else '??')))
        ok = all(p.startswith('✓') for p in preds)
        icon = '✓' if ok else '✗'
        if not ok:
            all_pass = False
        print(f'  {icon} {name:<16} ' + '  '.join(f'{p:>10}' for p in preds) +
              f'  {"PASS" if ok else "FAIL"}')

    print()
    return all_pass


# ── Majority vote ──────────────────────────────────────────────────────────────

def majority_vote(votes):
    valid = [v for v in votes if v != -1]
    if not valid:
        return -1
    return Counter(valid).most_common(1)[0][0]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    smoke_only = '--smoke' in sys.argv

    pool_path = CACHE / 'sentence_pool.parquet'
    if not pool_path.exists():
        sys.exit('sentence_pool.parquet not found — run §2 in the notebook first.')
    sentence_pool = pd.read_parquet(pool_path)
    gold_sample   = stratified_sample(sentence_pool, GOLD_N)
    print(f'Loaded pool: {len(sentence_pool):,} sentences')
    print(f'Gold sample: {len(gold_sample)} sentences (seed={RANDOM_SEED})\n')

    judges = [
        ('j3_cogito',     lambda t: ollama_judge(t, 'cogito:latest'),     GOLD_DIR / 'judge3_cogito.parquet',      True),  # cogito:8b
        ('j4_qwen314b',   lambda t: ollama_judge(t, 'qwen3:14b'),         GOLD_DIR / 'judge4_qwen314b.parquet',    True),
        ('j5_gemma12b',   lambda t: ollama_judge(t, 'gemma3:12b'),        GOLD_DIR / 'judge5_gemma12b.parquet',    True),
        ('j6_ministral3', lambda t: ollama_judge(t, 'ministral-3:latest'), GOLD_DIR / 'judge6_ministral3.parquet', True),  # ministral-3:8b
        ('j7_cogito14b',  lambda t: ollama_judge(t, 'cogito:14b'),        GOLD_DIR / 'judge7_cogito14b.parquet',   True),
    ]

    print('Judges:')
    for name, _, _, _ in judges:
        print(f'  ✓ {name}')
    print()

    ok = smoke_test(judges)
    if smoke_only:
        sys.exit(0 if ok else 1)
    if not ok:
        print('WARNING: some judges failed smoke test — continuing anyway.\n')

    results = {}

    t_start = time.time()
    print('Running all judges sequentially...\n')
    for name, fn, cache, _ in judges:
        t0 = time.time()
        s  = run_judge(gold_sample, fn, name, cache, force=False)
        results[name] = s
        print(f'[{name}] finished in {(time.time()-t0)/60:.1f} min')

    total = (time.time() - t_start) / 60
    print(f'\nAll judges finished in {total:.1f} min')

    if not results:
        print('No judges completed. Check Ollama is running.')
        return

    print('\n=== Judge summary ===')
    for name, s in results.items():
        vc = s.value_counts().to_dict()
        print(f'  {name:<16}  BP={vc.get(0,0):4d}  SB={vc.get(1,0):4d}  '
              f'fail={vc.get(-1,0):4d}')

    judge_names = list(results.keys())
    gold = gold_sample.copy()
    for name, s in results.items():
        gold[name] = s.values

    gold['label_mv'] = gold[judge_names].apply(
        lambda r: majority_vote([int(r[j]) for j in judge_names]), axis=1)

    dropped = (gold['label_mv'] == -1).sum()
    gold    = gold[gold['label_mv'] != -1].reset_index(drop=True)

    unanimous = gold[judge_names].apply(
        lambda r: len({r[j] for j in judge_names if r[j] != -1}) == 1, axis=1)

    print(f'\nGold set: {len(gold)} rows  (dropped {dropped} with all failures)')
    print(f'Unanimous: {unanimous.sum()} ({unanimous.mean()*100:.1f}%)')
    print('Label balance:')
    print(gold['label_mv'].value_counts()
          .rename({0:'boilerplate', 1:'substantive'}).to_string())

    keep = (['ticker','quarter','call_date','section','speaker_type','text','label_mv']
            + judge_names)
    out  = gold[keep].rename(columns={'label_mv': 'label'})
    out.to_parquet(GOLD_DIR / 'gold_labels.parquet', index=False)
    print(f'\nSaved gold_labels.parquet ({len(out)} rows, judges: {judge_names})')


if __name__ == '__main__':
    main()
