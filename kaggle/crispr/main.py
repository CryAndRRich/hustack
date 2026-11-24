import json
import os
import numpy as np
import pandas as pd
import features
import transfer
from arms import run_all
from config import ARMS, BAR, BAR_OOF_AUROC, BAR_OOF_TOL, CANDIDATE, GATE_MIN_WIN_FRACTION, REFERENCE
from data import load, load_external
from evaluate import auroc, guide_bootstrap_wins, report
from runtime import WORK, log, out_dir, start_logging, write_results_zip

def per_guide_table(train, y, oof):
    rows = []
    for g in np.unique(train['guide'].values):
        held = train['guide'].values == g
        row = {'guide': g, 'n': int(held.sum()), 'pos': int(y[held].sum())}
        for name in ARMS:
            row[name] = auroc(y[held], oof[name][held])
        rows.append(row)
    return pd.DataFrame(rows).sort_values('pos', ascending=False)

def pick(summary):
    if summary[CANDIDATE]['wins_vs_bar'] >= GATE_MIN_WIN_FRACTION and summary[CANDIDATE]['auroc'] > summary[BAR]['auroc']:
        return CANDIDATE
    return BAR

def run(smoke=False):
    train, test, sample = load(smoke=smoke)
    external = load_external()
    log('train %s test %s positives %d guides %d/%d' % (train.shape, test.shape, int(train['label'].sum()), train['guide'].nunique(), test['guide'].nunique()))
    train_feat = features.build(train)
    test_feat = features.build(test)
    if external is None:
        log('external table missing: tx_* set to 0, logit_v2 collapses to logit_moff')
        for feat in (train_feat, test_feat):
            feat['tx_ridge'] = 0.0
            feat['tx_gbdt'] = 0.0
    else:
        models = transfer.fit(external, smoke=smoke)
        for feat in (train_feat, test_feat):
            for name, values in transfer.apply(models, feat).items():
                feat[name] = values
    y = train['label'].values.astype(int)
    groups = train['guide'].values
    oof, test_scores, meta = run_all(train_feat, y, groups, test_feat)
    summary = {}
    for name in ARMS:
        summary[name] = report(y, oof[name])
        summary[name]['wins_vs_bar'] = 1.0 if name == BAR else guide_bootstrap_wins(y, groups, oof[name], oof[BAR])
        log('%-12s LOGO auroc %.5f  p50 %.3f r50 %.3f r100 %.3f  wins_vs_%s %.3f' % (name, summary[name]['auroc'], summary[name]['p50'], summary[name]['r50'], summary[name]['r100'], BAR, summary[name]['wins_vs_bar']))
    assert summary[REFERENCE]['auroc'] > 0.9, 'CFD reference below its known level'
    if not smoke and external is not None:
        delta = abs(summary[BAR]['auroc'] - BAR_OOF_AUROC)
        assert delta < BAR_OOF_TOL, "bar OOF moved by %.5f from V1's %.5f - the bar is not reproduced" % (delta, BAR_OOF_AUROC)
    table = per_guide_table(train, y, oof)
    log('per-guide AUROC\n%s' % table.to_string(index=False))
    chosen = pick(summary)
    log('chosen arm: %s (candidate %s, bar %s)' % (chosen, CANDIDATE, BAR))
    tag = 'smoke_' if smoke else ''
    pd.DataFrame(oof).assign(id=train['id'].values, guide=groups, label=y).to_csv(os.path.join(WORK, 'extra', tag + 'oof.csv'), index=False)
    pd.DataFrame(test_scores).assign(id=test['id'].values).to_csv(os.path.join(WORK, 'extra', tag + 'test_arms.csv'), index=False)
    table.to_csv(os.path.join(WORK, 'extra', tag + 'per_guide.csv'), index=False)
    with open(os.path.join(WORK, 'extra', tag + 'result.json'), 'w') as f:
        json.dump({'summary': summary, 'meta': meta, 'chosen': chosen, 'smoke': smoke, 'external': external is not None}, f, indent=2)
    for name in ARMS:
        probe = pd.DataFrame({'id': test['id'].values, 'score': test_scores[name]})
        probe.to_csv(os.path.join(WORK, 'extra', tag + 'submission_%s.csv' % name), index=False)
    sub = pd.DataFrame({'id': test['id'].values, 'score': test_scores[chosen]})
    assert list(sub['id'].values) == list(sample['id'].values)
    assert len(sub) == len(sample)
    assert np.isfinite(sub['score'].values).all()
    path = os.path.join(WORK, 'extra', 'smoke_submission.csv') if smoke else os.path.join(out_dir(), 'submission.csv')
    sub.to_csv(path, index=False)
    log('SUBMIT: %s' % path)
    return summary

if __name__ == '__main__':
    tee = start_logging()
    try:
        run(smoke=True)
        run(smoke=False)
    finally:
        import sys
        sys.stdout = tee.stdout
        tee.flush()
        print('results: %s' % write_results_zip())
