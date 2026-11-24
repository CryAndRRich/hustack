import numpy as np
import pandas as pd
import moff
from config import SEED_START, SPACER_LEN
NN_DG = {'AA': -1.0, 'AC': -2.1, 'AG': -1.8, 'AT': -0.9, 'CA': -0.9, 'CC': -2.1, 'CG': -1.7, 'CT': -0.9, 'GA': -1.3, 'GC': -2.7, 'GG': -2.9, 'GT': -1.1, 'TA': -0.6, 'TC': -1.5, 'TG': -1.6, 'TT': -0.2}
DG_INIT = 3.1
PURINE = set('AG')
SEQUENCE_FEATURES = ['mm_count', 'mm_pos_min', 'mm_pos_max', 'mm_span', 'mm_adjacent', 'mm_seed_count', 'mm_distal_count', 'n_wobble', 'n_transition', 'dg_hybrid', 'dg_seed', 'gc_candidate', 'log_distance_factor', 'log_moff_mde', 'log_moff_ce', 'pam_ngg', 'pam_nag']

def mismatch_positions(spacer, candidate):
    return [i for i in range(SPACER_LEN) if spacer[i] != candidate[i]]

def hybrid_dg(spacer, mismatch_set, start=0):
    total = DG_INIT if start == 0 else 0.0
    for i in range(start, SPACER_LEN - 1):
        if i in mismatch_set or i + 1 in mismatch_set:
            continue
        total += NN_DG[spacer[i:i + 2]]
    return total

def distance_factor(positions):
    if len(positions) < 2:
        return 1.0
    gaps = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
    mean_gap = sum(gaps) / len(gaps)
    return 1.0 / ((19.0 - mean_gap) / 19.0 * 4.0 + 1.0)

def build(df):
    rows = []
    for spacer, candidate in zip(df['spacer'].values, df['candidate_spacer20'].values):
        positions = mismatch_positions(spacer, candidate)
        mismatch_set = set(positions)
        n = len(positions)
        adjacent = sum((1 for i in range(n - 1) if positions[i + 1] - positions[i] == 1))
        wobble = sum((1 for i in positions if spacer[i] == 'G' and candidate[i] == 'A'))
        transition = sum((1 for i in positions if (spacer[i] in PURINE) == (candidate[i] in PURINE)))
        rows.append((float(n), float(positions[0]) if n else float(SPACER_LEN), float(positions[-1]) if n else -1.0, float(positions[-1] - positions[0]) if n > 1 else 0.0, float(adjacent), float(sum((1 for i in positions if i >= SEED_START))), float(sum((1 for i in positions if i < 4))), float(wobble), float(transition), hybrid_dg(spacer, mismatch_set, 0), hybrid_dg(spacer, mismatch_set, SEED_START), (candidate.count('G') + candidate.count('C')) / float(SPACER_LEN), np.log(distance_factor(positions))))
    out = pd.DataFrame(rows, columns=['mm_count', 'mm_pos_min', 'mm_pos_max', 'mm_span', 'mm_adjacent', 'mm_seed_count', 'mm_distal_count', 'n_wobble', 'n_transition', 'dg_hybrid', 'dg_seed', 'gc_candidate', 'log_distance_factor'], index=df.index)
    for name, values in moff.columns(df['spacer'].values, df['candidate_spacer20'].values).items():
        out[name] = values
    pam = df['pam'].str[1:].values
    out['pam_ngg'] = (pam == 'GG').astype(float)
    out['pam_nag'] = (pam == 'AG').astype(float)
    if 'published_cfd' in df.columns:
        out['cfd'] = df['published_cfd'].values.astype(float)
        out['log_cfd'] = np.log(df['published_cfd'].values + 1e-06)
        out['distance_factor'] = np.exp(out['log_distance_factor'].values)
    assert np.isfinite(out.values).all()
    return out
