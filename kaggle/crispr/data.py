import os
import numpy as np
import pandas as pd
from config import GUIDE_LEN, SPACER_LEN
from runtime import find_data

def _check(df, labelled):
    assert (df['spacer'] == df['guide'].str[:SPACER_LEN]).all()
    assert (df['candidate_spacer20'] == df['candidate_ot23'].str[:SPACER_LEN]).all()
    assert (df['pam'] == df['candidate_ot23'].str[-3:]).all()
    assert (df['guide'].str.len() == GUIDE_LEN).all()
    assert (df['candidate_ot23'].str.len() == GUIDE_LEN).all()
    assert df['published_cfd'].notna().all()
    hamming = [sum((a != b for a, b in zip(s, c))) for s, c in zip(df['candidate_spacer20'], df['spacer'])]
    assert (np.array(hamming) == df['mismatch_count'].values).all()
    if labelled:
        assert (df['label'].values == (df['read_fraction'].values > 0).astype(int)).all()

def load_external():
    roots = ['/kaggle/input', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'), '.']
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            if 'ext_changeseq.csv' in filenames:
                df = pd.read_csv(os.path.join(dirpath, 'ext_changeseq.csv'))
                assert set(['guide', 'spacer', 'candidate_spacer20', 'pam', 'mismatch_count', 'reads']).issubset(df.columns)
                assert (df['reads'] > 0).all()
                return df
    return None

def load(smoke=False):
    root = find_data()
    train = pd.read_csv(os.path.join(root, 'train.csv'))
    test = pd.read_csv(os.path.join(root, 'test.csv'))
    sample = pd.read_csv(os.path.join(root, 'sample_submission.csv'))
    _check(train, True)
    _check(test, False)
    assert set(train['guide']).isdisjoint(set(test['guide']))
    assert list(sample['id'].values) == list(test['id'].values)
    if smoke:
        keep = train['guide'].drop_duplicates().head(4)
        train = train[train['guide'].isin(keep)].reset_index(drop=True)
        test = test.head(500).reset_index(drop=True)
        sample = sample.head(500).reset_index(drop=True)
    return (train, test, sample)
