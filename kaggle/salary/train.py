from pathlib import Path

import numpy as np
import pandas as pd

from models import MEMBERS, WEIGHTS, low_rule, v1
from cv import run, splits
from preprocess import ID, TARGET, distinct_test, load

VERSION = "v1.11"


def main():
    out = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(__file__).resolve().parent.parent
    (out / "extra").mkdir(parents=True, exist_ok=True)
    tr, te, ss = load()

    res = run({"v1": v1}, tr, splits(tr))
    pd.DataFrame({ID: tr[ID], TARGET: tr[TARGET], "v1": res["v1"]["P"].mean(axis=0)}).to_csv(
        out / "extra" / f"oof_{VERSION}.csv", index=False)

    preds = {k: MEMBERS[k](tr, te) for k in WEIGHTS}
    p = sum(WEIGHTS[k] * preds[k] for k in WEIGHTS)
    first, key = distinct_test(te)
    for k, q in {**preds, "final": p}.items():
        assert np.isfinite(q).all() and pd.Series(q).round(3).groupby(key).nunique().max() == 1, f"{k}: non-finite or copies disagree"
    sub = pd.DataFrame({ID: te[ID].values, TARGET: p})
    assert list(sub.columns) == list(ss.columns) and len(sub) == len(ss) and (sub[ID].values == ss[ID].values).all()
    assert sub[TARGET].between(5e5, 3e6).all()
    sub.to_csv(out / "submission.csv", index=False)
    pd.DataFrame({ID: te[ID].values, **preds}).to_csv(out / "extra" / f"test_members_{VERSION}.csv", index=False)

    print(f"\nweights {WEIGHTS}; V1 CV rmse {res['v1']['rmse']:.0f}; test flagged low: {int(low_rule(te)[first].sum())} of {int(first.sum())} distinct rows")
    print(f"SUBMIT: {out / 'submission.csv'}")


if __name__ == "__main__":
    main()
