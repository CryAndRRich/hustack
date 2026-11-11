from pathlib import Path

import pandas as pd

ID, TARGET = "id", "Maintainability"
TOTAL = "CodeFactor Issue-Issue"
PARTS = ["CodeFactor Issue-Complexitiy", "CodeFactor Issue-Style", "CodeFactor Issue-Security"]


def find(name):
    root = Path("/kaggle/input")
    hits = sorted(root.rglob(name)) if root.exists() else []
    if hits:
        return hits[0].parent
    try:
        return Path(__file__).resolve().parent.parent / "data"
    except NameError:
        return Path.cwd() / "data"


def identity(df):
    return df[TOTAL] - df[PARTS].sum(axis=1)


if __name__ == "__main__":
    d = find("submission-example.csv")
    tr, te, ex = (pd.read_csv(d / f) for f in ("train.csv", "test.csv", "submission-example.csv"))
    print(f"train {tr.shape}, test {te.shape}")
    assert TARGET not in te and te[ID].is_unique and not tr[ID].isin(te[ID]).any()
    assert te[ID].tolist() == ex[ID].tolist()

    resid = tr[TARGET] - identity(tr)
    print(f"train identity residual: max |r| {resid.abs().max()}, rows off {(resid != 0).sum()} / {len(tr)}")
    assert (resid == 0).all(), tr.loc[resid != 0, [ID, "Framework", TARGET]]

    pred = identity(te)
    assert pred.notna().all() and (pred >= 0).all(), te.loc[~(pred >= 0), [ID, "Framework"]]
    sub = pd.DataFrame({ID: te[ID], TARGET: pred})
    out = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
    sub.to_csv(out / "submission.csv", index=False)
    print(pd.concat([te[[ID, "Framework"]], sub[TARGET]], axis=1).to_string(index=False))
    print(f"SUBMIT: submission.csv ({len(sub)} rows, Maintainability = Issue - Complexity - Style - Security)")
