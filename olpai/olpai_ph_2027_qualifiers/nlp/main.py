import time


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.runtime import find_folder, run_session
from nlp.config import Config
from nlp.scoring import load_model, score_texts

FEATURES = ["z", "ll_per_token", "rank_frac", "start_word", "end_function_word"]


def best_threshold(z, y):
    cands = np.unique(z)
    cuts = np.concatenate([[cands[0] - 1], (cands[:-1] + cands[1:]) / 2, [cands[-1] + 1]])
    acc = [((z < c).astype(int) == y).mean() for c in cuts]
    return float(cuts[int(np.argmax(acc))]), float(max(acc))


def logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))


def cross_validate(cfg, feats, y):
    folds = StratifiedKFold(cfg.n_folds, shuffle=True, random_state=cfg.seed)
    oof = {"threshold": np.zeros(len(y), int), "logreg": np.zeros(len(y), int)}
    for tr, va in folds.split(feats, y):
        cut, _ = best_threshold(feats["z"].values[tr], y[tr])
        oof["threshold"][va] = (feats["z"].values[va] < cut).astype(int)
        model = logreg().fit(feats[FEATURES].values[tr], y[tr])
        oof["logreg"][va] = model.predict(feats[FEATURES].values[va])
    return {k: float((v == y).mean()) for k, v in oof.items()}, oof


def write_submission(path, ids, labels, n_test):
    sub = pd.DataFrame({"id": ids, "label": labels.astype(int)})
    assert len(sub) == n_test and sub["id"].is_unique and set(sub["label"]) <= {0, 1}
    sub.to_csv(path, index=False)
    back = pd.read_csv(path, dtype=str)
    assert list(back.columns) == ["id", "label"] and back["id"].tolist() == list(ids)


def has_columns(folder):
    return (folder / "test.csv").exists() and \
        {"text", "label"} <= set(pd.read_csv(folder / "train.csv", nrows=1).columns) and \
        {"id", "text"} <= set(pd.read_csv(folder / "test.csv", nrows=1).columns)


def solve(cfg, work, submission):
    folder = find_folder(cfg, "train.csv", has_columns)
    train = pd.read_csv(folder / "train.csv", dtype={"text": str})
    test = pd.read_csv(folder / "test.csv", dtype=str)
    sample = pd.read_csv(folder / "sample_submission.csv", dtype=str)
    assert len(test) == cfg.n_test and test["id"].tolist() == sample["id"].tolist()
    y = train["label"].astype(int).values
    print(f"data {folder}: train {len(train)} (shuffled {y.sum()}), test {len(test)}")

    tok, model, device, start_id = load_model(cfg)
    t0 = time.time()
    rows = score_texts(cfg, train["text"].tolist() + test["text"].tolist(), tok, model, device, start_id)
    print(f"scored {len(rows)} texts x {cfg.n_shuffles + 1} orders in {time.time() - t0:.0f}s")
    feats = pd.DataFrame(rows)
    tr_f, te_f = feats.iloc[:len(train)].reset_index(drop=True), feats.iloc[len(train):].reset_index(drop=True)
    pd.concat([tr_f.assign(split="train", label=y, text=train["text"]),
               te_f.assign(split="test", id=test["id"].values, text=test["text"].values)]
              ).to_csv(work / "extra" / "scores.csv", index=False)

    cv, oof = cross_validate(cfg, tr_f, y)
    print("cv accuracy", cv)
    for arm, pred in oof.items():
        wrong = train.loc[pred != y, ["text", "label"]].assign(z=tr_f["z"][pred != y])
        print(f"{arm}: {len(wrong)} OOF errors\n{wrong.to_string()}")

    cut, train_acc = best_threshold(tr_f["z"].values, y)
    arms = {"threshold": (te_f["z"].values < cut).astype(int),
            "logreg": logreg().fit(tr_f[FEATURES].values, y).predict(te_f[FEATURES].values)}
    for arm, labels in arms.items():
        write_submission(work / "extra" / f"submission_{arm}.csv", test["id"].values, labels, cfg.n_test)
    pick = "logreg" if cv["logreg"] > cv["threshold"] else "threshold"
    labels = arms[pick]
    print(f"threshold z < {cut:.3f} (train acc {train_acc:.4f}); pick {pick}; "
          f"test shuffled share {labels.mean():.3f}; arms agree on {(arms['threshold'] == arms['logreg']).mean():.3f}")
    write_submission(submission, test["id"].values, labels, cfg.n_test)
    return {"cv": cv, "pick": pick, "threshold": cut, "train_acc_threshold": train_acc,
            "test_shuffled_share": float(labels.mean())}


def run(cfg=None):
    run_session(cfg or Config(), solve)


if __name__ == "__main__":
    run()
