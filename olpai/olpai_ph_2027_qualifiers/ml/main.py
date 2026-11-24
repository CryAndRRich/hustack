import csv

from common.runtime import find_folder, run_session
from ml.config import Config


def has_columns(cfg, folder):
    if not (folder / "test.csv").exists():
        return False
    with open(folder / "train.csv", encoding="utf-8") as f:
        header = next(csv.reader(f))
    return set(cfg.features) | {cfg.label} <= set(header)


def read_rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def predict(cfg, train, test):
    positives = {tuple(r[c] for c in cfg.features) for r in train if r[cfg.label] == "1"}
    print(f"train rows {len(train)}, positive tuples {sorted(positives)}")
    return ["1" if tuple(r[c] for c in cfg.features) in positives else "0" for r in test]


def check_submission(cfg, test, sample, labels):
    ids = [r[cfg.id_col] for r in test]
    assert len(ids) == cfg.n_test == len(sample), (len(ids), len(sample))
    assert ids == [r[cfg.id_col] for r in sample], "id order differs from sample_submission.csv"
    assert all(len(i) == 6 and i.isdigit() for i in ids), "ids must stay 6-digit strings"
    assert set(labels) <= {"0", "1"}
    share = labels.count("1") / len(labels)
    assert share <= cfg.max_positive_share, f"positive share {share} contradicts the all-1 sample's 0.00000"
    return ids


def write_submission(path, cfg, ids, labels):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow([cfg.id_col, cfg.label])
        w.writerows(zip(ids, labels))
    with open(path, encoding="utf-8") as f:
        assert next(f).strip() == f"{cfg.id_col},{cfg.label}"
        assert sum(1 for _ in f) == len(ids)


def solve(cfg, work, submission):
    folder = find_folder(cfg, "train.csv", lambda d: has_columns(cfg, d))
    print(f"data folder {folder}")
    train = read_rows(folder / "train.csv")
    test = read_rows(folder / "test.csv")
    sample = read_rows(folder / "sample_submission.csv")
    labels = predict(cfg, train, test)
    ids = check_submission(cfg, test, sample, labels)
    positive_ids = [i for i, y in zip(ids, labels) if y == "1"]
    print(f"test rows {len(ids)}, predicted positives {len(positive_ids)}: {positive_ids}")
    write_submission(submission, cfg, ids, labels)
    return {"n_test": len(ids), "n_positive": len(positive_ids), "positive_ids": positive_ids}


def run(cfg=None):
    run_session(cfg or Config(), solve)


if __name__ == "__main__":
    run()
