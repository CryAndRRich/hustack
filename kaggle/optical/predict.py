import numpy as np
import pandas as pd

import runtime
from rule import N_CLASSES, vote
from runtime import log

VAL_FLOOR = 0.925


def write_submission(path, sample, test, pred):
    sub = sample[["id"]].copy()
    sub["profile"] = sub["id"].map(pd.Series(pred, index=test["id"].to_numpy()))
    assert len(sub) == len(sample) and sub["profile"].notna().all()
    sub["profile"] = sub["profile"].astype(int)
    assert set(sub["profile"]) <= set(range(N_CLASSES))
    sub.to_csv(path, index=False)


def run():
    try:
        root = runtime.data_root()
        train, val, test, sample = (pd.read_csv(root / f"{n}.csv")
                                    for n in ("train", "validation", "test", "sample_submission"))
        log("data", root, "train", train.shape, "val", val.shape, "test", test.shape)
        assert set(sample["id"]) == set(test["id"])

        pred, share = vote(train, val)
        ok = pred == val["profile"].to_numpy()
        acc = float(ok.mean())
        log("VAL accuracy %.4f (floor %.3f, expected 0.9304)" % (acc, VAL_FLOOR))
        per_brand = {b: float(ok[(val["cartridge"] == b).to_numpy()].mean()) for b in sorted(val["cartridge"].unique())}
        log("VAL per cartridge", {b: round(v, 4) for b, v in per_brand.items()})
        conf = pd.crosstab(val["profile"], pred, rownames=["true"], colnames=["pred"])
        log("VAL confusion\n" + conf.to_string())
        bins = np.quantile(share, [0, 0.25, 0.5, 0.75, 1])
        log("VAL accuracy by vote-share quartile",
            [round(float(ok[(share >= lo) & (share <= hi)].mean()), 4) for lo, hi in zip(bins[:-1], bins[1:])])
        pd.DataFrame({"id": val["id"], "profile": val["profile"], "pred": pred, "share": share}).to_csv(
            runtime.WORK / "val_predictions.csv", index=False)
        if acc < VAL_FLOOR:
            raise RuntimeError("val accuracy %.4f below floor %.3f: rule not reproduced, no submission" % (acc, VAL_FLOOR))

        test_pred, test_share = vote(train, test)
        log("TEST class counts", np.bincount(test_pred, minlength=N_CLASSES).tolist(),
            "mean vote share %.3f (val %.3f)" % (test_share.mean(), share.mean()))
        extra = runtime.WORK / "extra"
        extra.mkdir(parents=True, exist_ok=True)
        alt_pred, _ = vote(pd.concat([train, val], ignore_index=True), test)
        write_submission(extra / "submission_archive_train_val.csv", sample, test, alt_pred)
        log("extra archive=train+val: agrees with main on %.4f of test" % (alt_pred == test_pred).mean())

        sub = runtime.out_dir() / "submission.csv"
        write_submission(sub, sample, test, test_pred)
        runtime.save_json({"val_accuracy": acc, "per_cartridge": per_brand,
                           "test_counts": np.bincount(test_pred, minlength=N_CLASSES).tolist()}, "result.json")
        log("SUBMIT:", sub, "(V1 recovered rule: per-cartridge 9-NN, Chebyshev, passport scale, 1/d)")
    finally:
        runtime.save_code()
        runtime.zip_results()


if __name__ == "__main__":
    run()
