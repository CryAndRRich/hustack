import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from preprocess import NUMERIC, TARGET, get_out_dir, load_data
from tune import fit_gam, fit_lgb

VERSION = "v1_3"
SEED = 42
CFG = {"max_rounds": 5000, "early_stop": 200, "final_lgb_folds": 5}
GAM_TUNED = {
    "knots_study_hours": 4, "knots_sleep_hours": 3, "knots_attendance_pct": 6,
    "knots_previous_score": 3, "knots_screen_time_hours": 6,
    "degree": 2, "knot_strategy": "quantile", "alpha": 0.09165846221016766,
}
LGB_TUNED = {
    "learning_rate": 0.029223219912235986, "num_leaves": 4, "max_depth": 2, "min_child_samples": 7,
    "subsample": 0.8925762353398154, "colsample_bytree": 0.4910499823553257,
    "reg_lambda": 6.348518508753555, "reg_alpha": 9.442864291879712,
    "min_split_gain": 0.09265738364673298, "path_smooth": 11.715768212554652,
    "max_bin": 255, "linear_tree": True, "linear_lambda": 0.0874464949822191,
}
W_LGB = 0.033
EXTRAPOLATION = "constant"


def lgb_test(X, y, X_test):
    preds = [fit_lgb(LGB_TUNED, X.iloc[tr], y[tr], CFG, SEED + k).predict(X_test)
             for k, (tr, _) in enumerate(KFold(CFG["final_lgb_folds"], shuffle=True, random_state=SEED).split(X))]
    return np.mean(preds, axis=0)


def main():
    t0 = time.time()
    X, y, X_test, train_ids, sample = load_data()

    outside = pd.Series(False, index=X_test.index)
    print("\ntest rows outside the train range:")
    for c in NUMERIC:
        lo, hi = X[c].min(), X[c].max()
        mask = (X_test[c] < lo) | (X_test[c] > hi)
        outside |= mask
        print(f"  {c:18s} train [{lo}, {hi}]  test outside: {int(mask.sum())}")

    gam_new = fit_gam(dict(GAM_TUNED, extrapolation=EXTRAPOLATION), X, y).predict(X_test)
    gam_old = fit_gam(dict(GAM_TUNED, extrapolation="linear"), X, y).predict(X_test)
    lgb = lgb_test(X, y, X_test)
    blend_new = gam_new + W_LGB * (lgb - gam_new)
    blend_old = gam_old + W_LGB * (lgb - gam_old)

    d = blend_new - blend_old
    assert np.allclose(d[~outside.to_numpy()], 0, atol=1e-9), "in-range predictions changed"
    print(f"\nchanged rows: {int((np.abs(d) > 1e-9).sum())} (expected = rows outside the range above)")
    show = X_test.loc[outside, NUMERIC].assign(id=sample["id"].to_numpy()[outside.to_numpy()])
    show["V1.2 blend (linear)"] = blend_old[outside.to_numpy()]
    show["V1.3 (constant)"] = blend_new[outside.to_numpy()]
    show["lgb_tuned"] = lgb[outside.to_numpy()]
    print(show.round(2).to_string(index=False))

    pred = np.clip(blend_new, 0, 100)
    assert len(pred) == len(sample) and np.isfinite(pred).all()
    sub = pd.DataFrame({"id": sample["id"].to_numpy(), TARGET: pred})
    assert list(sub.columns) == list(sample.columns)
    out = get_out_dir()
    sub.to_csv(out / "submission.csv", index=False)
    print(f"\ntest pred: mean {pred.mean():.3f}  sd {pred.std():.3f}  min {pred.min():.2f}  max {pred.max():.2f}")
    print(f"total {time.time() - t0:.0f}s")
    print("\n" + "=" * 60)
    print(f"SUBMIT: {out / 'submission.csv'}  ({VERSION}: V1.2 blend, {EXTRAPOLATION} extrapolation)")
    print("=" * 60)


if __name__ == "__main__":
    main()
