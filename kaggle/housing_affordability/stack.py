import json
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess import (
    ID,
    STORE_DIR,
    MONTH,
    TARGET,
    TRAIN_MONTHS,
    get_output_dir,
    oof_fingerprint,
    oof_folds,
)

STORE_DIRNAME = "family_store"


def store_dir() -> Path:

    d = get_output_dir() / STORE_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _find_stored(family: str = None) -> list[Path]:

    pattern = f"{family}.json" if family else "*.json"
    found = []
    for base in (get_output_dir() / STORE_DIRNAME, STORE_DIR / STORE_DIRNAME):
        if base.exists():
            found.extend(sorted(base.glob(pattern)))
    return found


def save_family(family: str, oof: np.ndarray, test_pred: np.ndarray,
                zips: pd.Series, test_zips: pd.Series, fingerprint: str,
                meta: dict = None, schema: str = None) -> Path:

    path = store_dir() / f"{family}.json"
    payload = {
        "family": family,
        "fingerprint": fingerprint,
        "schema": schema,
        "zips": list(map(str, zips)),
        "oof": [float(v) for v in oof],
        "test_zips": list(map(str, test_zips)),
        "test": [float(v) for v in test_pred],
        "meta": meta or {},
    }
    path.write_text(json.dumps(payload))
    print(f"  stored {family}: {len(oof)} oof, {len(test_pred)} test -> {path}")
    return path


def load_families(fingerprint: str, verbose: bool = True, schema_fn=None,
                  allow_legacy: bool = False) -> dict:

    out = {}
    for path in _find_stored():
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  !! unreadable {path}: {exc}")
            continue
        fam = payload.get("family", path.stem)
        if fam in out:
            continue
        if payload.get("fingerprint") != fingerprint:
            print(f"  !! SKIPPING {fam} ({path}): fingerprint {payload.get('fingerprint')} "
                  f"!= current {fingerprint} -- built on a different fold split")
            continue
        if schema_fn is not None:
            have = payload.get("schema")
            if have is None and not allow_legacy:
                print(f"  !! SKIPPING {fam} ({path}): no schema -- a pre-V9 file, built with "
                      f"different features or parameters")
                continue
            if have is not None and have != schema_fn(fam):
                print(f"  !! SKIPPING {fam} ({path}): schema {have} != current {schema_fn(fam)} "
                      f"-- built with different features or parameters")
                continue
        out[fam] = payload
        if verbose:
            print(f"  loaded {fam} from {path}")
    return out


def align(payload: dict, zips: pd.Series, key: str = "oof") -> np.ndarray:

    zcol = "zips" if key == "oof" else "test_zips"
    lookup = dict(zip(payload[zcol], payload[key]))
    return np.array([lookup.get(str(z), np.nan) for z in zips], dtype=float)

SAVE_MODELS = True

SKIP_MODEL_FAMILIES = {"forest", "knn", "tabpfn", "tabicl", "tabdpt", "realmlp", "tabm"}


def bundle_zip(family: str, pred_path: Path, models: list = None) -> Path:

    zip_path = get_output_dir() / f"{family}_store.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pred_path, arcname=f"{STORE_DIRNAME}/{pred_path.name}")
        if models and SAVE_MODELS and family not in SKIP_MODEL_FAMILIES:
            for tag, model in models:
                try:
                    zf.writestr(f"models/{family}/{tag}.pkl", pickle.dumps(model))
                except (pickle.PicklingError, TypeError, RecursionError) as exc:

                    print(f"  !! could not pickle {tag}: {exc}")
    mb = zip_path.stat().st_size / 1e6
    print(f"  bundled {zip_path.name} ({mb:.1f} MB)")
    return zip_path


def build_family_oof(family: str, train: pd.DataFrame, test: pd.DataFrame,
                     fit_predict, clear_cache=None,
                     collect=None) -> dict:

    folds = oof_folds(train)
    jun = train[train[MONTH] == TRAIN_MONTHS[1]].reset_index(drop=True)
    jun_fold = oof_folds(jun)

    oof = np.full(len(jun), np.nan)
    models = []
    if collect is not None:
        collect()
    for f in sorted(folds.unique()):
        fit = train[folds != f].reset_index(drop=True)
        held = jun[jun_fold == f].reset_index(drop=True)
        if not len(held):
            continue
        if clear_cache is not None:
            clear_cache()

        pred, = fit_predict(fit, [held], seed=f, family=family)
        oof[(jun_fold == f).to_numpy()] = pred
        if collect is not None:
            models.extend((f"fold{f}_{tag}", m) for tag, m in collect())
        print(f"    fold {f}: fit {len(fit)} rows, predicted {len(held)} zips")

    if clear_cache is not None:
        clear_cache()
    test_pred, = fit_predict(train, [test], seed=0, family=family)
    if collect is not None:
        models.extend((f"full_{tag}", m) for tag, m in collect())

    err = jun[TARGET].to_numpy() - oof
    rmse = float(np.sqrt(np.nanmean(err * err)))
    print(f"  {family}: OOF RMSE {rmse:.4f} over {int(np.isfinite(oof).sum())} zips")

    print(f"  {family}: {len(models)} estimators fitted")
    return {
        "oof": oof, "zips": jun[ID], "test": test_pred, "test_zips": test[ID],
        "fingerprint": oof_fingerprint(train), "rmse": rmse, "models": models,
    }

BLEND_RIDGE = 0.05


def optimise_weights(preds: dict, y: np.ndarray, families: list[str] = None,
                     ridge: float = BLEND_RIDGE) -> dict:

    fams = list(families or preds.keys())
    X = np.column_stack([preds[f] for f in fams])
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, yy = X[ok], y[ok]
    n = len(fams)

    prior = np.full(n, 1.0 / n)
    w = prior.copy()
    lr = 1.0 / (np.linalg.norm(X, 2) ** 2 / len(X) + ridge)
    for _ in range(2000):
        grad = X.T @ (X @ w - yy) / len(X) + ridge * (w - prior)
        w = _project_simplex(w - lr * grad)
    return dict(zip(fams, (float(v) for v in w)))


def _project_simplex(v: np.ndarray) -> np.ndarray:

    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > (css - 1))[0][-1]
    theta = (css[rho] - 1) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def blend_rmse(preds: dict, y: np.ndarray, weights: dict) -> float:
    acc = np.zeros(len(y))
    for fam, w in weights.items():
        acc += w * np.nan_to_num(preds[fam], nan=float(np.nanmean(y)))
    err = y - np.clip(acc, 0.0, 100.0)
    return float(np.sqrt(np.nanmean(err * err)))


def meta_matrix(preds: dict, families: list[str], extra: bool = True) -> np.ndarray:

    X = np.column_stack([preds[f] for f in families])
    if not extra:
        return X
    return np.column_stack([
        X,
        X.std(axis=1),
        X.max(axis=1) - X.min(axis=1),
        np.median(X, axis=1),
    ])


def level2_oof(preds: dict, y: np.ndarray, folds: np.ndarray, families: list[str],
               make_model, extra: bool = True) -> np.ndarray:

    X = meta_matrix(preds, families, extra)
    out = np.full(len(y), np.nan)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    for f in np.unique(folds):
        tr = ok & (folds != f)
        te = ok & (folds == f)
        if not tr.any() or not te.any():
            continue
        model = make_model()
        model.fit(X[tr], y[tr])
        out[te] = np.asarray(model.predict(X[te])).ravel()
    return out


def level2_fit_predict(preds: dict, y: np.ndarray, test_preds: dict,
                       families: list[str], make_model, extra: bool = True) -> np.ndarray:

    X = meta_matrix(preds, families, extra)
    Xt = meta_matrix(test_preds, families, extra)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    model = make_model()
    model.fit(X[ok], y[ok])
    col_med = np.nanmedian(X[ok], axis=0)
    Xt = np.where(np.isfinite(Xt), Xt, col_med)
    return np.asarray(model.predict(Xt)).ravel()
