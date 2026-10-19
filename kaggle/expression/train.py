import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

import fer
from fer import CLASSES, get_data_dir, load_images, near_pairs, union_find_groups

VERSION = "v3"
AUTO_SMOKE = True
SEED = 42
N_SPLITS, VAL_FOLD = 5, 0
CFG = {
    "seed": SEED, "arch": "vit_base_patch16_224", "img_size": 224, "epochs": 30, "warmup_epochs": 3, "bs": 64,
    "lr": 1e-3 * 64 / 256, "layer_decay": 0.65, "wd": 0.05, "drop_path": 0.1, "label_smoothing": 0.15,
    "crop": 56, "p_aug": 0.5, "scale": 0.2, "shift": 0.2, "rot": 10, "erase_p": 0.25, "eval_every": 5, "tta": True,
    "max_train": None, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
}
SMOKE = {"epochs": 1, "eval_every": 0, "max_train": 1500, "tta": False}


def get_out_dir():
    return Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")


def prepare():
    t0 = time.time()
    data_dir = get_data_dir()
    X, y, ids, X_te, te_ids, md5_tr, md5_te = load_images(data_dir)
    print(f"data {data_dir}: train {X.shape}, test {X_te.shape}, loaded in {time.time() - t0:.0f}s")
    print("class counts:", dict(zip(CLASSES, np.bincount(y, minlength=7).tolist())))

    blank = X.reshape(len(X), -1).std(1) < 3
    blank_te = X_te.reshape(len(X_te), -1).std(1) < 3
    print(f"near-uniform frames: train {blank.sum()} (dropped), test {blank_te.sum()}")

    t1 = time.time()
    n = len(X)
    pairs = near_pairs(np.concatenate([X, X_te]), np.concatenate([blank, blank_te]), torch.device("cuda"))
    a, b, corr = pairs[:, 0].astype(int), pairs[:, 1].astype(int), pairs[:, 2]
    tt = (a < n) & (b < n)
    cross = (a < n) != (b < n)
    exact_tr = Counter(md5_tr)
    print(f"near-duplicate pairs (thumbnail corr > 0.98): train-train {tt.sum()}, train-test {cross.sum()}, "
          f"test-test {((a >= n) & (b >= n)).sum()}  ({time.time() - t1:.0f}s)")
    print(f"exact-copy groups within train (md5): {sum(v > 1 for v in exact_tr.values())}")
    diff = y[a[tt]] != y[b[tt]]
    ex = np.array([md5_tr[i] == md5_tr[j] for i, j in zip(a[tt], b[tt])], dtype=bool)
    print(f"label disagreement within train duplicate pairs: {diff.mean():.3f} ({diff.sum()}/{tt.sum()})")
    print(f"label agreement of train copy pairs: exact {1 - diff[ex].mean():.3f} (n={ex.sum()}), "
          f"near {1 - diff[~ex].mean():.3f} (n={(~ex).sum()})")
    group = union_find_groups(n, np.stack([a[tt], b[tt]], 1))
    sizes = Counter(group)
    print(f"train groups: {len(sizes)} ({sum(s > 1 for s in sizes.values())} with >1 image, largest {max(sizes.values())})")

    tr_i = np.minimum(a, b)[cross]
    te_i = np.maximum(a, b)[cross] - n
    links = pd.DataFrame({"test_id": te_ids[te_i], "train_id": ids[tr_i], "train_label": y[tr_i],
                          "corr": corr[cross], "exact": [md5_te[j] == md5_tr[i] for i, j in zip(tr_i, te_i)]})
    print(f"test images with a train copy: {links['test_id'].nunique()} "
          f"(exact {links.loc[links['exact'], 'test_id'].nunique()})")

    split = np.full(n, -1)
    keep = np.where(~blank)[0]
    sgkf = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=SEED)
    for k, (_, va) in enumerate(sgkf.split(keep, y[keep], group[keep])):
        split[keep[va]] = 0 if k == VAL_FOLD else 1
    print(f"split: train {(split == 1).sum()}, val {(split == 0).sum()} (fold {VAL_FOLD} of {N_SPLITS}, group-aware)")

    return {"cfg": dict(CFG), "X": X, "y": y, "ids": ids, "X_te": X_te, "te_ids": te_ids, "split": split,
            "group": group, "links": links}


def fetch_checkpoint(cfg):
    name = fer.FMAE_FILES[cfg["arch"]]
    path = fer.find_input(name) if Path("/kaggle/input").exists() else None
    if path is not None:
        return path
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    try:
        return Path(hf_hub_download(fer.FMAE_REPO, name))
    except Exception as e:
        raise RuntimeError(f"cannot get {name} from {fer.FMAE_REPO}: turn Internet ON or attach the file ({e})") from e


def prepare_init(out, cfg, path):
    t0 = time.time()
    try:
        torch.serialization.add_safe_globals([argparse.Namespace, (pathlib.PosixPath, "pathlib.PosixPath")])
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        print(f"  weights-only load refused ({type(e).__name__}: {str(e)[:120]}); loading the full pickle")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    enc = {k: v for k, v in state.items() if not k.startswith(("decoder", "mask_token", "norm."))}
    model = fer.build_model(cfg)
    msg = fer.load_encoder(model, enc)
    assert not msg.unexpected_keys, f"checkpoint keys the model does not have: {msg.unexpected_keys[:10]}"
    n_par = sum(p.numel() for p in model.parameters())
    print(f"FMAE {cfg['arch']} <- {path} ({path.stat().st_size / 1e9:.2f} GB, pretraining epoch {ckpt.get('epoch')}): "
          f"{len(enc)} encoder tensors loaded, {len(state) - len(enc)} dropped (decoder / mask_token / final norm), "
          f"new {sorted(msg.missing_keys)}; {n_par / 1e6:.1f} M params; {time.time() - t0:.0f}s")
    init = out / "fmae_init.pt"
    torch.save(enc, init)
    del ckpt, state, model
    return init


def save_data(out, version, X, y, split, X_te):
    path = out / f"{version}_data.npz"
    np.savez(path, X_tr=X, y=y, split=split, X_te=X_te)
    return path


def launch(cfg, data_path, run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    n_gpu = torch.cuda.device_count()
    assert n_gpu >= 1, "needs Accelerator = GPU"
    worker = Path(fer.__file__).resolve().parent / "worker.py"
    log_path = run_dir / "run.log"
    log = open(log_path, "w")
    p = subprocess.Popen([sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={n_gpu}",
                          str(worker), "--data", str(data_path), "--out", str(run_dir), "--cfg", json.dumps(cfg)],
                         stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2"))
    print(f"launched torchrun: 1 model on {n_gpu} GPU(s) (DDP)")
    seen = 0
    while True:
        time.sleep(2)
        lines = log_path.read_text().splitlines()
        for line in lines[seen:]:
            print(line)
        seen = len(lines)
        if p.poll() is not None:
            break
    log.close()
    for line in log_path.read_text().splitlines()[seen:]:
        print(line)
    if p.returncode != 0:
        tail = "\n".join(log_path.read_text().splitlines()[-30:])
        raise RuntimeError(f"torchrun failed (exit {p.returncode}):\n{tail}")


def evaluate(run_dir, y, split):
    d = np.load(run_dir / "run.npz")
    va, va_probs = d["va_idx"], d["va_probs"]
    assert np.array_equal(va, np.where(split == 0)[0])
    pred, yv = va_probs.argmax(1), y[va]
    print("\n=== validation (group-aware holdout) ===")
    print(f"  val acc {np.mean(pred == yv):.4f}  (n={len(va)}; same images: V1 fold 0 0.6908, V1.1 0.6898, V2 0.6944)")
    cm = np.zeros((7, 7), dtype=int)
    for t, p in zip(yv, pred):
        cm[t, p] += 1
    print("\n  confusion matrix (rows = true, cols = predicted):")
    print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string())
    rec = cm.diagonal() / cm.sum(1)
    print("  per-class recall: " + "  ".join(f"{c} {r:.3f}" for c, r in zip(CLASSES, rec)))
    return va, va_probs, d["te_probs"], cm


def copy_label_lookup(te_probs, te_ids, links):
    model_pred = te_probs.argmax(1)
    final = model_pred.copy()
    pos = {t: i for i, t in enumerate(te_ids)}
    rows = []
    for tid, g in links.groupby("test_id"):
        src = g[g["exact"]] if g["exact"].any() else g
        votes = np.bincount(src["train_label"].to_numpy(), minlength=7)
        tied = np.flatnonzero(votes == votes.max())
        i = pos[tid]
        lab = int(tied[np.argmax(te_probs[i, tied])])
        rows.append((tid, bool(g["exact"].any()), len(src), bool(votes.max() < len(src)), int(model_pred[i]), lab))
        final[i] = lab
    info = pd.DataFrame(rows, columns=["test_id", "exact", "n_copies", "copies_disagree", "model_label", "lookup_label"])
    same = info["model_label"] == info["lookup_label"]
    print("\n=== copy-label lookup on test ===")
    print(f"  test images with a train copy: {len(info)} (exact {info['exact'].sum()}, near only {(~info['exact']).sum()}); "
          f"copies disagree among themselves on {info['copies_disagree'].sum()}")
    print(f"  model already gives the copy's label: {same.mean():.3f} "
          f"(exact {same[info['exact']].mean():.3f}, near {same[~info['exact']].mean():.3f})")
    print(f"  labels changed by the lookup: {(~same).sum()} of 7000 test images ({(~same).sum() / 70:.2f}% of test)")
    return final, model_pred, info


def write_outputs(out, va, va_probs, te_probs, final, model_pred, info, cm, d, version):
    y, ids, te_ids, group = d["y"], d["ids"], d["te_ids"], d["group"]
    extra = out / "extra"
    extra.mkdir(parents=True, exist_ok=True)
    for name, pred in (("submission.csv", final), ("submission_model_only.csv", model_pred)):
        sub = pd.DataFrame({"id": te_ids, "label": pred})
        assert len(sub) == 7000 and sub["id"].is_unique and sub["label"].between(0, 6).all()
        sub.to_csv(out / name if name == "submission.csv" else extra / name, index=False)
    pcols = [f"p{c}" for c in range(7)]
    pd.concat([pd.DataFrame({"id": ids[va], "label": y[va], "group": group[va]}),
               pd.DataFrame(va_probs, columns=pcols)], axis=1).to_csv(extra / f"val_probs_{version}.csv", index=False)
    pd.concat([pd.DataFrame({"id": te_ids}), pd.DataFrame(te_probs, columns=pcols)], axis=1).to_csv(
        extra / f"test_probs_{version}.csv", index=False)
    d["links"].to_csv(extra / "test_train_links.csv", index=False)
    info.to_csv(extra / f"copy_lookup_{version}.csv", index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        norm = cm / cm.sum(1, keepdims=True)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(7), CLASSES, rotation=45)
        ax.set_yticks(range(7), CLASSES)
        for i in range(7):
            for j in range(7):
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                        color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{version} validation confusion matrix (row-normalised)")
        fig.tight_layout()
        fig.savefig(extra / f"confusion_{version}.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"confusion plot skipped: {e}")
    print("\n  test prediction shares: " + "  ".join(f"{c} {s:.3f}" for c, s in zip(CLASSES, np.bincount(final, minlength=7) / len(final))))
    print("  train label shares:     " + "  ".join(f"{c} {s:.3f}" for c, s in zip(CLASSES, np.bincount(y, minlength=7) / len(y))))


def run(cfg, data_path, run_dir, out, d, version):
    launch(cfg, data_path, run_dir)
    va, va_probs, te_probs, cm = evaluate(run_dir, d["y"], d["split"])
    final, model_pred, info = copy_label_lookup(te_probs, d["te_ids"], d["links"])
    write_outputs(out, va, va_probs, te_probs, final, model_pred, info, cm, d, version)


def main():
    t_all = time.time()
    out = get_out_dir()
    print(f"GPUs: {torch.cuda.device_count()} — {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
    assert torch.cuda.is_available(), "needs Accelerator = GPU"
    with ThreadPoolExecutor(1) as pool:
        ckpt_job = pool.submit(fetch_checkpoint, CFG)
        d = prepare()
        ckpt_path = ckpt_job.result()
    cfg = d["cfg"]
    init_path = prepare_init(out, cfg, ckpt_path)
    cfg["init"] = str(init_path)
    data_path = save_data(out, VERSION, d["X"], d["y"], d["split"], d["X_te"])
    torch.cuda.empty_cache()

    if AUTO_SMOKE:
        t0 = time.time()
        print("\n=== smoke pass: 1 epoch, 1,500 rows, DDP on all GPUs (discarded) ===")
        smoke_dir = out / "smoke_discard"
        run({**cfg, **SMOKE}, data_path, smoke_dir / "run", smoke_dir, d, VERSION)
        sec = json.loads((smoke_dir / "run" / "hist.json").read_text())[0]["sec"]
        n_tr = int((d["split"] == 1).sum())
        print(f"projected training time: {sec / SMOKE['max_train'] * n_tr * cfg['epochs'] / 60:.0f} min "
              f"({sec:.0f}s for the 1,500-row smoke epoch, warm-up included: an upper bound)")
        shutil.rmtree(smoke_dir)
        print(f"smoke ok in {time.time() - t0:.0f}s")

    print(f"\n=== {VERSION}: FMAE {cfg['arch']} fine-tune, one group-aware train/val split, {cfg['epochs']} epochs, "
          f"copy-label lookup ===")
    print("cfg: " + json.dumps(cfg))
    run_dir = out / f"{VERSION}_run"
    run(cfg, data_path, run_dir, out, d, VERSION)
    for name in ("hist.json", "run.log", "model.pt"):
        shutil.copy(run_dir / name, out / "extra" / f"{VERSION}_{name}")
    data_path.unlink()
    init_path.unlink()
    shutil.rmtree(run_dir)
    print(f"total {time.time() - t_all:.0f}s")
    print("\n" + "=" * 70)
    print(f"SUBMIT: {out / 'submission.csv'}  ({VERSION}: one FMAE ViT-B + copy-label lookup; "
          f"model-only predictions and diagnostics in {out / 'extra'})")
    print("=" * 70)


if __name__ == "__main__":
    main()
