import json
import os
import subprocess
import sys
import time
import traceback

import numpy as np

cfg = json.load(open(sys.argv[1]))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "olmoearth-pretrain==0.1.2", "einops",
                "universal-pathlib", "open_clip_torch==3.3.0", "ftfy"], check=False)
import torch

torch.set_num_threads(1)
if not torch.cuda.is_available():
    sys.exit("embed worker: no GPU")
dev = torch.device("cuda")
OURS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
CLASSES = ["an oil refinery", "a gas flare at an oil field", "oil well pads in a desert", "a coal power plant",
           "a power plant with cooling towers", "a gas power plant", "an open-pit mine", "a steel mill",
           "a cement plant", "storage tanks", "a chemical plant", "a port", "a city", "farmland", "bare desert",
           "mountains", "a forest"]
BS = 32


def olmoearth(size, patch):
    from olmoearth_pretrain.data.constants import Modality
    from olmoearth_pretrain.data.normalize import Normalizer, Strategy
    from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
    from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
    spec = Modality.SENTINEL2_L2A
    names = spec.band_order
    order = [OURS.index(b if b == "B8A" else f"B{int(b[1:])}") for b in names]
    norm = Normalizer(Strategy.COMPUTED).norm_config[spec.name]
    mean = torch.tensor([norm[b]["mean"] for b in names], device=dev)
    std = torch.tensor([norm[b]["std"] for b in names], device=dev)
    lo, span = mean - 2 * std, 4 * std
    model = load_model_from_id(ModelID[f"OLMOEARTH_V1_{size}"]).to(dev).eval()

    def run(a, months):
        x = torch.from_numpy(a[:, order].astype(np.float32)).to(dev).permute(0, 2, 3, 1)
        x = ((x - lo) / span)[:, :, :, None, :]
        b, h, w = x.shape[:3]
        mask = torch.full((b, h, w, 1, spec.num_band_sets), float(MaskValue.ONLINE_ENCODER.value), device=dev)
        ts = torch.tensor([[[15, int(m), 2023]] for m in months], device=dev)
        sample = MaskedOlmoEarthSample(sentinel2_l2a=x, sentinel2_l2a_mask=mask, timestamps=ts)
        with torch.no_grad():
            tok = model.encoder(sample, fast_pass=True, patch_size=patch)["tokens_and_masks"].sentinel2_l2a
        tok = tok.float().mean(dim=[3, 4])
        hh, ww = tok.shape[1], tok.shape[2]
        ctr = tok[:, hh // 4:hh - hh // 4, ww // 4:ww - ww // 4]
        return torch.cat([tok.mean((1, 2)), tok.amax((1, 2)), ctr.mean((1, 2))], 1).cpu().numpy()
    return run


def remoteclip():
    import open_clip
    from huggingface_hub import hf_hub_download
    model = open_clip.create_model_and_transforms("ViT-B-32")[0]
    model.load_state_dict(torch.load(hf_hub_download("chendelong/RemoteCLIP", "RemoteCLIP-ViT-B-32.pt"), map_location="cpu"))
    model = model.to(dev).eval()
    tok = open_clip.get_tokenizer("ViT-B-32")
    cm = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev)[None, :, None, None]
    cs = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev)[None, :, None, None]
    with torch.no_grad():
        txt = model.encode_text(tok([f"a satellite image of {c}." for c in CLASSES]).to(dev)).float()
        txt = txt / txt.norm(dim=-1, keepdim=True)

    def run(a, months):
        with torch.no_grad():
            rgb = torch.from_numpy(a[:, [3, 2, 1]].astype(np.float32)).to(dev)
            rgb = (rgb / 10000.0 * 3.5).clamp(0, 1)
            rgb = torch.nn.functional.interpolate(rgb, size=224, mode="bilinear", align_corners=False)
            img = model.encode_image((rgb - cm) / cs).float()
            img = img / img.norm(dim=-1, keepdim=True)
            return torch.log_softmax(100.0 * img @ txt.T, dim=1).cpu().numpy()
    return run


def embed(fn, arr, months):
    return np.concatenate([fn(arr[i:i + BS], months[i:i + BS]) for i in range(0, len(arr), BS)])


def ext_batches():
    for p in cfg["ext_chips"]:
        z = np.load(p)
        yield z["x"][:, :12], (np.array([int(d[5:7]) for d in z["date"]]) - 1), z["fac_id"], z["date"]


models = {}
for name, make in (("olmo", lambda: olmoearth("NANO", 4)), ("olmo_base", lambda: olmoearth("BASE", 8)),
                   ("clip", remoteclip)):
    try:
        models[name] = make()
    except Exception:
        print(f"embed {name} load failed:\n{traceback.format_exc()[-1500:]}", flush=True)

comp = {}
fm, months = np.load(cfg["fm"], mmap_mode="r"), np.load(cfg["months"])
for name, fn in models.items():
    t0 = time.time()
    try:
        comp[name] = embed(fn, fm, months)
        print(f"embed {name}: {comp[name].shape} in {time.time() - t0:.0f}s, finite {np.isfinite(comp[name]).all()}", flush=True)
    except Exception:
        print(f"embed {name} failed:\n{traceback.format_exc()[-1500:]}", flush=True)
path = os.path.join(cfg["out"], "emb.npz")
np.savez(path + ".tmp.npz", **comp)
os.replace(path + ".tmp.npz", path)

if cfg["ext_chips"] and comp:
    t0 = time.time()
    parts = {k: [] for k in comp}
    keys = {"fac_id": [], "date": []}
    for x, m, fac, date in ext_batches():
        if time.time() > cfg["deadline"]:
            print("embed ext: deadline, skipped", flush=True)
            break
        for k in comp:
            parts[k].append(embed(models[k], x, m))
        keys["fac_id"].append(fac)
        keys["date"].append(date)
    else:
        out = {k: np.concatenate(v) for k, v in parts.items()}
        out.update({k: np.concatenate(v) for k, v in keys.items()})
        path = os.path.join(cfg["out"], "emb_ext.npz")
        np.savez(path + ".tmp.npz", **out)
        os.replace(path + ".tmp.npz", path)
        print(f"embed ext: {len(out['fac_id'])} chips, {list(comp)} in {time.time() - t0:.0f}s", flush=True)
print("embed worker finished", flush=True)
