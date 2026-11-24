import torch
import torch.nn.functional as F

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255
M2F_BACKBONE = "pixel_level_module.encoder."
HEAD_KEYS = ("class_predictor", "mask_head", "upscale_block", "query")
EMBED_KEYS = ("embeddings", "register", "cls_token", "mask_token")


def to_f32(x):
    if torch.is_tensor(x):
        return x.float() if x.is_floating_point() else x
    if isinstance(x, (list, tuple)):
        return type(x)(to_f32(v) for v in x)
    if isinstance(x, dict):
        return {k: to_f32(v) for k, v in x.items()}
    return x


def loss_in_fp32(model):
    if not hasattr(model, "criterion"):
        return False
    crit, forward = model.criterion, model.criterion.forward

    def fp32_forward(*args, **kwargs):
        with torch.autocast("cuda", enabled=False):
            return forward(*to_f32(args), **to_f32(kwargs))

    crit.forward = fp32_forward
    return True


def load_m2f(path):
    from transformers import Mask2FormerForUniversalSegmentation
    model = Mask2FormerForUniversalSegmentation.from_pretrained(path, dtype=torch.float32)
    model.loss_fp32 = loss_in_fp32(model)
    return model


def load_eomt(path, size):
    from transformers import EomtDinov3ForUniversalSegmentation
    model = EomtDinov3ForUniversalSegmentation.from_pretrained(path, image_size=size, dtype=torch.float32)
    bufs = [b for n, b in model.named_buffers() if n.endswith("attn_mask_probs")]
    assert len(bufs) == 1, f"expected one attn_mask_probs buffer, found {len(bufs)}"
    bufs[0].zero_()
    model.loss_fp32 = loss_in_fp32(model)
    return model


def block_depth(name):
    comps = name.split(".")
    if any(c in HEAD_KEYS for c in comps[:2]):
        return None
    for a, b in zip(comps, comps[1:]):
        if a in ("layer", "layers") and b.isdigit():
            return int(b)
    return -1 if any(k in name for k in EMBED_KEYS) else None


def param_groups(model, kind, lr, cfg):
    named = list(model.named_parameters())
    groups = {}
    if kind == "m2f":
        for n, p in named:
            groups.setdefault(cfg.backbone_mult_plant if M2F_BACKBONE in n else 1.0, []).append(p)
        summary = {f"x{m}": round(sum(p.numel() for p in ps) / 1e6, 1) for m, ps in groups.items()}
        assert len(groups) == 2, f"Mask2Former backbone parameters not found ({M2F_BACKBONE}): {summary}"
    else:
        depth = {n: block_depth(n) for n, _ in named}
        n_blocks = max([d for d in depth.values() if d is not None] + [0]) + 1
        assert n_blocks >= 12, f"found only {n_blocks} transformer blocks"
        for n, p in named:
            d = depth[n]
            groups.setdefault(1.0 if d is None else cfg.llrd ** (n_blocks - 1 - d), []).append(p)
        summary = {"blocks": n_blocks, "groups": len(groups)}
    return [{"params": v, "lr": lr * m, "base_lr": lr * m} for m, v in sorted(groups.items())], summary


@torch.no_grad()
def predict_batch(model, x, cfg, out_size):
    dev = x.device
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(pixel_values=(x - MEAN.to(dev)) / STD.to(dev))
    cls = out.class_queries_logits.float().softmax(-1)[..., 0]
    res = []
    for b in range(x.shape[0]):
        idx = (cls[b] > cfg.min_score).nonzero()[:, 0]
        if idx.numel() == 0:
            res.append((torch.zeros((0, out_size, out_size), dtype=torch.float16, device=dev), torch.zeros(0, device=dev)))
            continue
        prob = F.interpolate(out.masks_queries_logits[b, idx].float()[None], size=(out_size, out_size),
                             mode="bilinear", align_corners=False)[0].sigmoid()
        binm = prob > 0.5
        area = binm.flatten(1).sum(1)
        score = cls[b, idx] * (prob * binm).flatten(1).sum(1) / area.clamp(min=1)
        score = torch.where(area > 0, score, torch.zeros_like(score))
        top = torch.argsort(score, descending=True)[:cfg.max_cands]
        top = top[score[top] >= cfg.min_score]
        res.append((prob[top].half(), score[top]))
    return res
