import numpy as np
import torch
import torch.nn.functional as F

IOU_SIZE = 256


def view(x, v):
    return [x, x.flip(-1), x.flip(-2), torch.rot90(x, 1, (-2, -1))][v]


def unview(x, v):
    return [x, x.flip(-1), x.flip(-2), torch.rot90(x, -1, (-2, -1))][v]


def fuse(views, n_views, iou_thr):
    probs = [p for p, _ in views]
    scores = [s for _, s in views]
    dev = probs[0].device
    if n_views == 1:
        return probs[0], scores[0]
    allp, alls = torch.cat(probs), torch.cat(scores)
    if allp.shape[0] == 0:
        return allp, alls
    vid = np.concatenate([np.full(len(p), v) for v, p in enumerate(probs)])
    small = F.interpolate(allp[:, None].float(), size=(IOU_SIZE, IOU_SIZE), mode="bilinear", align_corners=False)[:, 0] > 0.5
    small = small.flatten(1).float()
    inter = small @ small.T
    ar = small.sum(1)
    iou = (inter / (ar[:, None] + ar[None, :] - inter).clamp(min=1)).cpu().numpy()
    sc = alls.cpu().numpy()
    assigned = np.zeros(len(sc), bool)
    out_p, out_s = [], []
    for i in np.argsort(-sc, kind="stable"):
        if assigned[i]:
            continue
        members = [i]
        assigned[i] = True
        for v in range(n_views):
            if v == vid[i]:
                continue
            cand = np.where((vid == v) & ~assigned & (iou[i] > iou_thr))[0]
            if len(cand):
                j = cand[np.argmax(iou[i, cand])]
                members.append(j)
                assigned[j] = True
        out_p.append(allp[members].float().mean(0).half())
        out_s.append(float(sc[members].mean()) * len(members) / n_views)
    order = np.argsort(-np.array(out_s), kind="stable")
    return torch.stack([out_p[k] for k in order]), torch.tensor([out_s[k] for k in order], device=dev)
