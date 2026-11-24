import hashlib
from collections import deque

import numpy as np

from common.items import box, boxes_array, iou, overlapping, place

SIZE = 1024
HALF = SIZE // 2


def _digest(a):
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


def half_digests(a):
    return _digest(a[:, :HALF]), _digest(a[:HALF]), _digest(a[:, HALF:]), _digest(a[HALF:])


def link(digests):
    left, top = {}, {}
    for i, (lft, tp, _, _) in digests.items():
        left.setdefault(lft, []).append(i)
        top.setdefault(tp, []).append(i)
    edges = {i: [] for i in digests}
    for i, (_, _, rgt, bot) in digests.items():
        for table, key, off in ((left, rgt, (0, HALF)), (top, bot, (HALF, 0))):
            hits = [j for j in table.get(key, []) if j != i]
            if len(hits) == 1:
                j = hits[0]
                edges[i].append((j, off))
                edges[j].append((i, (-off[0], -off[1])))
    return edges


def find_groups(imgs, idx, log, tag):
    edges = link({i: half_digests(np.asarray(imgs[i])) for i in idx})
    pos, groups, bad = positions(edges, idx, lambda members, p: _consistent(imgs, members, p))
    sizes = sorted((len(m) for m in groups), reverse=True)
    log(f"{tag}: {len(pos)} of {len(idx)} tiles in {len(groups)} overlap groups (largest {sizes[:5]}), "
        f"{bad} inconsistent groups dropped")
    return {int(i): [int(v) for v in p] for i, p in pos.items()}


def positions(edges, idx, check=None):
    pos, groups, bad = {}, [], 0
    for root in idx:
        if root in pos or not edges[root]:
            continue
        g, members, ok = len(groups), [root], True
        pos[root] = (g, 0, 0)
        queue = deque([root])
        while queue:
            i = queue.popleft()
            for j, (dy, dx) in edges[i]:
                p = (g, pos[i][1] + dy, pos[i][2] + dx)
                if j not in pos:
                    pos[j] = p
                    members.append(j)
                    queue.append(j)
                elif pos[j] != p:
                    ok = False
        ok = ok and (check is None or check(members, pos))
        if not ok:
            bad += 1
            for j in members:
                del pos[j]
            continue
        groups.append(members)
    return pos, groups, bad


def _consistent(imgs, members, pos):
    for a in members:
        for b in members:
            dy, dx = pos[b][1] - pos[a][1], pos[b][2] - pos[a][2]
            if b <= a or abs(dy) >= SIZE or abs(dx) >= SIZE:
                continue
            ya, xa = max(dy, 0), max(dx, 0)
            h, w = SIZE - abs(dy), SIZE - abs(dx)
            if not np.array_equal(np.asarray(imgs[a][ya:ya + h, xa:xa + w]), np.asarray(imgs[b][ya - dy:ya - dy + h, xa - dx:xa - dx + w])):
                return False
    return True


def members_of(pos):
    groups = {}
    for i, (g, _, _) in pos.items():
        groups.setdefault(g, []).append(i)
    return groups


def stitch(i, load, pos, groups, thr):
    if i not in pos:
        return load(i)
    g, gy, gx = pos[i]
    cands = []
    for j in groups[g]:
        _, jy, jx = pos[j]
        dy, dx = jy - gy, jx - gx
        if abs(dy) >= SIZE or abs(dx) >= SIZE:
            continue
        for y0, x0, m, s, tag in load(j):
            depth = min(y0, x0, SIZE - y0 - m.shape[0], SIZE - x0 - m.shape[1])
            if j != i and depth == 0:
                continue
            p = place(m, y0 + dy, x0 + dx)
            if p is not None:
                cands.append(((p[0], p[1], p[2], s, tag), j, depth))
    cands.sort(key=lambda c: -c[0][3])
    items = [c[0] for c in cands]
    bxs = boxes_array(items)
    taken = np.zeros(len(items), bool)
    out = []
    for k in range(len(items)):
        if taken[k]:
            continue
        taken[k] = True
        best = {}
        for o in overlapping(bxs, box(items[k])):
            if taken[o] or cands[o][1] == cands[k][1]:
                continue
            v = iou(items[k], items[o])
            if v > thr and v > best.get(cands[o][1], (0.0, -1))[0]:
                best[cands[o][1]] = (v, o)
        members = [k] + [o for _, o in best.values()]
        taken[members] = True
        deep = max(members, key=lambda o: cands[o][2])
        y0, x0, m = items[deep][:3]
        out.append((y0, x0, m, items[k][3], items[k][4]))
    return out
