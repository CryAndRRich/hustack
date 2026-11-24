import sys

import torch

from common import cropcls
from common.runtime import Worker
from common.stages import crop_leaf_stage, crop_stage, load_crop_cls, plant_stage
from leaf_instance.config import Config

DIRS = ("plant_val", "plant_test", "maps_val", "maps_test", "crop_val", "crop_test", "leaf_val", "leaf_test",
        "pred/leaf_instances", "val_pred/leaf_instances", "models/crop_cls")


def main(rank, mode, cfg_path):
    w = Worker(Config, rank, mode, cfg_path, DIRS)
    rank, cfg, smoke, log, work = w.rank, w.cfg, w.smoke, w.log, w.work
    cache, dev, sync, dirs, res, tm = w.cache, w.dev, w.sync, w.dirs, w.res, w.tm
    pos, grp = w.groups()
    mdir, qidx = dirs["models/crop_cls"], w.quick_idx()
    gates = sorted(set(cfg.leaf_gates))

    model = cropcls.build(cfg).to(dev)
    stats = cropcls.train_cls(cfg, model, cropcls.PlantSet(cfg, work / "cache"), dev, smoke, log)
    stats["val"] = cropcls.val_check(cfg, [model], cache, qidx, dev)
    log(f"crop classifier {rank}: {stats}")
    res["train_cls"] = stats
    torch.save(model.state_dict(), mdir / f"rank{rank}.pt")
    del model
    torch.cuda.empty_cache()
    sync.barrier("trained")

    plant_stage(w, gates[0])

    sync.barrier("cls_saved")
    cls = load_crop_cls(cfg, mdir, dev)
    res["cls_val_ensemble"] = cropcls.val_check(cfg, cls, cache, qidx, dev)
    log(f"crop classifier ensemble on val GT plants: {res['cls_val_ensemble']}")
    crop_stage(w, cls)
    del cls
    torch.cuda.empty_cache()

    dec, tables, counts = crop_leaf_stage(w, "maps", gates, pos, grp)
    res.update({"leaf_test_instances": counts, "decision": dec, "tables": tables, "timing": tm})
    w.finish(["decision", "timing"])


if __name__ == "__main__":
    main(*sys.argv[1:4])
