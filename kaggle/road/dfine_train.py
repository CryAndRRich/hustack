import json
import sys
import time
from argparse import Namespace


def main(args):
    sys.path.insert(0, args["repo"])
    import train
    from src.solver import det_solver
    inner = det_solver.train_one_epoch
    start = {}

    def budgeted(*a, **kw):
        loader, epoch = a[2], a[5]
        now = time.time()
        start.setdefault("t", now)
        start.setdefault("epoch", epoch)
        if epoch > start["epoch"]:
            per = (now - start["t"]) / (epoch - start["epoch"])
            end = epoch + int((args["deadline"] - now) / per)
            collate = loader.collate_fn
            if epoch < collate.stop_epoch:
                collate.stop_epoch = max(epoch + 1, end - args["plain_epochs"])
                loader.dataset._transforms.policy["epoch"] = collate.stop_epoch
            print(f"budget: {per:.0f} s per epoch, last epoch {end - 1}, plain from {collate.stop_epoch}", flush=True)
            if epoch >= end:
                print("budget: deadline reached, training stops", flush=True)
                raise SystemExit(0)
        return inner(*a, **kw)

    det_solver.train_one_epoch = budgeted
    train.main(Namespace(config=args["config"], resume=None, tuning=args["tuning"], device=None, seed=args["seed"],
                         use_amp=True, output_dir=args["output_dir"], summary_dir=None, test_only=False, update=None,
                         print_method="builtin", print_rank=0, local_rank=None))


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        main(json.load(f))
