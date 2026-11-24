import json
import sys


def degradations():
    import albumentations as A
    return [
        A.OneOf([A.GaussianBlur(blur_limit=(3, 9)), A.MotionBlur(blur_limit=(3, 11))], p=0.12),
        A.GaussNoise(std_range=(0.03, 0.12), p=0.10),
        A.ImageCompression(quality_range=(10, 60), p=0.15),
        A.RandomBrightnessContrast(brightness_limit=(-0.45, 0.10), contrast_limit=(-0.30, 0.10), p=0.15),
        A.RandomGamma(gamma_limit=(100, 250), p=0.10),
        A.ToGray(p=0.01),
    ]


def fit(args):
    from ultralytics import YOLO
    model = YOLO(args["weights"])
    model.train(data=args["data"], imgsz=args["imgsz"], epochs=args["epochs"], time=args["hours"], batch=args["batch"],
                device=0, workers=args["workers"], cache="disk", fraction=args["fraction"], optimizer="SGD",
                lr0=args["lr0"], momentum=0.937, weight_decay=5e-4, close_mosaic=min(10, args["epochs"] // 3),
                mosaic=args["mosaic"], mixup=0.0, fliplr=0.5, flipud=0.0, degrees=0.0, augmentations=degradations(),
                val=True, plots=False, patience=1000, deterministic=False, seed=args["seed"],
                project=args["project"], name=args["name"], exist_ok=True, amp=True)


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        fit(json.load(f))
