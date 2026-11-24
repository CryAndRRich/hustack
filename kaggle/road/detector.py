import cv2
import numpy as np
import torch
import torchvision
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils.nms import non_max_suppression


class Yolo:
    def __init__(self, weights, imgsz, cfg):
        model, _ = load_checkpoint(weights, device="cuda", fuse=True)
        self.model = model.half().eval()
        self.size = imgsz
        self.post = {"conf_thres": cfg.conf, "iou_thres": cfg.nms_iou, "multi_label": cfg.multi_label,
                     "max_det": cfg.max_det, "max_time_img": 2.0}
        torch.backends.cudnn.benchmark = True
        print(f"yolo {weights} at {imgsz}: torchvision {torchvision.__version__} NMS")

    @torch.inference_mode()
    def __call__(self, image, flip):
        h, w = image.shape[:2]
        s = self.size
        x = cv2.resize(image, (s, s), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(x).cuda().permute(2, 0, 1)[None].half().div_(255)
        if flip:
            t = torch.cat([t, t.flip(3)])
        dets = non_max_suppression(self.model(t)[0].float(), **self.post)
        res = []
        for k, d in enumerate(dets):
            d = d.cpu().numpy().astype(np.float64)
            if k == 1:
                d[:, [0, 2]] = s - d[:, [2, 0]]
            d[:, [0, 2]] = np.clip(d[:, [0, 2]] * (w / s), 0, w)
            d[:, [1, 3]] = np.clip(d[:, [1, 3]] * (h / s), 0, h)
            d = d[(d[:, 2] - d[:, 0] >= 0.05) & (d[:, 3] - d[:, 1] >= 0.05)]
            res.append(np.column_stack([d[:, 5], d[:, 4], d[:, :4]]))
        return res


def degrade(image, i):
    kind = i % 4
    if kind == 0:
        return cv2.GaussianBlur(image, (9, 9), 3)
    if kind == 1:
        return np.clip(((image / 255.0) ** 1.8) * 0.5 * 255, 0, 255).astype(np.uint8)
    if kind == 2:
        noise = np.random.default_rng(i).normal(0, 20, image.shape)
        return np.clip(image + noise, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 10])
    assert ok
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
