import math

import cv2
import numpy as np


def to_gray(img, cfg):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return cv2.GaussianBlur(g, (0, 0), cfg.blur_sigma) if cfg.blur_sigma > 0 else g


def centre(img):
    h, w = img.shape[:2]
    return (w - 1) / 2.0, (h - 1) / 2.0


def rotate(img, angle_ccw):
    m = cv2.getRotationMatrix2D(centre(img), angle_ccw, 1.0)
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderValue=0)


def disc_mask(shape, radius):
    h, w = shape[:2]
    yy, xx = np.mgrid[:h, :w]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2


class NccMatcher:
    def __init__(self, base, cfg):
        self.cfg = cfg
        self.mask = disc_mask(base.shape, cfg.radius)
        ref = to_gray(base, cfg)[self.mask]
        ref = ref - ref.mean()
        self.ref = ref / (np.linalg.norm(ref) + 1e-8)

    def ncc(self, gray, angle):
        v = rotate(gray, -angle)[self.mask]
        v = v - v.mean()
        return float(v @ self.ref / (np.linalg.norm(v) + 1e-8))

    def estimate(self, img):
        cfg = self.cfg
        gray = to_gray(img, cfg)
        coarse = np.arange(0.0, 360.0, cfg.coarse_step)
        best = coarse[int(np.argmax([self.ncc(gray, a) for a in coarse]))]
        fine = best + np.arange(-cfg.fine_half_width, cfg.fine_half_width + 1e-9, cfg.fine_step)
        scores = np.array([self.ncc(gray, a) for a in fine])
        k = int(np.argmax(scores))
        angle = fine[k]
        if 0 < k < len(fine) - 1:
            y0, y1, y2 = scores[k - 1:k + 2]
            denom = y0 - 2 * y1 + y2
            if denom < 0:
                angle += 0.5 * (y0 - y2) / denom * cfg.fine_step
        return float(angle % 360.0), float(scores[k])


class SiftMatcher:
    def __init__(self, base, cfg):
        self.cfg = cfg
        self.sift = cv2.SIFT_create()
        self.mask = disc_mask(base.shape, cfg.radius + 8).astype(np.uint8) * 255
        gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        self.kp, self.des = self.sift.detectAndCompute(gray, None)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    def estimate(self, img):
        nan = {"angle": math.nan, "inliers": 0, "scale": math.nan, "shift": math.nan}
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = self.sift.detectAndCompute(gray, self.mask)
        if des is None or len(kp) < self.cfg.min_inliers or self.des is None:
            return nan
        good = [m for m, n in (p for p in self.matcher.knnMatch(des, self.des, k=2) if len(p) == 2)
                if m.distance < self.cfg.sift_ratio * n.distance]
        if len(good) < self.cfg.min_inliers:
            return nan
        src = np.float32([kp[m.queryIdx].pt for m in good])
        dst = np.float32([self.kp[m.trainIdx].pt for m in good])
        m, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=self.cfg.ransac_px)
        if m is None or inl is None or int(inl.sum()) < self.cfg.min_inliers:
            return nan
        cx, cy = centre(img)
        mapped = m @ np.array([cx, cy, 1.0])
        return {"angle": float(math.degrees(math.atan2(m[1, 0], m[0, 0])) % 360.0), "inliers": int(inl.sum()),
                "scale": float(math.hypot(m[0, 0], m[1, 0])),
                "shift": float(math.hypot(mapped[0] - cx, mapped[1] - cy))}


def to_label(angle, label_max):
    a = angle - 360.0 if angle > 270.0 else angle
    return float(min(max(a, 0.0), label_max)), not (-5.0 <= a <= label_max + 5.0)


def angle_error(a, b):
    d = (a - b) % 360.0
    return min(d, 360.0 - d)


def synthetic_set(base, cfg):
    rng = np.random.default_rng(cfg.seed)
    out = []
    for theta in rng.uniform(0.0, cfg.label_max, cfg.n_synthetic):
        img = rotate(base, float(theta))
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
        assert ok
        out.append((float(theta), cv2.imdecode(buf, cv2.IMREAD_COLOR)))
    return out
