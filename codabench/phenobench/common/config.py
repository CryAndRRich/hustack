import dataclasses
import json
from dataclasses import dataclass, field

FINGERPRINTS = {"plant_m2f": "090330b69175a2b6267d39c389077324",
                "leaf_eomt": "5695010d45eb914495611e2f5ef4b2d0",
                "plant_zoom_m2f": "f6eeeaeb9215c8e4ab71e2d49245524c",
                "crop_cls/rank0.pt": "7d242e903fda42f155e0162138c86284",
                "crop_cls/rank1.pt": "4880d93d7b9ecd60984df34554d2f095",
                "sem_m2f": "937f790b0becba947a958eee47e618ec"}


def roles(*names):
    return field(default_factory=lambda: {n: FINGERPRINTS[n] for n in names})


def map_thresholds():
    return [round(0.05 * i, 2) for i in range(10, 19)] + [0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]


@dataclass
class Config:
    data_root: str = ""
    out_dir: str = "/kaggle/working"
    work_dir: str = "/tmp/work"
    best_models_dir: str = ""
    best_fingerprints: dict = roles("plant_m2f", "leaf_eomt")
    session_target_hours: float = 10.0
    session_hours: float = 11.0
    min_train_hours: float = 2.0
    max_train_hours: float = 8.0
    margin: float = 1.2
    train_hours: float = 0.0
    zoom_min: float = 1.6
    zoom_max: float = 4.0
    m2f_size: int = 1024
    m2f_stride: int = 2
    eomt_size: int = 768
    eomt_stride: int = 1
    accum: int = 4
    blur_p: float = 0.2
    paste_p: float = 0.5
    paste_max: int = 6
    small_area: int = 2500
    small_weight: float = 3.0
    plant_scale: float = 1.25
    tile_batch: int = 2
    max_cands: int = 150
    z2_gate: int = 5000
    z3_gates: list = field(default_factory=lambda: [1000, 2500])
    size_split: int = 500
    small_offsets: list = field(default_factory=lambda: [0.0, -0.15, -0.1, -0.05, 0.02, 0.04])
    leaf_size: int = 512
    leaf_fill: float = 0.85
    leaf_zoom_min: float = 0.5
    leaf_zoom_max: float = 16.0
    leaf_dilate: int = 3
    leaf_infer_batch: int = 32
    leaf_views: int = 4
    min_plant_area: int = 16
    leaf_gates: list = field(default_factory=lambda: [0.9, 0.92, 0.94, 0.96])
    lr_plant: float = 5e-5
    lr_eomt: float = 5e-5
    llrd: float = 0.8
    backbone_mult_plant: float = 0.1
    weight_decay: float = 0.05
    clip_norm: float = 0.01
    warmup_steps: int = 100
    swa_cycles: int = 4
    swa_min_frac: float = 0.05
    mem_limit_m2f: float = 0.95
    mem_limit_eomt: float = 0.9
    num_workers: int = 2
    quick_eval_images: int = 100
    min_score: float = 0.2
    keep_frac: float = 0.7
    fuse_iou: float = 0.5
    stitch_iou: float = 0.5
    thresholds: list = field(default_factory=lambda: [round(0.05 * i, 2) for i in range(4, 19)] + [0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99])
    smoke_steps: int = 2
    smoke_images: int = 6
    metric_check_images: int = 20
    seed: int = 0
    plant_candidate: str = "z2_m"
    plant_threshold: float = 0.92
    plant_offset: float = 0.04
    cls_backbone: str = "facebook/dinov2-base"
    cls_size: int = 224
    cls_context: float = 2.0
    cls_min_side: int = 64
    cls_dim: float = 0.4
    cls_jitter: int = 3
    cls_batch: int = 32
    cls_hours: float = 1.5
    cls_lr: float = 2e-5
    cls_head_lr: float = 1e-3
    cls_clip: float = 1.0
    crop_thresholds: list = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    sem_zoom_min: float = 1.0
    sem_zoom_max: float = 2.5
    sem_weed_p: float = 0.5
    sem_tile: int = 640
    weed_factors: list = field(default_factory=lambda: [0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    sem_views: dict = field(default_factory=lambda: {"g": [1024, 1344], "t": [672, 1344], "u": [512, 1344]})

    def to_json(self):
        return json.dumps(dataclasses.asdict(self))

    @classmethod
    def from_json(cls, s):
        return cls(**json.loads(s))
