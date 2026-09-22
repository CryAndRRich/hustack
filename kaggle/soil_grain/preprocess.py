import os
import warnings
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from skimage.morphology import disk, opening as morph_opening

warnings.filterwarnings("ignore")

GRAIN_DIAMETERS = np.array([0.002, 0.006, 0.020, 0.063, 0.200, 0.630,
                             2.000, 6.300, 20.00, 63.00, 200.0])
GRAIN_NAMES = [
    "Clay", "Fine silt", "Med silt", "Coarse silt",
    "Fine sand", "Med sand", "Coarse sand",
    "Fine gravel", "Med gravel", "Coarse gravel", "Cobbles",
]
IMG_SIZE = 512
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class Data:
    def __init__(self, data_path: str) -> None:
        self.data_dir = os.path.join(os.getcwd(), data_path)
        self.train_df = pd.read_csv(os.path.join(self.data_dir, "train.csv"))
        self.test_df = pd.read_csv(os.path.join(self.data_dir, "test.csv"))
        self.sample_sub = pd.read_csv(os.path.join(self.data_dir, "sample_submission.csv"))

        _non_target = {
            "image_id", "sample_id", "id", "filename", "image_path", "state", "split"
        }
        self.target_cols = [
            c for c in self.sample_sub.columns
            if c.lower() not in _non_target and c in self.train_df.columns
        ]

        for col in ["image_id", "filename", "id"]:
            if col in self.train_df.columns:
                self.img_col = col
                break
        else:
            self.img_col = self.train_df.columns[0]

        self.sample_id_col = (
            "sample_id" if "sample_id" in self.train_df.columns else self.img_col
        )

        print(f"Train: {len(self.train_df)} images | Test: {len(self.test_df)} images")
        print(f"Unique samples: {self.train_df[self.sample_id_col].nunique()}")
        print(f"Target cols ({len(self.target_cols)}): {self.target_cols}")

    def img_path(self, row, split: str = "train") -> str:
        fname = str(row[self.img_col])
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            fname += ".jpg"
        for base in [
            os.path.join(self.data_dir, split, "images"),
            os.path.join(self.data_dir, split),
            self.data_dir,
        ]:
            p = os.path.join(base, fname)
            if os.path.exists(p):
                return p
        return os.path.join(self.data_dir, split, fname)


class MorphGranulometry:
    N_LEVELS = 11

    def __call__(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape
        max_r = max(min(h, w) // 8, 2)
        radii = np.unique(
            np.logspace(0, np.log10(max_r), self.N_LEVELS).astype(int)
        )

        img = gray.astype(np.float32) / 255.0
        total = img.sum() + 1e-9
        feats = np.zeros(self.N_LEVELS, dtype=np.float32)
        prev = total

        for i in range(self.N_LEVELS):
            r = int(radii[min(i, len(radii) - 1)])
            opened = morph_opening(img, disk(r)).astype(np.float32)
            curr = opened.sum()
            feats[i] = max(0.0, (prev - curr) / total)
            prev = curr

        s = feats.sum()
        return feats / s if s > 0 else feats

_TRAIN_AUG = T.Compose([
    T.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomApply([T.RandomRotation(180)], p=0.5),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

_EVAL_AUG = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class SoilDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        data: Data,
        split: str = "train",
        augment: bool = True,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.data = data
        self.split = split
        self.transform = _TRAIN_AUG if augment else _EVAL_AUG
        self.granulometry = MorphGranulometry()
        self.is_train = split == "train"

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        try:
            img = Image.open(self.data.img_path(row, self.split)).convert("RGB")
        except Exception:
            img = Image.fromarray(np.full((IMG_SIZE, IMG_SIZE, 3), 128, dtype=np.uint8))

        gray = np.array(img.convert("L").resize((256, 256), Image.BILINEAR))
        domain_feat = torch.tensor(self.granulometry(gray), dtype=torch.float32)

        img_tensor = self.transform(img)

        if self.is_train:
            target = torch.tensor(
                row[self.data.target_cols].values.astype(np.float32),
                dtype=torch.float32,
            )
            return img_tensor, domain_feat, target
        return img_tensor, domain_feat

if __name__ == "__main__":
    data = Data("soil_grain/data")
    print("Sample submission columns:", data.sample_sub.columns.tolist())
