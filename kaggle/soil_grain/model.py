# Updated on July 9, 2026, 1:58 AM
# Public Score: 82.56482
# Rank: 40/50

import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import timm
import optuna
from sklearn.model_selection import GroupKFold

from preprocess import Data, SoilDataset, GRAIN_DIAMETERS

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

_LOG_D = np.log10(GRAIN_DIAMETERS)
_W = np.zeros(11)
_W[0] = (_LOG_D[1] - _LOG_D[0]) / 2
for _i in range(1, 10):
    _W[_i] = (_LOG_D[_i + 1] - _LOG_D[_i - 1]) / 2
_W[10] = (_LOG_D[10] - _LOG_D[9]) / 2
LOG_WEIGHTS = _W


class LogEMDLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weights", torch.tensor(LOG_WEIGHTS, dtype=torch.float32))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (torch.abs(pred - target) * self.weights).sum(dim=1).mean()


class SoilGrainNet(nn.Module):
    def __init__(
        self,
        backbone: str = "efficientnet_b4",
        pretrained: bool = True,
        domain_dim: int = 11,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        feat_dim = self.encoder.num_features

        self.head = nn.Sequential(
            nn.Linear(feat_dim + domain_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, 11),
        )

    def forward(self, img: torch.Tensor, domain_feat: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(img)                         
        feat = torch.cat([feat, domain_feat], dim=1)      
        logits = self.head(feat)                          

        increments = F.softmax(logits, dim=1)             
        csd = torch.cumsum(increments, dim=1) * 100.0    
        return csd


class SoilGrainModel:
    BATCH_SIZE = 16
    N_WORKERS = 2
    EPOCHS = 25

    def __init__(self, data_path: str) -> None:
        self.data_dir = os.path.join(os.getcwd(), data_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.data = Data(data_path)
        self.train_df = self.data.train_df
        self.test_df = self.data.test_df
        self.y = self.train_df[self.data.target_cols].values.astype(np.float32)
        self.groups = self.train_df[self.data.sample_id_col].values

        print(f"Device: {self.device}")
        print(f"Train: {len(self.train_df)} rows | Test: {len(self.test_df)} rows")

    def _loader(self, df, split, augment, shuffle):
        ds = SoilDataset(df, self.data, split=split, augment=augment)
        return DataLoader(
            ds, batch_size=self.BATCH_SIZE, shuffle=shuffle,
            num_workers=self.N_WORKERS, pin_memory=True,
        )

    def _train_epoch(self, model, loader, optimizer, scaler, criterion):
        model.train()
        total = 0.0
        for img, dom, tgt in loader:
            img, dom, tgt = img.to(self.device), dom.to(self.device), tgt.to(self.device)
            optimizer.zero_grad()
            with autocast():
                loss = criterion(model(img, dom), tgt)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
        return total / len(loader)

    @torch.no_grad()
    def _validate(self, model, loader, criterion):
        model.eval()
        total = 0.0
        preds = []
        for img, dom, tgt in loader:
            img, dom, tgt = img.to(self.device), dom.to(self.device), tgt.to(self.device)
            with autocast():
                pred = model(img, dom)
                total += criterion(pred, tgt).item()
            preds.append(pred.cpu().float().numpy())
        return total / len(loader), np.concatenate(preds, axis=0)

    @torch.no_grad()
    def _predict(self, model, loader):
        model.eval()
        preds = []
        for img, dom in loader:
            img, dom = img.to(self.device), dom.to(self.device)
            with autocast():
                preds.append(model(img, dom).cpu().float().numpy())
        return np.concatenate(preds, axis=0)

    def _optuna_search(self, n_trials: int = 20) -> dict:
        gkf = GroupKFold(n_splits=3)

        def objective(trial):
            lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
            dropout = trial.suggest_float("dropout", 0.1, 0.4)

            tr_idx, va_idx = next(gkf.split(self.train_df, self.y, self.groups))
            model = SoilGrainNet(dropout=dropout).to(self.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            scaler = GradScaler()
            criterion = LogEMDLoss().to(self.device)
            tr_loader = self._loader(self.train_df.iloc[tr_idx], "train", True,  True)
            va_loader = self._loader(self.train_df.iloc[va_idx], "train", False, False)

            for _ in range(5):
                self._train_epoch(model, tr_loader, optimizer, scaler, criterion)
            val_loss, _ = self._validate(model, va_loader, criterion)
            return val_loss

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        print(f"Best trial EMD: {study.best_value:.4f} | params: {study.best_params}")
        return study.best_params

    def train(self, n_trials: int = 20, n_splits: int = 5) -> None:
        best = self._optuna_search(n_trials)
        lr = best.get("lr", 3e-4)
        dropout = best.get("dropout", 0.2)

        gkf = GroupKFold(n_splits=n_splits)
        criterion = LogEMDLoss().to(self.device)

        self.oof_preds = np.zeros_like(self.y)
        test_preds = np.zeros((len(self.test_df), len(self.data.target_cols)))

        for fold, (tr_idx, va_idx) in enumerate(
            gkf.split(self.train_df, self.y, self.groups)
        ):
            print(f"\nFold {fold + 1}/{n_splits}")
            model = SoilGrainNet(dropout=dropout).to(self.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.EPOCHS, eta_min=lr / 100
            )
            scaler = GradScaler()
            tr_loader = self._loader(self.train_df.iloc[tr_idx], "train", True,  True)
            va_loader = self._loader(self.train_df.iloc[va_idx], "train", False, False)
            te_loader = self._loader(self.test_df,               "test",  False, False)

            best_val = float("inf")
            for epoch in range(self.EPOCHS):
                tr_loss = self._train_epoch(model, tr_loader, optimizer, scaler, criterion)
                val_loss, va_preds = self._validate(model, va_loader, criterion)
                scheduler.step()

                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(model.state_dict(), f"best_fold{fold}.pt")
                    self.oof_preds[va_idx] = va_preds

                if (epoch + 1) % 5 == 0:
                    print(f"  Epoch {epoch+1:3d} | train={tr_loss:.4f} | val={val_loss:.4f}")

            model.load_state_dict(
                torch.load(f"best_fold{fold}.pt", map_location=self.device)
            )
            test_preds += self._predict(model, te_loader) / n_splits

        oof_emd = float(
            (np.abs(self.oof_preds - self.y) * LOG_WEIGHTS).sum(axis=1).mean()
        )
        print(f"\nOOF EMD (x100 => competition score): {oof_emd:.4f} ({oof_emd * 100:.2f})")
        self.raw_test_preds = test_preds

    def _postprocess(self, preds: np.ndarray) -> np.ndarray:
        return np.maximum.accumulate(preds, axis=1)

    def predict(self) -> None:
        final_preds = self._postprocess(self.raw_test_preds)

        sub = self.data.sample_sub.copy()
        for i, col in enumerate(self.data.target_cols):
            sub[col] = final_preds[:, i]

        out = os.path.join(self.data_dir, "soil_grain_submission.csv")
        sub.to_csv(out, index=False)
        print(f"Submission saved => soil_grain_submission.csv ({len(sub)} rows)")

    def feature_importance_proxy(self) -> pd.DataFrame:
        df = self.train_df.copy()
        df["oof_emd"] = (
            np.abs(self.oof_preds - self.y) * LOG_WEIGHTS
        ).sum(axis=1)
        by_sample = (
            df.groupby(self.data.sample_id_col)["oof_emd"]
            .mean()
            .sort_values(ascending=False)
        )
        print("Hardest samples (highest OOF EMD):")
        print(by_sample.head(10).to_string())
        return by_sample


if __name__ == "__main__":
    data_path = "soil_grain/data"
    model = SoilGrainModel(data_path)
    model.train(n_trials=20, n_splits=5)
    model.predict()
    model.feature_importance_proxy()