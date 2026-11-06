import gzip
import os
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
from scipy.stats import rankdata

DATA_URL = "https://storage.googleapis.com/open-courses/austral2026-5da5/labo1/analistasr_competencia_2026.csv.gz"
DATA_NAME = "analistasr_competencia_2026.csv"
MACRO_NAME = "ext_macro.csv"
ID, PERIOD, TARGET = "numero_de_cliente", "foto_mes", "clase_ternaria"
CLASSES = ("CONTINUA", "BAJA+1", "BAJA+2")
FUTURE = 202109
N_FUTURE = 165076
N_COLS = 155
HIT, MISS = 975_000, -25_000
KAGGLE_INPUT = Path("/kaggle/input")


_A, _B = (202006,), (201905, 201910, 202006)
_D = (201910, 202002, 202006, 202009, 202010, 202102)
COURSE_BROKEN = {
    **dict.fromkeys(["mrentabilidad", "mrentabilidad_annual", "mcomisiones", "mactivos_margen", "mpasivos_margen",
                     "ccomisiones_otras", "mcomisiones_otras"], _B),
    **dict.fromkeys(["ccajeros_propios_descuentos", "mcajeros_propios_descuentos", "ctarjeta_visa_descuentos",
                     "mtarjeta_visa_descuentos", "ctarjeta_master_descuentos", "mtarjeta_master_descuentos"], _D),
    **dict.fromkeys(["active_quarter", "internet", "mcuentas_saldo", "ctarjeta_debito_transacciones", "mautoservicio",
                     "ctarjeta_visa_transacciones", "mtarjeta_visa_consumo", "ctarjeta_master_transacciones",
                     "mtarjeta_master_consumo", "cextraccion_autoservicio", "mextraccion_autoservicio",
                     "ccheques_depositados", "mcheques_depositados", "ccheques_emitidos", "mcheques_emitidos",
                     "ccheques_depositados_rechazados", "mcheques_depositados_rechazados",
                     "ccheques_emitidos_rechazados", "mcheques_emitidos_rechazados", "tcallcenter",
                     "ccallcenter_transacciones", "thomebanking", "ccajas_transacciones", "ccajas_consultas",
                     "ccajas_extracciones", "ccajas_otras", "catm_trx", "matm", "catm_trx_other", "matm_other"], _A),
    "ctarjeta_visa_debitos_automaticos": (201904,),
    "mttarjeta_visa_debitos_automaticos": (201904,),
    "chomebanking_transacciones": (201910, 202006),
    "ccajas_depositos": (202006, 202105),
}

_PRE_201906 = (201901, 201902, 201903, 201904, 201905)
EXTRA_BROKEN = {
    **dict.fromkeys(["ctransferencias_recibidas", "mtransferencias_recibidas"], _PRE_201906),
    **dict.fromkeys(["tmobile_app", "cmobile_app_trx"], _PRE_201906 + (201906, 202006)),
}

MONTH_RANK = ("Visa_fultimo_cierre", "Master_fultimo_cierre")


def data_dir():
    try:
        return Path(__file__).resolve().parent.parent / "data"
    except NameError:
        return Path.cwd() / "data"


def out_dir():
    return Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")


def find_input(prefix):
    roots = [KAGGLE_INPUT] if KAGGLE_INPUT.exists() else []
    roots.append(data_dir())
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            for f in sorted(filenames):
                if f.startswith(prefix):
                    return Path(dirpath) / f
    return None


def dataset_path():
    path = find_input(DATA_NAME)
    if path is not None:
        print(f"dataset: {path}")
        return path
    tmp = Path("/kaggle/temp") if Path("/kaggle/temp").exists() else Path(tempfile.gettempdir())
    path = tmp / (DATA_NAME + ".gz")
    if not path.exists():
        print(f"dataset not attached; downloading {DATA_URL} (needs Internet ON)", flush=True)
        t = time.time()
        urllib.request.urlretrieve(DATA_URL, path)
        print(f"downloaded {path.stat().st_size / 1e6:.0f} MB in {time.time() - t:.0f}s")
    return path


def header(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [c.strip().strip('"') for c in f.readline().rstrip("\n").split(",")]


def load():
    path = dataset_path()
    cols = header(path)
    assert len(cols) == N_COLS and {ID, PERIOD, TARGET} <= set(cols), (len(cols), cols[:5], cols[-3:])
    types = {c: pa.float32() for c in cols}
    types.update({ID: pa.int64(), PERIOD: pa.int32(), TARGET: pa.string()})
    t = time.time()
    tbl = pacsv.read_csv(str(path), convert_options=pacsv.ConvertOptions(column_types=types, strings_can_be_null=True))
    df = tbl.to_pandas()
    del tbl
    df = df.sort_values([ID, PERIOD], kind="stable").reset_index(drop=True)
    assert not df.duplicated([ID, PERIOD]).any(), "duplicate (client, month) rows"
    bad = set(df[TARGET].dropna().unique()) - set(CLASSES)
    assert not bad, f"unexpected classes {bad}"
    n_future = int((df[PERIOD] == FUTURE).sum())
    print(f"loaded {df.shape} in {time.time() - t:.0f}s, {df.memory_usage().sum() / 1e9:.2f} GB; "
          f"{FUTURE} rows {n_future} (page: {N_FUTURE})", flush=True)
    if n_future != N_FUTURE:
        print(f"WARNING: {FUTURE} has {n_future} rows, the submission needs {N_FUTURE}")
    return df


def month_index(period):
    period = np.asarray(period)
    return (period // 100) * 12 + period % 100 - 1


def load_macro():
    path = find_input(MACRO_NAME)
    if path is None:
        print(f"{MACRO_NAME} not attached; macro checks skipped")
        return None
    m = pd.read_csv(path)
    print(f"macro: {path} {m.shape}")
    return m.set_index(PERIOD)


def auc(x, y):
    npos = int(y.sum())
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return np.nan
    r = rankdata(np.where(np.isnan(x), -np.inf, x))
    return (r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def is_money(col):
    return col.startswith("m") or col.startswith(("Master_m", "Visa_m"))


def clean(df):
    cells = {c: set(m) for c, m in COURSE_BROKEN.items()}
    for c, m in EXTRA_BROKEN.items():
        cells.setdefault(c, set()).update(m)
    for col, months in cells.items():
        df.loc[df[PERIOD].isin(sorted(months)), col] = np.nan
    for col in MONTH_RANK:
        df[col] = df.groupby(PERIOD)[col].rank(pct=True).astype(np.float32)
    print(f"clean: {sum(len(m) for m in cells.values())} (column, month) cells set to NA; "
          f"within-month rank for {', '.join(MONTH_RANK)}")
    return df


class Features:
    def __init__(self, df, money_rank=False, like=None):
        self.cols = [c for c in df.columns if c not in (ID, PERIOD, TARGET)]
        self.base = df[self.cols].to_numpy(np.float32)
        if money_rank:
            g = df.groupby(PERIOD)
            for k, col in enumerate(self.cols):
                if is_money(col):
                    self.base[:, k] = g[col].rank(pct=True).to_numpy(np.float32)
        if like is not None:
            self.prev, self.hist = like.prev, like.hist
        else:
            pidx = month_index(df[PERIOD].to_numpy())
            keys = df[ID].to_numpy() * 1000 + (pidx - pidx.min())
            index = pd.Index(keys)
            assert index.is_unique
            self.prev = [index.get_indexer(keys - k) for k in (1, 2)]
            self.hist = df.groupby(ID, sort=False).cumcount().to_numpy(np.float32)
        self.names = self.cols + [f"{c}_delta{k}" for k in (1, 2) for c in self.cols] + ["hist_n"]

    def matrix(self, rows):
        k = len(self.cols)
        out = np.empty((len(rows), len(self.names)), np.float32)
        out[:, :k] = self.base[rows]
        for i, prev in enumerate(self.prev, start=1):
            j = prev[rows]
            lag = self.base[np.maximum(j, 0)]
            lag[j < 0] = np.nan
            np.subtract(out[:, :k], lag, out=out[:, i * k:(i + 1) * k])
        out[:, -1] = self.hist[rows]
        return out
