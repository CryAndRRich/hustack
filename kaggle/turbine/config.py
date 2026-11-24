K = 10_000
NONINSTALL_KEEP = 0.30
VAL_DAYS = 7
MIN_PRESENT = 100
CHUNK = 100_000
TRAIN_ROW_CAP = 870_000
CAND_ROW_CAP = 1_010_000
CAT_PREFIX = "cat_"

ADV_ROWS = 300_000
ADV_ROUNDS = 400
IW_CLIP_Q = 0.99

CANDIDATE_ARM = "tweedie"
BOOTSTRAP = 1000
GATE_WIN = 0.95

LGB_BASE = {
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 255,
    "verbose": -1,
    "seed": 0,
}
TWEEDIE_POWER = 1.5
MAX_ROUNDS = 2000
PATIENCE = 150
REFIT_FACTOR = 1.1

SMOKE_TRAIN_ROWS = 60_000
SMOKE_CAND_ROWS = 30_000
SMOKE_ROUNDS = 15

SEED = 0
WORK = "/tmp/work"
