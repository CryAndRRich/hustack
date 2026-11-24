from datetime import datetime, timedelta
from pathlib import Path

TEAM = "Namhair"
SUBMISSION_NO = 3
DTFMT = "%Y-%m-%d %H:%M:%S"

CASES = ["AO_WPS", "AO_noWPS", "AGC_WPS", "AGC_noWPS", "PO_WPS", "PO_noWPS", "PGC_WPS", "PGC_noWPS"]
CORRIDORS = {
    "atlantic": {"prefix": "A", "src": (43.6, -4.0), "dst": (40.6, -69.0), "passage_h": 354},
    "pacific": {"prefix": "P", "src": (34.8, 140.0), "dst": (34.4, -121.0), "passage_h": 583},
}
DEPARTURES = [datetime(2024, 1, 1, 12) + timedelta(days=d) for d in range(366)]
T0 = datetime(2024, 1, 1, 0)
WP_PER_H = 4

MAX_WIND = 20.0
MAX_HS = 7.0
EARTH_R = 6371000.0
NM = 1852.0

BOARD_BASELINE = {
    "AO_WPS": 92614.2, "AO_noWPS": 113498.6, "AGC_WPS": 92614.2, "AGC_noWPS": 113498.6,
    "PO_WPS": 57778.74, "PO_noWPS": 92231.98, "PGC_WPS": 57778.74, "PGC_noWPS": 92231.98,
}

VARIANTS = {
    "A_trap_cubic_raw": ("trap", 3, "raw"),
    "B_trap_cubic_sincos": ("trap", 3, "sincos"),
    "C_mid_linear_sincos": ("mid", 1, "sincos"),
    "D_trap_linear_raw": ("trap", 1, "raw"),
}
OPT_VARIANT = "B_trap_cubic_sincos"

GC_DP = {"step_nm": 0.25, "stage_h": 0.5, "vmax_kn": 25.0}
O_DP = {"dalpha_deg": 0.0667, "dbeta_deg": 0.0667, "beta_max_deg": 16.0, "stage_h": 2, "vmax_kn": 16.0,
        "node_land_km": 5.0, "edge_land_km": 2.0, "box_margin": 0.25, "hs_lim": 6.95, "tws_lim": 19.9,
        "big": 1.0e4, "dilate": False}
BANDS = [(-16.0, -3.0), (-5.0, 5.0), (3.0, 16.0)]
BAND_SPAN = (0.2, 0.8)
TUBE = {"step_nm": 2.0, "half_nm": 30.0, "stage_h": 1, "vmax_kn": 20.0, "node_land_km": 4.0, "edge_land_km": 2.0,
        "box_margin": 0.25, "slope_max": 1.5}
TUBE2 = dict(TUBE, step_nm=1.0, half_nm=12.0, stage_h=0.5)
NEWTON = {"stages": [(20.0, 0.05, 1.0e3, 30), (5.0, 0.02, 1.0e4, 30), (1.0, 0.01, 1.0e5, 30),
                     (0.1, 0.002, 1.0e6, 30)],
          "lam0": 1.0e-3, "max_step_deg": 0.5, "hs_lim": 6.98, "tws_lim": 19.98, "land_km": 2.0, "box_margin": 0.25,
          "feas": {"hs": 6.985, "tws": 19.985, "land_km": 0.5}}
NEWTON_FINE = [(0.1, 0.002, 1.0e6, 30), (0.02, 0.001, 1.0e6, 30)]
NEWTON_CHUNK = 64
SELECT = {"hs_max": 6.99, "tws_max": 19.99}
V2_SUB = 2
LAND_RES_DEG = 0.02
BATCH_DAYS = 16
BUDGET_H = 10.0
RESERVE_H = 0.45

SMOKE = {"days": [0, 1], "gc_step_nm": 2.0, "dalpha_deg": 0.4, "dbeta_deg": 0.4, "beta_max_deg": 8.0,
         "tube_step_nm": 8.0, "stages": [(20.0, 0.05, 1.0e3, 2), (1.0, 0.01, 1.0e5, 2)]}

WORK = Path("/tmp/work")
OUT = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(".")
