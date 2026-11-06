from pathlib import Path

import numpy as np
import pandas as pd

MARKER = "reserve_2016Q3_train.csv"
EXT = "ext_slee.csv"
TARGET = "NumReserveDays2016Q3"
ORIGIN = pd.Timestamp("2016-01-01")
Q2, Q3 = 91, 182
LEAD = (pd.Timestamp("2016-10-15") - ORIGIN).days - Q3
PRICE_CAP = 5000
CATS = ["PropertyType", "ListingType", "Neighborhood", "CancellationPolicy", "Superhost", "InstantbookEnabled",
        "BusinessReady", "status_last"]
DROP = ["HostID", "ListingTitle", "CreatedDate", "Country", "State", "City", "MetropolitanStatisticalArea",
        "NumberofReviews"]

def find(name):
    root = Path("/kaggle/input")
    hits = sorted(root.rglob(name)) if root.exists() else []
    if hits:
        return hits[0]
    try:
        return Path(__file__).resolve().parent.parent / "data" / name
    except NameError:
        return Path.cwd() / "data" / name

def load_listings(d):
    dtype = {"PropertyID": "int32", "Status": str, "Price": "float32", "BookedDate": str, "ReservationID": "float64"}
    L = pd.concat([pd.read_csv(d / f"listing_2016{q}.csv", dtype=dtype) for q in ("Q1", "Q2")], ignore_index=True)
    assert set(L.Status.unique()) <= {"A", "B", "R"}, L.Status.unique()
    date = pd.to_datetime(L.Date, format="%Y-%m-%d")
    L["day"] = (date - ORIGIN).dt.days.astype("int16")
    L["month"] = date.dt.month.astype("int8")
    L["booked"] = (pd.to_datetime(L.BookedDate, format="%Y-%m-%d") - ORIGIN).dt.days
    L["Price"] = L.Price.clip(upper=PRICE_CAP)
    for s in "RAB":
        L[s] = L.Status.eq(s).astype("int8")
    assert not L.duplicated(["PropertyID", "day"]).any()
    return L.drop(columns=["Date", "BookedDate"])

def reservations(L):
    res = L[L.ReservationID.notna()].groupby("ReservationID").agg(
        pid=("PropertyID", "first"), npid=("PropertyID", "nunique"), start=("day", "min"), end=("day", "max"),
        nights=("day", "size"), booked=("booked", "min"))
    print(f"reservations {len(res)}, spanning >1 property {(res.npid > 1).sum()}")
    res["lead"] = res.start - res.booked
    return res

def listing_features_abs(L, res):
    g = L.groupby("PropertyID")
    F = pd.DataFrame({"n_days": g.size(), "first_day": g.day.min()})
    m = L.pivot_table(index="PropertyID", columns="month", values=["R", "A", "B"], aggfunc="sum", fill_value=0)
    m.columns = [f"{s}_m{k}" for s, k in m.columns]
    F = F.join(m)
    for q, months in (("q1", (1, 2, 3)), ("q2", (4, 5, 6))):
        for s in "RAB":
            F[f"{s}_{q}"] = F[[f"{s}_m{k}" for k in months]].sum(axis=1)
        F[f"occ_{q}"] = F[f"R_{q}"] / (F[f"R_{q}"] + F[f"A_{q}"]).replace(0, np.nan)
    F["occ_m6"] = F.R_m6 / (F.R_m6 + F.A_m6).replace(0, np.nan)
    for w in (7, 14):
        F = F.join(L[L.day >= Q3 - w].groupby("PropertyID")[["R", "A", "B"]].sum().add_suffix(f"_last{w}"))
    F["status_last"] = L[L.day == Q3 - 1].set_index("PropertyID").Status.reindex(F.index).fillna("none")
    q1, q2 = L[L.day < Q2], L[L.day >= Q2]
    F["price_med_q1"] = q1.groupby("PropertyID").Price.median()
    F["price_med_q2"] = q2.groupby("PropertyID").Price.median()
    F["price_min_q2"] = q2.groupby("PropertyID").Price.min()
    F["price_max_q2"] = q2.groupby("PropertyID").Price.max()
    F["price_med_A_q2"] = q2[q2.A == 1].groupby("PropertyID").Price.median()
    F["price_med_R_q2"] = q2[q2.R == 1].groupby("PropertyID").Price.median()
    F["price_ratio_q2_q1"] = F.price_med_q2 / F.price_med_q1
    r, r2 = res, res[res.start >= Q2]
    g, g2 = r.groupby("pid"), r2.groupby("pid")
    F["res_n_q1"] = r[r.start < Q2].groupby("pid").size()
    F["res_n_q2"] = g2.size()
    F["res_nights_mean"] = g.nights.mean()
    F["res_nights_mean_q2"] = g2.nights.mean()
    F["res_lead_mean"] = g.lead.mean()
    F["res_lead_med_q2"] = g2.lead.median()
    F["res_since_last_booking"] = Q3 - g.booked.max()
    for w in (30, 60):
        recent = r[r.booked >= Q3 - w].groupby("pid")
        F[f"res_n_booked_last{w}"] = recent.size()
        F[f"res_nights_booked_last{w}"] = recent.nights.sum()
    open_stay = r[r.end == Q3 - 1].groupby("pid")
    F["res_open_lead"] = open_stay.lead.min()
    F["res_open_nights"] = open_stay.nights.max()
    F["r_no_id"] = L[(L.R == 1) & L.ReservationID.isna()].groupby("PropertyID").size()
    cnt = [c for c in F.columns if c.startswith(("res_n_", "res_nights_booked", "r_no_id"))]
    F[cnt] = F[cnt].fillna(0)
    return F

def slee_features_abs(S, nr):
    rv, pr = S.filter(like="rv_"), S.filter(like="pr_")
    dd = (pd.to_datetime(rv.columns.str[3:], format="%Y%m%d") - ORIGIN).days.to_numpy()
    oct1 = (pd.Timestamp("2016-10-01") - ORIGIN).days
    pre, q3, post = dd < Q3, (dd >= Q3) & (dd < oct1), dd >= oct1
    have = rv.notna()
    hp = have.loc[:, pre].to_numpy()
    last_i = hp.shape[1] - 1 - hp[:, ::-1].argmax(1)
    F = pd.DataFrame(index=S.index)
    F["sv_last_pre"] = rv.loc[:, pre].ffill(axis=1).iloc[:, -1]
    F["sv_pre_gap"] = np.where(hp.any(1), Q3 - dd[pre][last_i], np.nan)
    F["sv_new_nr"] = nr - F.sv_last_pre
    F["sv_new_nr_rate"] = F.sv_new_nr / (F.sv_pre_gap + LEAD)
    F["sv_new_post"] = rv.loc[:, post].bfill(axis=1).iloc[:, 0] - F.sv_last_pre
    cols = [c for c in rv.columns if c >= "rv_20160324"]
    for a, b in zip(cols, cols[1:]):
        F[f"sv_d_{a[5:]}_{b[5:]}"] = rv[b] - rv[a]
    F["sv_d_0324_0623"] = rv.rv_20160623 - rv.rv_20160324
    F["sv_d_0623_0920"] = rv.rv_20160920 - rv.rv_20160623
    F["sv_n_pre"], F["sv_n_q3"], F["sv_n_post"] = have.loc[:, pre].sum(1), have.loc[:, q3].sum(1), have.loc[:, post].sum(1)
    F["sv_in_0623"] = have.rv_20160623.astype(int)
    F["sv_price_last_pre"] = pr.loc[:, pre].ffill(axis=1).iloc[:, -1]
    F["sv_price_q3"] = pr.loc[:, q3].median(axis=1)
    return F

def features_abs(P, L, res, S):
    X = P.join(listing_features_abs(L, res), how="left").join(slee_features_abs(S, P.NumberofReviews))
    assert len(X) == len(P)
    X["age_days"] = Q3 - X.pop("created")
    X["sv_price_ratio"] = X.sv_price_q3 / X.price_med_q2
    host = X.groupby("HostID").R_q2.transform("sum")
    X["host_R_q2_other"] = (host - X.R_q2) / (X.host_n - 1).replace(0, np.nan)
    return X.drop(columns=[c for c in DROP if c != "NumberofReviews"])

def calendar_features(L, o):
    W = L[L.day < o]
    g = W.groupby("PropertyID")
    F = pd.DataFrame({"n_days": g.size(), "first_day_rel": g.day.min() - o})
    blk = W.assign(b=(o - 1 - W.day) // 30)
    blk = blk[blk.b < min(o // 30, 6)]
    m = blk.pivot_table(index="PropertyID", columns="b", values=["R", "A", "B"], aggfunc="sum", fill_value=0)
    m.columns = [f"{s}_b{k}" for s, k in m.columns]
    F = F.join(m)
    for s in "RAB":
        F[f"{s}_90"] = F[[f"{s}_b{k}" for k in range(3)]].sum(axis=1)
    F["occ_90"] = F.R_90 / (F.R_90 + F.A_90).replace(0, np.nan)
    F["occ_30"] = F.R_b0 / (F.R_b0 + F.A_b0).replace(0, np.nan)
    for w in (7, 14):
        F = F.join(W[W.day >= o - w].groupby("PropertyID")[["R", "A", "B"]].sum().add_suffix(f"_last{w}"))
    F["status_last"] = W[W.day == o - 1].set_index("PropertyID").Status.reindex(F.index).fillna("none")
    last90, last30 = W[W.day >= o - 90], W[W.day >= o - 30]
    F["price_med_90"] = last90.groupby("PropertyID").Price.median()
    F["price_min_90"] = last90.groupby("PropertyID").Price.min()
    F["price_max_90"] = last90.groupby("PropertyID").Price.max()
    F["price_med_A_90"] = last90[last90.A == 1].groupby("PropertyID").Price.median()
    F["price_med_R_90"] = last90[last90.R == 1].groupby("PropertyID").Price.median()
    F["price_ratio_30_90"] = last30.groupby("PropertyID").Price.median() / F.price_med_90
    if o >= 180:
        F["price_med_prev90"] = W[W.day < o - 90].groupby("PropertyID").Price.median()
    F["r_no_id_90"] = last90[(last90.R == 1) & last90.ReservationID.isna()].groupby("PropertyID").size()
    return F

def reservation_features(res, o, idx):
    r = res[res.start < o].assign(known=lambda d: np.minimum(d.end, o - 1) - d.start + 1,
                                  booked=lambda d: d.booked.where(d.booked < o))
    g, q = r.groupby("pid"), r[r.start >= o - 90].groupby("pid")
    F = pd.DataFrame(index=idx)
    F["res_n_90"] = q.size()
    F["res_nights_mean"] = g.known.mean()
    F["res_nights_mean_90"] = q.known.mean()
    F["res_lead_mean"] = g.lead.mean()
    F["res_lead_med_90"] = q.lead.median()
    F["res_since_last_booking"] = o - g.booked.max()
    for w in (30, 60):
        recent = r[r.booked >= o - w].groupby("pid")
        F[f"res_n_booked_last{w}"] = recent.size()
        F[f"res_nights_booked_last{w}"] = recent.known.sum()
    if o >= 180:
        F["res_n_prev"] = r[r.start < o - 90].groupby("pid").size()
    open_stay = r[r.end >= o - 1].groupby("pid")
    F["res_open_lead"] = open_stay.lead.min()
    F["res_open_nights"] = open_stay.known.max()
    cnt = [c for c in F.columns if c.startswith(("res_n_", "res_nights_booked"))]
    F[cnt] = F[cnt].fillna(0)
    return F

def last_obs(C, dd, mask):
    v = pd.DataFrame(np.where(mask, C, np.nan)).ffill(axis=1).iloc[:, -1].to_numpy()
    t = pd.DataFrame(np.where(mask & ~np.isnan(C), dd, np.nan)).ffill(axis=1).iloc[:, -1].to_numpy()
    return v, t

def cum_at(C, dd, t, max_span=75):
    have = ~np.isnan(C)
    vl, tl = last_obs(C, dd, dd <= t)
    vr = pd.DataFrame(np.where(dd >= t, C, np.nan)).bfill(axis=1).iloc[:, 0].to_numpy()
    tr = pd.DataFrame(np.where((dd >= t) & have, dd, np.nan)).bfill(axis=1).iloc[:, 0].to_numpy()
    span = tr - tl
    w = np.where(span > 0, (t - tl) / np.where(span > 0, span, 1), 0.0)
    return np.where(span <= max_span, vl + w * (vr - vl), np.nan)

def slee_features(S, o, cum_end):
    rv, pr = S.filter(like="rv_"), S.filter(like="pr_")
    dd = (pd.to_datetime(rv.columns.str[3:], format="%Y%m%d") - ORIGIN).days.to_numpy()
    C, Pr = rv.to_numpy(float), pr.to_numpy(float)
    F = pd.DataFrame(index=S.index)
    last, last_day = last_obs(C, dd, dd < o)
    F["sv_last_pre"], F["sv_pre_gap"] = last, o - last_day
    c = {t: cum_at(C, dd, o + t) for t in (-90, 0, 30, 60, 92, LEAD, 130)}
    if cum_end is not None:
        c[LEAD] = cum_end
    F["sv_new_nr"] = c[LEAD] - last
    F["sv_new_nr_rate"] = F.sv_new_nr / (F.sv_pre_gap + LEAD)
    F["sv_d_pre90"] = c[0] - c[-90]
    F["sv_d_0_30"], F["sv_d_30_60"], F["sv_d_60_92"] = c[30] - c[0], c[60] - c[30], c[92] - c[60]
    F["sv_d_0_92"], F["sv_d_92_lead"], F["sv_d_lead_130"] = c[92] - c[0], c[LEAD] - c[92], c[130] - c[LEAD]
    have = ~np.isnan(C)
    for name, (a, b) in (("pre", (-100, 0)), ("in", (0, 92)), ("post", (92, 200))):
        F[f"sv_n_{name}"] = have[:, (dd >= o + a) & (dd < o + b)].sum(1)
    F["sv_price_last_pre"] = last_obs(Pr, dd, dd < o)[0]
    F["sv_price_in"] = pd.DataFrame(Pr[:, (dd >= o) & (dd < o + 92)]).median(axis=1).to_numpy()
    return F

def property_features(P):
    P = P.set_index("PropertyID")
    P["PropertyType"] = P.PropertyType.str.replace("&amp;", "&", regex=False)
    P["created"] = (pd.to_datetime(P.CreatedDate) - ORIGIN).dt.days
    P["host_n"] = P.groupby("HostID").HostID.transform("size")
    P["title_len"] = P.ListingTitle.str.len()
    return P

def features(P, L, res, S, o, cum_end):
    X = P.join(calendar_features(L, o), how="left").join(reservation_features(res, o, P.index))
    X = X.join(slee_features(S, o, cum_end))
    assert len(X) == len(P)
    X["age_days"] = o - X.pop("created")
    X["sv_price_ratio"] = X.sv_price_in / X.price_med_90
    host = X.groupby("HostID").R_90.transform("sum")
    X["host_R_90_other"] = (host - X.R_90) / (X.host_n - 1).replace(0, np.nan)
    X["task"] = int(o == Q3)
    return X.drop(columns=DROP)

def load():
    d = find(MARKER).parent
    print(f"data: {d}")
    tr = pd.read_csv(d / MARKER)
    te = pd.read_csv(d / "PropertyID_test.csv")
    P = property_features(pd.read_csv(d / "property_info.csv"))
    L = load_listings(d)
    res = reservations(L)
    ext = find(EXT)
    assert ext.exists(), f"{EXT} not found: attach the booking-ext dataset"
    S = pd.read_csv(ext).set_index("PropertyID")
    assert S.index.is_unique and P.index.isin(S.index).all(), "stale ext_slee.csv: re-upload booking-ext.zip"
    S = S.reindex(P.index)
    X3 = features(P, L, res, S, Q3, P.NumberofReviews.to_numpy(float))
    X2 = features(P, L, res, S, Q2, None)
    assert set(X2.columns) <= set(X3.columns), set(X2.columns) - set(X3.columns)

    y2 = L[(L.day >= Q2) & (L.day < Q3)].groupby("PropertyID").R.sum().reindex(X2.index).fillna(0).to_numpy(float)
    Z = pd.concat([X3, X2.reindex(columns=X3.columns)], keys=["q3", "q2"])
    for c in CATS:
        Z[c] = Z[c].astype(str).astype("category")
    X3, X2 = Z.loc["q3"], Z.loc["q2"]
    XA = features_abs(P, L, res, S)
    for c in CATS:
        XA[c] = XA[c].astype(str).astype("category")
    print(f"slee: {ext}; sv_new_nr coverage Q3 {X3.sv_new_nr.notna().mean():.3f}, Q2 {X2.sv_new_nr.notna().mean():.3f}")
    print(f"view A {XA.shape[1]} features, view B {X3.shape[1]}: train {len(tr)}, test {len(te)}, "
          f"auxiliary Q2 rows {len(X2)} (y mean {y2.mean():.2f})")
    assert tr.PropertyID.isin(X3.index).all() and te.PropertyID.isin(X3.index).all()

    XC = XA.join(X3[[c for c in X3.columns if c not in XA.columns]])
    views = {"A": (XA.loc[tr.PropertyID], XA.loc[te.PropertyID], None),
             "B": (X3.loc[tr.PropertyID], X3.loc[te.PropertyID], X2),
             "C": (XC.loc[tr.PropertyID], XC.loc[te.PropertyID], None)}
    return views, tr[TARGET].to_numpy(float), y2, te.PropertyID.to_numpy()
