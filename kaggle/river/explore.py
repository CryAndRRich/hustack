import numpy as np

from preprocess import TARGET


def report(df):
    tr = df[(df.part == "train") & df[TARGET].notna()]
    q = tr[TARGET]
    print("== target per basin")
    print(tr.groupby("basin")[TARGET].describe(percentiles=[0.5, 0.9, 0.99, 0.999]).round(4).to_string())
    print("== share of sum q^2 in the top 0.5 % of hours: " + ", ".join(
        f"{b} {np.sort(s.values ** 2)[::-1][:len(s) // 200].sum() / (s.values ** 2).sum():.3f}"
        for b, s in tr.groupby("basin")[TARGET]))
    print(f"== constant baselines on train: zero {np.sqrt((q ** 2).mean()):.5f}, mean {q.std():.5f}")

    y = df.date.dt.year
    yr = df.groupby([df.part, y]).agg(p_mm=("total_precipitation", "sum"), pet_mm=("potential_evaporation", "sum"),
                                       q_mean=(TARGET, "mean"), q_max=(TARGET, "max"))
    yr[["p_mm", "pet_mm"]] = yr[["p_mm", "pet_mm"]].div(df.groupby([df.part, y]).basin.nunique(), axis=0)
    print("== per year (precip and PET per basin)")
    print(yr.round(4).to_string())

    print("== max rolling precip (mm/h mean) per basin: train max vs 2018 max, and 2018 hours above the train max")
    t8 = (df.part == "test") & (y == 2018)
    for w in [1, 24, 72, 720]:
        r = df.groupby(["part", "basin"]).total_precipitation.transform(lambda s: s.rolling(w, min_periods=1).mean())
        m_tr = r[df.part == "train"].groupby(df.basin).max()
        m_te = r[t8].groupby(df.basin).max()
        above = (r[t8] > df.basin[t8].map(m_tr)).groupby(df.basin[t8]).sum()
        print(f"  {w:>4}h: " + ", ".join(f"{b} {m_tr[b]:.2f}/{m_te[b]:.2f} ({above[b]})" for b in m_te.index))

    print("== 2018 monthly precip per basin (mm)")
    t = df[t8]
    print(t.groupby([t.basin, t.date.dt.month]).total_precipitation.sum().unstack().round(0).to_string())

    print("== lag (h) of max corr between precip and q, per basin")
    for b, s in df[df.part == "train"].groupby("basin"):
        c = [s[TARGET].corr(s.total_precipitation.shift(L)) for L in range(0, 49)]
        o = s[TARGET].notna()
        print(f"  {b}: lag {int(np.argmax(c))}, corr {max(c):.3f}, "
              f"runoff ratio {s[TARGET][o].sum() / s.total_precipitation[o].sum():.3f}")
    print("== annual precip vs annual mean q (train), corr per basin: " + ", ".join(
        f"{b} {np.corrcoef(s.total_precipitation.values, s[TARGET].values)[0, 1]:.2f}"
        for b, s in tr.groupby(["basin", tr.date.dt.year])[["total_precipitation", TARGET]].sum()
        .groupby(level=0)))
