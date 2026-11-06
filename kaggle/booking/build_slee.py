from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
SRC = DATA / "external" / "tomslee"

def main():
    ids = pd.read_csv(DATA / "property_info.csv", usecols=["PropertyID"]).PropertyID
    out = pd.DataFrame(index=pd.Index(ids, name="PropertyID"))
    for f in sorted(SRC.glob("tomslee_airbnb_new_york_*.csv")):
        tag = f.stem.split("_")[-1].replace("-", "")
        d = pd.read_csv(f, usecols=["room_id", "reviews", "price"])
        assert d.room_id.is_unique, f
        d = d.set_index("room_id").reindex(ids)
        out[f"rv_{tag}"], out[f"pr_{tag}"] = d.reviews.values, d.price.values
    out.to_csv(DATA / "ext_slee.csv")
    print(f"ext_slee.csv: {out.shape}, any survey {out.filter(like='rv_').notna().any(axis=1).mean():.3f}")

if __name__ == "__main__":
    main()
