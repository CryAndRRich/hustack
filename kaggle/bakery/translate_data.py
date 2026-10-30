import re
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
PRODUCTS = {"Хлеб на закваске": "sourdough bread", "Круассан": "croissant", "Булочка с корицей": "cinnamon bun"}
TEMPLATES = [
    (r"^Регистрация: около ([\d.]+) тыс\. чел\.$", "registration", 1000, "Registration: about {} thousand people."),
    (r"^План: ([\d ]+) чел\.; согласовано с площадкой$", "plan", 1, "Plan: {} people; agreed with the venue."),
    (r"^Ожидаемая аудитория: (\d+) человек\. Предварительно\.$", "expected", 1, "Expected audience: {} people. Preliminary."),
]
CYRILLIC = re.compile("[Ѐ-ӿ]")


def parse(text):
    for pat, fmt, scale, en in TEMPLATES:
        m = re.match(pat, text)
        if m:
            raw = m.group(1).replace(" ", "")
            return fmt, float(raw) * scale, en.format(raw)
    raise ValueError(f"unknown announcement: {text!r}")


def main():
    pr = pd.read_csv(DATA / "products.csv")
    pr["name"] = pr.name.map(PRODUCTS)
    ev = pd.read_csv(DATA / "events.csv")
    ev[["announcement_format", "attendance", "announcement"]] = pd.DataFrame(
        [parse(a) for a in ev.announcement], index=ev.index)
    for name, df in (("products", pr), ("events", ev)):
        assert df.notna().all().all() and not df.astype(str).apply(lambda c: c.str.contains(CYRILLIC)).any().any()
        df.to_csv(DATA / f"{name}_en.csv", index=False)
        print(f"wrote {name}_en.csv: {df.shape}")
    print(ev.groupby("announcement_format").attendance.describe().round(0))


if __name__ == "__main__":
    main()
