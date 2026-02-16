from __future__ import annotations

import pandas as pd
import requests
from io import StringIO


WIKI_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"


def nasdaq100_tickers() -> list[str]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    r = requests.get(WIKI_URL, headers=headers, timeout=30)
    r.raise_for_status()

    tables = pd.read_html(StringIO(r.text))

    for t in tables:
        cols = [str(c).strip().lower() for c in t.columns]
        if "ticker" in cols:
            tickers = t.iloc[:, cols.index("ticker")].astype(str).str.strip().tolist()
            tickers = [x.replace(".", "-") for x in tickers]
            tickers = [x for x in tickers if x and x != "nan"]
            unique = list(dict.fromkeys(tickers))
            return unique

    raise RuntimeError("Could not find Nasdaq-100 tickers table on the page")


if __name__ == "__main__":
    syms = nasdaq100_tickers()
    print("count:", len(syms))
    print(syms[:25])
