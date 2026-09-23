from pathlib import Path
import pandas as pd
import yfinance as yf

def download_ticker(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    kwargs = dict(start=start, auto_adjust=True, progress=False)
    if end:
        kwargs["end"] = end
    df = yf.download(ticker, **kwargs)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.dropna()

def download_universe(tickers, start, end=None, output="data/processed/raw_market.parquet"):
    rows = []
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {ticker}")
        try:
            df = download_ticker(ticker, start, end)
            if df.empty:
                print("  no data")
                continue
            df = df.reset_index()
            df["ticker"] = ticker
            rows.append(df)
        except Exception as exc:
            print(f"  ERROR: {exc}")
    if not rows:
        raise RuntimeError("No market data downloaded.")
    result = pd.concat(rows, ignore_index=True)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    print(f"Saved {len(result):,} rows -> {output}")
    return result
