from pathlib import Path
import pandas as pd
from src.features.technical import add_technical_features, FEATURE_COLUMNS
RAW="data/processed/raw_market.parquet"; OUT="data/processed/global_features.parquet"
def main():
    df=pd.read_parquet(RAW)
    date_source=next((c for c in ("Date","Datetime","date") if c in df.columns),None)
    if date_source is None: raise ValueError(f"No date column found. Columns: {list(df.columns)}")
    df["date"]=pd.to_datetime(df[date_source],errors="coerce")
    df=df.drop(columns=[c for c in ("Date","Datetime") if c in df.columns],errors="ignore").dropna(subset=["date"])
    market=df[df.ticker=="^NSEI"][["date","Close"]].rename(columns={"Close":"market_close"})
    market["market_return_1d"]=market.market_close.pct_change()
    market["market_return_5d"]=market.market_close.pct_change(5)
    market["market_volatility_20"]=market.market_return_1d.rolling(20).std()
    parts=[]
    for ticker,g in df[df.ticker!="^NSEI"].groupby("ticker"):
        g=add_technical_features(g.sort_values("date").copy()).merge(market,on="date",how="left")
        g["relative_return_5d"]=g.return_5d-g.market_return_5d
        g["future_return_5d"]=g.Close.shift(-5)/g.Close-1
        g["target"]=(g.future_return_5d>0).astype(int)
        parts.append(g)
    out=pd.concat(parts,ignore_index=True)
    out=out[FEATURE_COLUMNS+["target","future_return_5d","date","ticker"]].replace([float("inf"),float("-inf")],pd.NA).dropna()
    Path(OUT).parent.mkdir(parents=True,exist_ok=True); out.to_parquet(OUT,index=False)
    print(f"Saved {len(out):,} observations -> {OUT}; stocks={out.ticker.nunique()}")
if __name__=="__main__": main()
