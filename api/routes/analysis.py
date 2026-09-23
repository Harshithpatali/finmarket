from pathlib import Path
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.data.market import download_ticker
from src.data.news import search_gnews
from src.news.features import aggregate_news
from src.features.technical import add_technical_features, FEATURE_COLUMNS
from src.rag.retrieve import FinancialRetriever
from src.llm.groq_client import analyze

router=APIRouter()

class AnalysisRequest(BaseModel):
    ticker:str
    company:str
    days:int=3

def _with_date_column(df):
    out=df.reset_index().copy()
    for candidate in ("Date","Datetime","index"):
        if candidate in out.columns:
            out=out.rename(columns={candidate:"date"}); break
    if "date" not in out.columns:
        raise ValueError(f"Could not find a date column. Columns received: {list(out.columns)}")
    out["date"]=pd.to_datetime(out["date"],errors="coerce")
    if getattr(out["date"].dt,"tz",None) is not None: out["date"]=out["date"].dt.tz_localize(None)
    return out.dropna(subset=["date"])

@router.post("/analyze")
def stock_analysis(req:AnalysisRequest):
    try:
        df=download_ticker(req.ticker,"2023-01-01")
        if df.empty: raise ValueError("No market data returned.")
        feat=add_technical_features(df).copy()
        market_df=download_ticker("^NSEI","2023-01-01")
        if market_df.empty: raise ValueError("No NIFTY 50 market data returned.")
        market_df["market_return_1d"]=market_df["Close"].pct_change()
        market_df["market_return_5d"]=market_df["Close"].pct_change(5)
        market_df["market_volatility_20"]=market_df["market_return_1d"].rolling(20).std()
        market_df=_with_date_column(market_df)[["date","market_return_1d","market_return_5d","market_volatility_20"]]
        feat=_with_date_column(feat).merge(market_df,on="date",how="left")
        feat["relative_return_5d"]=feat["return_5d"]-feat["market_return_5d"]
        feat=feat.replace([float("inf"),float("-inf")],pd.NA).dropna()
        if feat.empty: raise ValueError("Not enough recent data for model features.")
        latest=feat.iloc[-1]
        model_path=Path("models/xgboost_global.json")
        if not model_path.exists(): raise ValueError("XGBoost model not found. Train it first.")
        from xgboost import XGBClassifier
        model=XGBClassifier(); model.load_model(str(model_path))
        model_input=pd.DataFrame([latest[FEATURE_COLUMNS].values],columns=FEATURE_COLUMNS)
        prob=float(model.predict_proba(model_input)[:,1][0])
        articles=search_gnews(req.company,max_results=10,days=req.days); news=aggregate_news(articles)
        rag=None
        if Path("data/knowledge/faiss.index").exists():
            rag=FinancialRetriever().search(f"{req.company} stock investment valuation risk market",k=5)
        context={"ticker":req.ticker,"company":req.company,"latest_features":latest[FEATURE_COLUMNS].to_dict(),"xgboost_probability_positive_5d":prob,"news_summary":news,"news_articles":[{"title":a.get("title"),"publishedAt":a.get("publishedAt"),"url":a.get("url")} for a in articles],"retrieved_financial_knowledge":rag}
        return {"ticker":req.ticker,"probability_positive_5d":prob,"news":news,"analysis":analyze(context)}
    except Exception as exc:
        raise HTTPException(status_code=500,detail=str(exc))
