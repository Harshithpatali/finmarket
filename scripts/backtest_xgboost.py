from pathlib import Path
import pandas as pd
from xgboost import XGBClassifier
from src.features.technical import FEATURE_COLUMNS
DATA="data/processed/global_features.parquet"
def max_drawdown(equity):
    peak=equity.cummax(); return float((equity/peak-1).min())
def main():
    df=pd.read_parquet(DATA).sort_values(["date","ticker"]); train=df[df.date<="2024-12-31"]; test=df[df.date>"2024-12-31"].copy()
    model=XGBClassifier(n_estimators=500,max_depth=6,learning_rate=0.03,subsample=0.8,colsample_bytree=0.8,eval_metric="logloss",tree_method="hist",random_state=42)
    model.fit(train[FEATURE_COLUMNS],train.target); test["prob"]=model.predict_proba(test[FEATURE_COLUMNS])[:,1]
    test["position"]=(test.prob>=0.60).astype(float); test["strategy_return"]=test.position*test.future_return_5d
    daily=test.groupby("date").strategy_return.mean().sort_index(); equity=(1+daily).cumprod()
    metrics={"threshold":0.60,"observations":int(len(test)),"dates":int(daily.shape[0]),"cumulative_return":float(equity.iloc[-1]-1),"max_drawdown":max_drawdown(equity),"average_daily_strategy_return":float(daily.mean()),"daily_volatility":float(daily.std())}
    Path("models/backtest_metrics.json").write_text(__import__("json").dumps(metrics,indent=2),encoding="utf-8"); equity.to_frame("equity").to_parquet("models/backtest_equity.parquet"); print(metrics)
if __name__=="__main__": main()
