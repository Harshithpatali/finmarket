import json
from pathlib import Path
import pandas as pd
from xgboost import XGBClassifier
from src.features.technical import FEATURE_COLUMNS
from src.models.global_models import metrics
DATA="data/processed/global_features.parquet"
def main():
    df=pd.read_parquet(DATA).sort_values("date"); periods=[("2022-12-31","2023-12-31"),("2023-12-31","2024-12-31"),("2024-12-31","2025-12-31")]
    results=[]
    for train_end,test_end in periods:
        train=df[df.date<=train_end]; test=df[(df.date>train_end)&(df.date<=test_end)]
        if train.empty or test.empty: continue
        model=XGBClassifier(n_estimators=400,max_depth=6,learning_rate=0.03,subsample=0.8,colsample_bytree=0.8,eval_metric="logloss",tree_method="hist",random_state=42)
        model.fit(train[FEATURE_COLUMNS],train.target); prob=model.predict_proba(test[FEATURE_COLUMNS])[:,1]
        row={"train_end":train_end,"test_end":test_end}; row.update(metrics(test.target,prob)); results.append(row); print(row)
    Path("models/walk_forward_metrics.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
if __name__=="__main__": main()
