from pathlib import Path
import pandas as pd
import shap
from xgboost import XGBClassifier
from src.features.technical import FEATURE_COLUMNS
def main():
    df=pd.read_parquet("data/processed/global_features.parquet"); model=XGBClassifier(); model.load_model("models/xgboost_global.json")
    sample=df[FEATURE_COLUMNS].tail(min(5000,len(df))); values=shap.TreeExplainer(model).shap_values(sample)
    importance=pd.DataFrame({"feature":FEATURE_COLUMNS,"mean_abs_shap":abs(values).mean(axis=0)}).sort_values("mean_abs_shap",ascending=False)
    Path("models").mkdir(exist_ok=True); importance.to_csv("models/xgboost_shap_importance.csv",index=False); print(importance.head(20).to_string(index=False))
if __name__=="__main__": main()
