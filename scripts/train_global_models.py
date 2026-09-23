import json
from pathlib import Path
import pandas as pd
from src.features.technical import FEATURE_COLUMNS
from src.models.global_models import chronological_split, train_tree_models
from src.models.lstm import train_lstm
DATA="data/processed/global_features.parquet"
def main():
    df=pd.read_parquet(DATA).sort_values(["date","ticker"])
    train,val,test=chronological_split(df)
    print("Train:",train.shape); print("Val:",val.shape); print("Test:",test.shape)
    tree_report=train_tree_models(train,val,test,FEATURE_COLUMNS,output_dir="models")
    lstm_report=train_lstm(train,val,test,FEATURE_COLUMNS,output_path="models/lstm_global.pt",seq_len=30,epochs=8)
    report={"tree_models":tree_report,"lstm":lstm_report}
    Path("models/model_comparison.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
if __name__=="__main__": main()
