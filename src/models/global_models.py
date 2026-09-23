from pathlib import Path
import json, joblib, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score, log_loss, brier_score_loss
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def metrics(y_true, prob):
    pred=(prob>=0.5).astype(int)
    return {
        "roc_auc":float(roc_auc_score(y_true,prob)),
        "pr_auc":float(average_precision_score(y_true,prob)),
        "accuracy":float(accuracy_score(y_true,pred)),
        "precision":float(precision_score(y_true,pred,zero_division=0)),
        "recall":float(recall_score(y_true,pred,zero_division=0)),
        "f1":float(f1_score(y_true,pred,zero_division=0)),
        "log_loss":float(log_loss(y_true,prob)),
        "brier":float(brier_score_loss(y_true,prob)),
    }

def chronological_split(df, train_end="2023-12-31", val_end="2024-12-31"):
    date=pd.to_datetime(df["date"])
    return (df[date<=train_end].copy(),
            df[(date>train_end)&(date<=val_end)].copy(),
            df[date>val_end].copy())

def train_tree_models(train,val,test,features,output_dir="models"):
    Path(output_dir).mkdir(parents=True,exist_ok=True)
    Xtr,ytr=train[features],train["target"]; Xv,yv=val[features],val["target"]; Xt,yt=test[features],test["target"]
    models={
        "logistic":Pipeline([("scaler",StandardScaler()),("model",LogisticRegression(max_iter=2000,class_weight="balanced",C=0.5))]),
        "xgboost":XGBClassifier(n_estimators=500,max_depth=6,learning_rate=0.03,subsample=0.8,colsample_bytree=0.8,objective="binary:logistic",eval_metric="logloss",tree_method="hist",random_state=42),
        "lightgbm":LGBMClassifier(n_estimators=500,learning_rate=0.03,num_leaves=31,subsample=0.8,colsample_bytree=0.8,objective="binary",random_state=42,verbosity=-1),
    }
    report={}
    for name,model in models.items():
        print(f"Training {name}")
        model.fit(Xtr,ytr)
        vp=model.predict_proba(Xv)[:,1]; tp=model.predict_proba(Xt)[:,1]
        report[name]={"validation":metrics(yv,vp),"test":metrics(yt,tp)}
        if name=="xgboost": model.save_model(str(Path(output_dir)/"xgboost_global.json"))
        else: joblib.dump(model,Path(output_dir)/f"{name}_global.joblib")
        pd.DataFrame({"date":test["date"],"ticker":test["ticker"],"y_true":yt.values,"probability":tp}).to_parquet(Path(output_dir)/f"{name}_test_predictions.parquet",index=False)
    Path(output_dir,"tree_metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report
