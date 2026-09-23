from pathlib import Path
import json, numpy as np, pandas as pd, torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score

class LSTMClassifier(nn.Module):
    def __init__(self,n_features,hidden=64,layers=2,dropout=0.2):
        super().__init__()
        self.lstm=nn.LSTM(input_size=n_features,hidden_size=hidden,num_layers=layers,batch_first=True,dropout=dropout if layers>1 else 0)
        self.head=nn.Sequential(nn.Linear(hidden,32),nn.ReLU(),nn.Dropout(dropout),nn.Linear(32,1))
    def forward(self,x):
        out,_=self.lstm(x); return self.head(out[:,-1,:]).squeeze(-1)

def make_sequences(df,features,seq_len=30):
    X=[]; y=[]; meta=[]
    for ticker,g in df.sort_values("date").groupby("ticker"):
        g=g.sort_values("date").reset_index(drop=True)
        values=g[features].values.astype("float32"); target=g["target"].values.astype("float32")
        for i in range(seq_len-1,len(g)):
            X.append(values[i-seq_len+1:i+1]); y.append(target[i]); meta.append((g.loc[i,"date"],ticker))
    return np.array(X),np.array(y),meta

def train_lstm(train,val,test,features,output_path="models/lstm_global.pt",seq_len=30,epochs=8,batch_size=256):
    scaler=StandardScaler(); scaler.fit(train[features])
    train_s,val_s,test_s=[d.copy() for d in (train,val,test)]
    for d in (train_s,val_s,test_s): d.loc[:,features]=scaler.transform(d[features])
    Xtr,ytr,_=make_sequences(train_s,features,seq_len); Xv,yv,_=make_sequences(val_s,features,seq_len); Xt,yt,meta=make_sequences(test_s,features,seq_len)
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=LSTMClassifier(len(features)).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3); loss_fn=nn.BCEWithLogitsLoss()
    loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(Xtr),torch.tensor(ytr)),batch_size=batch_size,shuffle=True)
    for epoch in range(epochs):
        model.train(); losses=[]
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device); optimizer.zero_grad(); loss=loss_fn(model(xb),yb); loss.backward(); optimizer.step(); losses.append(loss.item())
        model.eval()
        with torch.no_grad(): val_prob=torch.sigmoid(model(torch.tensor(Xv).to(device))).cpu().numpy()
        print(f"LSTM epoch {epoch+1}/{epochs} loss={np.mean(losses):.4f} val_auc={roc_auc_score(yv,val_prob):.4f}")
    model.eval()
    with torch.no_grad(): test_prob=torch.sigmoid(model(torch.tensor(Xt).to(device))).cpu().numpy()
    pred=(test_prob>=0.5).astype(int)
    result={"roc_auc":float(roc_auc_score(yt,test_prob)),"pr_auc":float(average_precision_score(yt,test_prob)),"accuracy":float(accuracy_score(yt,pred)),"f1":float(f1_score(yt,pred,zero_division=0))}
    Path(output_path).parent.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":model.state_dict(),"features":features,"seq_len":seq_len,"scaler_mean":scaler.mean_,"scaler_scale":scaler.scale_},output_path)
    Path(str(output_path)+".metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    pd.DataFrame({"date":[m[0] for m in meta],"ticker":[m[1] for m in meta],"y_true":yt,"probability":test_prob}).to_parquet(Path(output_path).with_suffix(".predictions.parquet"),index=False)
    return result
