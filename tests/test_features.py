import numpy as np
import pandas as pd
from src.features.technical import add_technical_features

def test_features_are_created():
    n=250
    close=pd.Series(np.linspace(100,150,n))
    df=pd.DataFrame({"Open":close*0.99,"High":close*1.01,"Low":close*0.98,"Close":close,"Volume":np.full(n,100000.0)})
    out=add_technical_features(df)
    assert "rsi_14" in out.columns
    assert "macd" in out.columns
    assert len(out)==n
