"""Observed NYC TLC trip-duration (ETA proxy), not a dispatch-efficiency experiment."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,median_absolute_error
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if not a.input.is_file():p.error("Official NYC TLC parquet missing; no synthetic fallback")
    d=pd.read_parquet(a.input)
    cols=["tpep_pickup_datetime","tpep_dropoff_datetime","trip_distance","PULocationID","DOLocationID"]
    assert all(c in d for c in cols)
    start=pd.to_datetime(d.tpep_pickup_datetime);end=pd.to_datetime(d.tpep_dropoff_datetime)
    d["duration_min"]=(end-start).dt.total_seconds()/60
    d=d[d.duration_min.between(1,180)&d.trip_distance.between(.1,100)].sort_values("tpep_pickup_datetime")
    assert len(d)>=1000
    d["hour"]=pd.to_datetime(d.tpep_pickup_datetime).dt.hour
    d["weekday"]=pd.to_datetime(d.tpep_pickup_datetime).dt.dayofweek
    features=["trip_distance","PULocationID","DOLocationID","hour","weekday"]
    n=int(.8*len(d));tr,te=d.iloc[:n],d.iloc[n:]
    models={"median baseline":DummyRegressor(strategy="median"),"gradient-boosted ETA":HistGradientBoostingRegressor(max_iter=110,random_state=42)}
    fig,ax=plt.subplots(figsize=(6,4),dpi=180);rows=[]
    for name,m in models.items():
        m.fit(tr[features],tr.duration_min)
        pred=m.predict(te[features]);err=np.abs(te.duration_min.to_numpy()-pred)
        rows.append({"model":name,"n_test":len(te),"MAE_minutes":mean_absolute_error(te.duration_min,pred),"median_AE_minutes":median_absolute_error(te.duration_min,pred)})
        qs=np.linspace(.05,.95,19);ax.plot(np.quantile(err,qs),qs,label=name)
    ax.set(xlabel="Absolute trip-duration prediction error (min)",ylabel="Empirical CDF")
    ax.grid(alpha=.2);ax.legend(frameon=False)
    a.output.mkdir(parents=True,exist_ok=True);fig.tight_layout();fig.savefig(a.output/"observed_tlc_eta_error.png",bbox_inches="tight")
    pd.DataFrame(rows).to_csv(a.output/"observed_tlc_metrics.csv",index=False)
if __name__=="__main__":main()
