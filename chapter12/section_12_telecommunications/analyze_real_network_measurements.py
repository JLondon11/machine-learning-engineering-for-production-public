"""Time-split anomaly benchmark on measured telecom telemetry (not unrelated datasets)."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, RocCurveDisplay
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if not a.input.is_file():p.error("Measured network dataset absent; no synthetic fallback")
    d=pd.read_csv(a.input,parse_dates=["timestamp"]).sort_values("timestamp")
    cols=["throughput_mbps","latency_ms","packet_loss_pct"]
    assert all(c in d for c in cols+["anomaly_label"]) and len(d)>=200
    n=int(len(d)*.75);tr,te=d.iloc[:n],d.iloc[n:]
    assert tr.anomaly_label.nunique()==2 and te.anomaly_label.nunique()==2
    models={"prior":DummyClassifier(strategy="prior"),"telemetry classifier":make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),LogisticRegression(max_iter=1200,class_weight="balanced"))}
    fig,ax=plt.subplots(figsize=(6,4),dpi=180);rows=[]
    for name,m in models.items():
        m.fit(tr[cols],tr.anomaly_label);s=m.predict_proba(te[cols])[:,1]
        rows.append({"model":name,"n_test":len(te),"ROC_AUC":roc_auc_score(te.anomaly_label,s),"AP":average_precision_score(te.anomaly_label,s)})
        RocCurveDisplay.from_predictions(te.anomaly_label,s,ax=ax,name=name)
    a.output.mkdir(parents=True,exist_ok=True)
    ax.grid(alpha=.25);fig.tight_layout();fig.savefig(a.output/"measured_network_anomaly_roc.png",bbox_inches="tight")
    pd.DataFrame(rows).to_csv(a.output/"measured_network_metrics.csv",index=False)
if __name__=="__main__":main()
