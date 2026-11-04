"""Measured semiconductor quality benchmark; never synthetic. Input UCI SECOM CSV with label +/-1."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, RocCurveDisplay
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    if not a.input.is_file(): p.error("Measured SECOM CSV absent; no synthetic fallback")
    d=pd.read_csv(a.input)
    assert "label" in d and len(d)>=100
    y=d.label.replace({-1:0,1:1}).astype(int)
    assert y.nunique()==2
    cols=[c for c in d.select_dtypes(include="number") if c!="label" and d[c].notna().sum()>10]
    assert len(cols)>20
    xtr,xte,ytr,yte=train_test_split(d[cols],y,test_size=.25,random_state=42,stratify=y)
    models={"majority":DummyClassifier(strategy="prior"),
            "logistic":make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),LogisticRegression(max_iter=1500,class_weight="balanced"))}
    fig,ax=plt.subplots(figsize=(6,4),dpi=180);rows=[]
    for name,m in models.items():
        m.fit(xtr,ytr);s=m.predict_proba(xte)[:,1]
        rows.append({"model":name,"n_test":len(yte),"ROC_AUC":roc_auc_score(yte,s),"average_precision":average_precision_score(yte,s)})
        RocCurveDisplay.from_predictions(yte,s,ax=ax,name=name)
    a.output.mkdir(parents=True,exist_ok=True)
    ax.grid(alpha=.25);fig.tight_layout();fig.savefig(a.output/"measured_secom_roc.png",bbox_inches="tight")
    pd.DataFrame(rows).to_csv(a.output/"measured_secom_metrics.csv",index=False)
    print(pd.DataFrame(rows).to_string(index=False))
if __name__=="__main__": main()
