"""External-site verification of *observed* clinical predictions and adjudicated labels.
No image model is trained here. No patient-level information is exported.
"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,average_precision_score,RocCurveDisplay,PrecisionRecallDisplay
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if not a.input.is_file():p.error("Actual adjudicated site-level predictions absent")
    d=pd.read_csv(a.input)
    required=["study_id","y_true","y_score","site_id"]
    assert all(x in d for x in required) and d.study_id.is_unique and len(d)>=100
    assert d.y_true.isin([0,1]).all() and d.y_score.between(0,1).all()
    fig,ax=plt.subplots(1,2,figsize=(10,4),dpi=180);rows=[]
    for site,g in d.groupby("site_id"):
        if len(g)<30 or g.y_true.nunique()!=2: continue
        rows.append({"site":str(site),"n_studies":len(g),"ROC_AUC":roc_auc_score(g.y_true,g.y_score),"AP":average_precision_score(g.y_true,g.y_score)})
        RocCurveDisplay.from_predictions(g.y_true,g.y_score,ax=ax[0],name=str(site))
        PrecisionRecallDisplay.from_predictions(g.y_true,g.y_score,ax=ax[1],name=str(site))
    assert rows,"No site has adequate positive and negative cases"
    for axis in ax: axis.grid(alpha=.2)
    a.output.mkdir(parents=True,exist_ok=True);fig.tight_layout();fig.savefig(a.output/"clinical_external_site_roc_pr.png",bbox_inches="tight")
    pd.DataFrame(rows).to_csv(a.output/"clinical_external_site_metrics.csv",index=False)
if __name__=="__main__":main()
