"""Plot original UCI SECOM published 10-fold CV benchmark, not a new experiment.
Source: McCann & Johnston (2008), doi:10.24432/C54305
https://archive.ics.uci.edu/dataset/179/secom
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
d=pd.read_csv(root/"uci_secom_published_baselines.csv")
assert len(d)==6 and d.feature_selection.is_unique
assert d.balanced_error_rate_percent.between(0,100).all()
order=d.sort_values("balanced_error_rate_percent")
fig,ax=plt.subplots(1,2,figsize=(11,4.4),dpi=190)
y=range(len(order))
ax[0].barh(y,order.balanced_error_rate_percent,xerr=order.ber_reported_plusminus_percent,capsize=3,color="#254878",alpha=.86)
ax[0].set_yticks(list(y),order.feature_selection)
ax[0].invert_yaxis()
ax[0].set(xlabel="Balanced error rate (%)",xlim=(0,48))
ax[0].text(.02,.96,"(a)",transform=ax[0].transAxes,va="top",fontweight="bold")
ax[0].grid(axis="x",alpha=.18)
ax[1].errorbar(order.true_positive_percent,list(y),xerr=order.tp_reported_plusminus_percent,fmt="o",capsize=3,color="#1b7772",label="Fail-class recall")
ax[1].errorbar(order.true_negative_percent,list(y),xerr=order.tn_reported_plusminus_percent,fmt="s",capsize=3,color="#4c566a",label="Pass-class recall")
ax[1].set_yticks(list(y),order.feature_selection)
ax[1].invert_yaxis()
ax[1].set(xlabel="Class recall (%)",xlim=(25,90))
ax[1].legend(frameon=False,loc="lower left")
ax[1].text(.02,.96,"(b)",transform=ax[1].transAxes,va="top",fontweight="bold")
ax[1].grid(axis="x",alpha=.18)
fig.tight_layout()
out=root/"output";out.mkdir(exist_ok=True)
fig.savefig(out/"uci_secom_published_feature_selection_benchmarks.png",bbox_inches="tight")
fig.savefig(out/"uci_secom_published_feature_selection_benchmarks.pdf",bbox_inches="tight")
print(order.to_string(index=False))
