"""Analyze measured Delaney/ESOL solubility and source-supplied predictions.
Not a new fitted model or independent held-out benchmark.
Source: DeepChem delaney-processed.csv; Delaney (2004), DOI 10.1021/ci034243x.
Download original CSV from https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

m="measured log solubility in mols per litre"
p="ESOL predicted log solubility in mols per litre"
w="Molecular Weight"

def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("--input",type=Path,required=True)
    cli.add_argument("--output",type=Path,required=True)
    a=cli.parse_args()
    if not a.input.is_file():cli.error("Original measured ESOL CSV required")
    df=pd.read_csv(a.input)
    if len(df)!=1128 or not all(c in df for c in [m,p,w,"smiles"]):
        raise ValueError("Expected complete 1128-record source file")
    y=df[m].to_numpy(float); z=df[p].to_numpy(float)
    assert np.isfinite(y).all() and np.isfinite(z).all()
    metrics={"n":len(df),"MAE":np.mean(np.abs(y-z)),
             "RMSE":np.sqrt(np.mean((y-z)**2)),
             "R2":1-np.sum((y-z)**2)/np.sum((y-y.mean())**2)}
    bins=[0,150,250,350,500,np.inf]
    labels=["0-150","150-250","250-350","350-500","500+"]
    df["mw_bin"]=pd.cut(df[w],bins,labels=labels,right=False)
    rows=[]
    for label,g in df.groupby("mw_bin",observed=True):
        yt=g[m].to_numpy(float); zp=g[p].to_numpy(float)
        rows.append(dict(molecular_weight_group_da=str(label),n=len(g),
            mean_measured_log_molar=yt.mean(),
            mean_original_esol_predicted_log_molar=zp.mean(),
            mae_log_molar=np.mean(np.abs(yt-zp)),
            rmse_log_molar=np.sqrt(np.mean((yt-zp)**2))))
    b=pd.DataFrame(rows)
    a.output.mkdir(parents=True,exist_ok=True)
    b.to_csv(a.output/"esol_measured_group_results.csv",index=False)
    pd.DataFrame([metrics]).to_csv(a.output/"esol_summary_metrics.csv",index=False)
    fig,axs=plt.subplots(1,2,figsize=(10,4),dpi=180)
    x=np.arange(len(b))
    axs[0].plot(x,b.mean_measured_log_molar,"o-",color="#164b77",label="Measured")
    axs[0].plot(x,b.mean_original_esol_predicted_log_molar,"s--",color="#0b8077",label="Source ESOL predictions")
    axs[0].set_ylabel("Mean log10 solubility (mol/L)")
    axs[0].legend(frameon=False)
    axs[1].bar(x,b.mae_log_molar,color="#244d78")
    axs[1].set_ylabel("Absolute error, log10(mol/L)")
    for ax in axs:
        ax.set_xticks(x,labels)
        ax.set_xlabel("Molecular weight bin (Da)")
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(a.output/"esol_source_measured_comparison.png",bbox_inches="tight")
    print(metrics)

if __name__=="__main__":main()
