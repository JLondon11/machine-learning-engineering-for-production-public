"""Scaffold holdout on experimentally measured IC50; never simulated molecular outcomes."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error,r2_score
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if not a.input.is_file():p.error("Experimental bioactivity CSV absent")
    d=pd.read_csv(a.input).dropna(subset=["smiles","ic50_nM"])
    d=d[d.ic50_nM>0].copy()
    d["mol"]=d.smiles.map(Chem.MolFromSmiles);d=d[d.mol.notna()].copy()
    assert len(d)>=100
    d["scaffold"]=d.mol.map(lambda m:Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m)))
    groups=sorted(d.scaffold.unique());assert len(groups)>=10
    np.random.default_rng(42).shuffle(groups);held=set(groups[:max(1,int(.25*len(groups)))])
    train=~d.scaffold.isin(held);test=~train
    assert train.sum()>=40 and test.sum()>=20
    fg=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=1024)
    X=np.stack([fg.GetFingerprintAsNumPy(m) for m in d.mol])
    y=9-np.log10(d.ic50_nM.to_numpy())
    models={"median":DummyRegressor(strategy="median"),"Morgan RF":RandomForestRegressor(n_estimators=250,min_samples_leaf=2,random_state=42)}
    fig,ax=plt.subplots(figsize=(5,5),dpi=180);rows=[]
    for name,m in models.items():
        m.fit(X[train],y[train]);pred=m.predict(X[test])
        rows.append({"model":name,"n_test":int(test.sum()),"MAE_pIC50":mean_absolute_error(y[test],pred),"R2_pIC50":r2_score(y[test],pred)})
        ax.scatter(y[test],pred,label=name,s=15,alpha=.35)
    ax.set(xlabel="Measured pIC50",ylabel="Predicted pIC50");ax.legend(frameon=False);ax.grid(alpha=.2)
    a.output.mkdir(parents=True,exist_ok=True);fig.tight_layout();fig.savefig(a.output/"measured_ic50_scaffold_holdout.png",bbox_inches="tight")
    pd.DataFrame(rows).to_csv(a.output/"measured_bioactivity_metrics.csv",index=False)
if __name__=="__main__":main()
