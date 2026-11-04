"""Convert the UCI SECOM original paired text files into a labeled CSV.

Dataset: McCann & Johnston (2008), UCI SECOM, DOI 10.24432/C54305.
Use original secom.data and secom_labels.data, downloaded from the cited repository.
A -1 label is pass; +1 is fail. Do not use the timestamp as a predictor.
"""
from pathlib import Path
import argparse
import pandas as pd

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--measurements",required=True,type=Path)
    p.add_argument("--labels",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    for path in (a.measurements,a.labels):
        if not path.is_file(): p.error(f"Real UCI file required: {path}")
    x=pd.read_csv(a.measurements,sep=r"\s+",header=None,na_values=["NaN"])
    meta=pd.read_csv(a.labels,sep=r"\s+",header=None)
    if x.shape != (1567,590): raise ValueError(f"Expected 1567x590 SECOM measurements; got {x.shape}")
    if meta.shape[0] != 1567 or not set(meta.iloc[:,0].unique()) <= {-1,1}:
        raise ValueError("Invalid original UCI labels")
    x.columns=[f"sensor_{i:03d}" for i in range(x.shape[1])]
    x.insert(0,"label",meta.iloc[:,0].to_numpy())
    a.output.parent.mkdir(parents=True,exist_ok=True)
    x.to_csv(a.output,index=False)
    print("Prepared 1567 measured rows; 1=fail, -1=pass; timestamp not used as predictor.")

if __name__ == "__main__": main()
