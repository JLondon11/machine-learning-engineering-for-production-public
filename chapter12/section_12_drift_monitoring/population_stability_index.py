"""Population Stability Index with fixed reference quantile bins.

PSI is a distribution shift diagnostic, not a causal concept-drift test.
"""
import numpy as np

def population_stability_index(reference, current, bins=10):
    ref=np.asarray(reference,dtype=float)
    cur=np.asarray(current,dtype=float)
    if (ref.ndim!=1 or cur.ndim!=1 or ref.size==0 or cur.size==0
            or not np.isfinite(ref).all() or not np.isfinite(cur).all()
            or bins<2 or bins>100):
        raise ValueError("Need finite, nonempty one-dimensional arrays and 2-100 bins")
    edges=np.quantile(ref,np.linspace(0,1,bins+1))
    edges[0],edges[-1]=-np.inf,np.inf
    edges=np.unique(edges)
    if len(edges)<3:raise ValueError("Reference distribution has insufficient variation")
    rc,_=np.histogram(ref,bins=edges)
    cc,_=np.histogram(cur,bins=edges)
    eps=1e-8
    p=np.maximum(rc/rc.sum(),eps)
    q=np.maximum(cc/cc.sum(),eps)
    return float(np.sum((q-p)*np.log(q/p)))
