import sys,numpy as np,pandas as pd,matplotlib.pyplot as plt
d=pd.read_csv(sys.argv[1]); out=sys.argv[3] if len(sys.argv)>3 and sys.argv[2]=="--out" else "Figure31_GLUE.png"
tasks=["cola","mrpc","rte","qnli"]; labels={"cola":"CoLA","mrpc":"MRPC","rte":"RTE","qnli":"QNLI"}
params=[("lr","Learning rate"),("warmup_ratio","Warm-up"),("batch_size","Batch size"),("weight_decay","Weight decay")]
base={"lr":2e-5,"warmup_ratio":.10,"batch_size":16,"weight_decay":.01}
fig=plt.figure(figsize=(11.5,7.4),constrained_layout=True); gs=fig.add_gridspec(2,1,height_ratios=[1.35,1])
ax=fig.add_subplot(gs[0]); x=np.arange(4); offs=np.linspace(-.24,.24,4); marks=["o","s","^","D"]
for off,t,mk in zip(offs,tasks,marks):
 q=d[(d.task==t)&(d.seed==37)]
 b=q[(np.isclose(q.lr,base["lr"]))&(np.isclose(q.warmup_ratio,base["warmup_ratio"]))&(q.batch_size==16)&(np.isclose(q.weight_decay,.01))].primary_metric.iloc[0]
 vals=[]
 for p,_ in params:
  qq=q.copy()
  for k,v in base.items():
   if k!=p: qq=qq[np.isclose(qq[k],v) if k!="batch_size" else qq[k]==v]
  vals.append(100*np.max(np.abs(qq.primary_metric-b)))
 ax.scatter(x+off,vals,marker=mk,s=55,facecolors="white",label=labels[t])
ax.set_xticks(x,[z[1] for z in params]); ax.set_ylabel("Max |Δ validation metric| (pp)")
ax.grid(axis="y",alpha=.15); ax.spines[["top","right"]].set_visible(False); ax.legend(frameon=False,ncol=4)
ax2=fig.add_subplot(gs[1]); means=[]; sds=[]
for t in tasks:
 q=d[(d.task==t)&(np.isclose(d.lr,2e-5))&(np.isclose(d.warmup_ratio,.1))&(d.batch_size==16)&(np.isclose(d.weight_decay,.01))]
 means.append(100*q.primary_metric.mean()); sds.append(100*q.primary_metric.std(ddof=1))
ax2.errorbar(np.arange(4),means,yerr=sds,fmt="o",mfc="white",capsize=4,linestyle="none")
ax2.set_xticks(np.arange(4),[labels[t] for t in tasks]); ax2.set_ylabel("Validation metric (%)")
ax2.set_xlabel("Baseline repeated seeds; error bars = ±1 SD")
ax2.grid(axis="y",alpha=.15); ax2.spines[["top","right"]].set_visible(False)
fig.savefig(out,dpi=450,bbox_inches="tight",facecolor="white"); fig.savefig(out.rsplit(".",1)[0]+".pdf",bbox_inches="tight",facecolor="white")
