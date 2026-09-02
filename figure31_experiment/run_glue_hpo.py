import os,time,random,math
import numpy as np,pandas as pd,torch
from datasets import load_dataset
from transformers import AutoTokenizer,AutoModelForSequenceClassification,get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score,f1_score,matthews_corrcoef

MODEL="prajjwal1/bert-tiny"; TASKS=["cola","mrpc","rte","qnli"]
KEYS={"cola":("sentence",None),"mrpc":("sentence1","sentence2"),"rte":("sentence1","sentence2"),"qnli":("question","sentence")}
BASE={"lr":2e-5,"warmup":.10,"batch":16,"wd":.01}; SEEDS=[13,37,71]
def seed(s):
 random.seed(s); np.random.seed(s); torch.manual_seed(s)
def configs(task):
 z=[]
 for s in SEEDS:z.append((task,BASE["lr"],BASE["warmup"],BASE["batch"],BASE["wd"],s,"seed"))
 for v in [1e-5,5e-5]:z.append((task,v,BASE["warmup"],BASE["batch"],BASE["wd"],37,"lr"))
 for v in [0.,.20]:z.append((task,BASE["lr"],v,BASE["batch"],BASE["wd"],37,"warmup"))
 for v in [8,32]:z.append((task,BASE["lr"],BASE["warmup"],v,BASE["wd"],37,"batch"))
 for v in [0.,.10]:z.append((task,BASE["lr"],BASE["warmup"],BASE["batch"],v,37,"wd"))
 return z
def metric(task,y,p):
 if task=="cola":return matthews_corrcoef(y,p),"matthews_correlation"
 if task=="mrpc":return f1_score(y,p),"f1"
 return accuracy_score(y,p),"accuracy"
def one(c):
 task,lr,warm,batch,wd,s,factor=c; seed(s); dev=torch.device("cpu")
 ds=load_dataset("nyu-mll/glue",task); tok=AutoTokenizer.from_pretrained("bert-base-uncased")
 k1,k2=KEYS[task]
 tr=ds["train"].shuffle(seed=2026).select(range(min(2000,len(ds["train"]))))
 va=ds["validation"]
 def coll(ex):
  a=[e[k1] for e in ex]; b=None if k2 is None else [e[k2] for e in ex]
  q=tok(a,b,padding=True,truncation=True,max_length=128,return_tensors="pt")
  q["labels"]=torch.tensor([e["label"] for e in ex]); return q
 g=torch.Generator().manual_seed(s)
 dl=DataLoader(tr,batch_size=batch,shuffle=True,collate_fn=coll,generator=g)
 vl=DataLoader(va,batch_size=batch,shuffle=False,collate_fn=coll)
 m=AutoModelForSequenceClassification.from_pretrained(MODEL,num_labels=2).to(dev)
 opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=wd); steps=len(dl)
 sch=get_linear_schedule_with_warmup(opt,int(round(warm*steps)),steps)
 t=time.perf_counter(); m.train()
 for q in dl:
  q={k:v.to(dev) for k,v in q.items()}; opt.zero_grad(); loss=m(**q).loss; loss.backward()
  torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); sch.step()
 sec=time.perf_counter()-t; m.eval(); yy=[]; pp=[]
 with torch.no_grad():
  for q in vl:
   y=q.pop("labels"); q={k:v.to(dev) for k,v in q.items()}
   p=m(**q).logits.argmax(-1).cpu(); yy+=y.tolist(); pp+=p.tolist()
 sc,mn=metric(task,yy,pp)
 return dict(task=task,lr=lr,warmup_ratio=warm,batch_size=batch,weight_decay=wd,seed=s,factor=factor,
             primary_metric=sc,metric_name=mn,train_seconds=sec,optimizer_steps=steps,
             train_examples=len(tr),validation_examples=len(va),model=MODEL)
os.makedirs("figure31_results",exist_ok=True); rows=[]
allc=sum([configs(t) for t in TASKS],[])
for i,c in enumerate(allc,1):
 print(i,len(allc),c,flush=True); rows.append(one(c))
 pd.DataFrame(rows).to_csv("figure31_results/run_results.csv",index=False)
