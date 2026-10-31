import json,math,collections,numpy as np
from scipy.stats import spearmanr
M="qwen3vl8b"; rng=np.random.default_rng(0)
truth=json.load(open("rank_truth.json"))
rows={json.loads(l)["id"]:json.loads(l) for l in open("rank_pairs.jsonl")}
R=collections.defaultdict(dict)
for l in open(f"raw_rankpairs_{M}.jsonl"):
    d=json.loads(l); R[d["id"]][d["order"]]=d["lp"]
def sm(lp):
    m=max(lp.values()); e={k:math.exp(v-m) for k,v in lp.items()}; z=sum(e.values()); return {k:v/z for k,v in e.items()}
img_rows=[json.loads(l) for l in open("rank_images.jsonl")]; E={}
for l in open(f"raw_rankimages_{M}.jsonl"):
    d=json.loads(l)
    if d["order"]=="A": p=sm(d["lp"]); E[(img_rows[d["id"]]["category"],img_rows[d["id"]]["left_id"])]=sum(int(k)*v for k,v in p.items())
def bt(n,pairs,w,iters=200,prior=None):
    # pairs: (i,j); w: prob i beats j (soft). MM with small regularization (virtual 0.5 game vs mean)
    s=np.zeros(n) if prior is None else prior.copy()
    for _ in range(iters):
        g=np.zeros(n); h=np.zeros(n)
        p=1/(1+np.exp(-(s[pairs[:,0]]-s[pairs[:,1]])))
        r=w-p
        np.add.at(g,pairs[:,0],r); np.add.at(g,pairs[:,1],-r)
        q=p*(1-p); np.add.at(h,pairs[:,0],q); np.add.at(h,pairs[:,1],q)
        g-=0.1*s; h+=0.1
        s+=g/h
    return s
out={}
for cat,tr in truth.items():
    ids=list(tr); ix={k:i for i,k in enumerate(ids)}; t=np.array([tr[k] for k in ids])
    P=[];W=[];DP=[];CONS=[]
    for i,r in rows.items():
        if r["category"]!=cat or "AB" not in R[i] or "BA" not in R[i]: continue
        ab=sm(R[i]["AB"]);ba=sm(R[i]["BA"]); pa=(ab["A"]+ba["B"])/2
        P.append((ix[r["left_id"]],ix[r["right_id"]])); W.append(pa); DP.append(abs(2*pa-1)); CONS.append((ab["A"]>.5)==(ba["B"]>.5))
    P=np.array(P);W=np.array(W);DP=np.array(DP); n=len(ids)
    gt=np.sign(t[P[:,0]]-t[P[:,1]]); pair_acc=float(((W>0.5)==(gt>0)).mean())
    res=dict(pool_pairs=len(P),pair_acc_vs_trueskill=round(pair_acc,3),swap_consistency=round(float(np.mean(CONS)),3))
    e=np.array([E.get((cat,k),np.nan) for k in ids]); res["pointwise_spearman"]=round(spearmanr(e,t,nan_policy="omit")[0],3)
    full_soft=bt(n,P,W); full_hard=bt(n,P,(W>0.5).astype(float))
    res["all_pool_spearman_soft"]=round(spearmanr(full_soft,t)[0],3); res["all_pool_spearman_hard"]=round(spearmanr(full_hard,t)[0],3)
    # Δp calibration on pool vs trueskill
    qb=np.quantile(DP,[.2,.4,.6,.8]); b=np.digitize(DP,qb); res["dp_bins_acc"]=[round(float(((W[b==k]>0.5)==(gt[b==k]>0)).mean()),3) for k in range(5)]
    curves=collections.defaultdict(list)
    for k in [1,2,4,8]:
        m=int(k*n/2)
        for rep in range(20):
            sel=rng.choice(len(P),m,replace=False)
            curves[f"random_soft_k{k}"].append(spearmanr(bt(n,P[sel],W[sel]),t)[0])
            curves[f"random_hard_k{k}"].append(spearmanr(bt(n,P[sel],(W[sel]>0.5).astype(float)),t)[0])
            # pointwise prior + pairwise
            pr=(e-np.nanmean(e))/np.nanstd(e); pr=np.nan_to_num(pr)
            curves[f"prior_plus_soft_k{k}"].append(spearmanr(bt(n,P[sel],W[sel],prior=pr),t)[0])
            # active: half random, half closest current scores
            m0=m//2; sel0=list(rng.choice(len(P),m0,replace=False)); rest=np.setdiff1d(np.arange(len(P)),sel0)
            s=bt(n,P[sel0],W[sel0])
            while len(sel0)<m:
                gap=np.abs(s[P[rest,0]]-s[P[rest,1]]); take=rest[np.argsort(gap)[:max(1,(m-len(sel0))//3)]]
                sel0+=list(take); rest=np.setdiff1d(rest,take); s=bt(n,P[sel0],W[sel0])
            curves[f"active_soft_k{k}"].append(spearmanr(s,t)[0])
    res["curves"]={k:[round(float(np.mean(v)),3),round(float(np.std(v)),3)] for k,v in curves.items()}
    out[cat]=res
json.dump(out,open("summary_rank.json","w"),indent=1); print(json.dumps(out,indent=1))
