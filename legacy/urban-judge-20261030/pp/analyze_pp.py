import json,sys,math,re,collections,numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
rng=np.random.default_rng(0)
def auroc(score,y):
    score=np.asarray(score,float);y=np.asarray(y,int)
    pos=score[y==1];neg=score[y==0]
    if len(pos)==0 or len(neg)==0: return float("nan")
    r=np.argsort(np.argsort(np.concatenate([pos,neg])))+1
    return float((r[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))
def boot(f,n,B=1000):
    v=[f(rng.integers(0,n,n)) for _ in range(B)]; return [round(float(np.percentile(v,2.5)),3),round(float(np.percentile(v,97.5)),3)]
def sm(lp):
    m=max(lp.values()); e={k:math.exp(v-m) for k,v in lp.items()}; z=sum(e.values()); return {k:v/z for k,v in e.items()}
def load(raw,inp):
    rows={json.loads(l)["id"]:json.loads(l) for l in open(inp)}
    R=collections.defaultdict(dict)
    for l in open(raw):
        d=json.loads(l); R[d["id"]][(d["cond"],d["order"])]=d
    return rows,R
def ptilde(R,c,i):
    ab=sm(R[i][(c,"AB")]["lp"]); ba=sm(R[i][(c,"BA")]["lp"])
    if c=="tri":
        pa=(ab["A"]+ba["B"])/2; pb=(ab["B"]+ba["A"])/2; pc=(ab["C"]+ba["C"])/2
        return pa,pb,pc,ab,ba
    pa=(ab["A"]+ba["B"])/2
    return pa,(ab["A"]>0.5)==(ba["B"]>0.5),ab["A"],ba["A"]
def escore(R,i,o):
    p=sm(R[i][("score",o)]["lp"]); return sum(int(k)*v for k,v in p.items())
KEYS=["greenery","sky_open","cars","people","building_upkeep","litter_graffiti","road_width","shops_signs","building_height_floors","sidewalk_quality","fences_walls","vacant_or_derelict"]
def facts(R,i,o):
    d=R[i].get(("facts",o))
    if not d: return None
    out=[]
    for k in KEYS:
        m=re.search(r"\""+k+r"\"\s*:\s*(-?\d+(\.\d+)?)",d["text"]); out.append(float(m.group(1)) if m else np.nan)
    return out
def main(model,tag="sample"):
    rows,R=load(f"raw_{tag}_{model}.jsonl",f"{tag}.jsonl")
    ids=[i for i in rows if all(k in R[i] for k in [("bin","AB"),("bin","BA")])]
    res={"model":model,"n":len(ids)}
    if tag=="sample":
        S=[];
        for i in ids:
            r=rows[i]; pa,cons,pab,pba=ptilde(R,"bin",i)
            d=dict(id=i,cat=r["category"],tie=r["choice"]=="equal",y=1 if r["choice"]=="left" else 0,gap=r["gap"],pa=pa,dp=abs(2*pa-1),cons=cons,pfirst=(pab+(1-pba))/2)
            if ("low","AB") in R[i]: pl,_,_,_=ptilde(R,"low",i); d.update(pl=pl,dpl=abs(2*pl-1))
            if ("tri","AB") in R[i]: a,b,c,ab,ba=ptilde(R,"tri",i); d.update(tA=a,tB=b,tC=c,tiepick=int(np.argmax([ab["A"],ab["B"],ab["C"]])==2)+int(np.argmax([ba["A"],ba["B"],ba["C"]])==2))
            if ("score","A") in R[i]: d.update(eA=escore(R,i,"A"),eB=escore(R,i,"B"))
            fa,fb=facts(R,i,"A"),facts(R,i,"B")
            if fa and fb: d["fd"]=np.array(fa)-np.array(fb)
            S.append(d)
        B=[d for d in S if not d["tie"]]; y=np.array([d["y"] for d in B]); pa=np.array([d["pa"] for d in B]); dp=np.array([d["dp"] for d in B])
        corr=((pa>0.5)==(y==1)).astype(int); n=len(B)
        res["binary_n"]=n; res["acc_direct"]=round(corr.mean(),3); res["acc_direct_ci"]=boot(lambda ix:corr[ix].mean(),n)
        res["swap_consistency"]=round(np.mean([d["cons"] for d in B]),3); res["mean_P_first"]=round(np.mean([d["pfirst"] for d in S]),3)
        gap=np.array([d["gap"] for d in B]); ysk=(gap>0).astype(int)
        res["acc_vs_trueskill_sign"]=round(((pa>0.5)==(ysk==1)).mean(),3)
        res["human_vote_vs_trueskill_sign"]=round((ysk==y).mean(),3)
        # by |gap| tercile
        qs=np.quantile(np.abs(gap),[1/3,2/3]); res["acc_by_gap_tercile"]=[]
        for lo,hi in [(0,qs[0]),(qs[0],qs[1]),(qs[1],1e9)]:
            m=(np.abs(gap)>=lo)&(np.abs(gap)<hi); res["acc_by_gap_tercile"].append(dict(n=int(m.sum()),acc_vote=round(corr[m].mean(),3),acc_ts=round(((pa[m]>0.5)==(ysk[m]==1)).mean(),3),vote_ts_agree=round((ysk[m]==y[m]).mean(),3),mean_dp=round(dp[m].mean(),3)))
        # calibration: dp quintiles
        qb=np.quantile(dp,[.2,.4,.6,.8]); bins=np.digitize(dp,qb); res["dp_calibration"]=[dict(bin=int(b),n=int((bins==b).sum()),dp_mean=round(dp[bins==b].mean(),3),acc=round(corr[bins==b].mean(),3),acc_ts=round(((pa[bins==b]>0.5)==(ysk[bins==b]==1)).mean(),3)) for b in range(5)]
        res["auroc_dp_correct"]=round(auroc(dp,corr),3); res["auroc_dp_correct_ts"]=round(auroc(dp,((pa>0.5)==(ysk==1)).astype(int)),3)
        res["spearman_dp_absgap"]=round(spearmanr(dp,np.abs(gap))[0],3)
        res["by_category"]={}
        for c in sorted(set(d["cat"] for d in B)):
            m=np.array([d["cat"]==c for d in B]); res["by_category"][c]=dict(n=int(m.sum()),acc=round(corr[m].mean(),3),acc_ts=round(((pa[m]>0.5)==(ysk[m]==1)).mean(),3),swap=round(np.mean([d["cons"] for d,mm in zip(B,m) if mm]),3),auroc_dp=round(auroc(dp[m],corr[m]),3))
        # low-res
        if "pl" in B[0]:
            pl=np.array([d["pl"] for d in B]); dpl=np.array([d["dpl"] for d in B]); cl=((pl>0.5)==(y==1)).astype(int)
            res["lowres"]=dict(acc=round(cl.mean(),3),agree_with_full=round(((pl>0.5)==(pa>0.5)).mean(),3),spearman_dp_dpl=round(spearmanr(dp,dpl)[0],3),auroc_dpl_correct_full=round(auroc(dpl,corr),3))
            X=np.c_[dp,dpl,((pl>0.5)==(pa>0.5)).astype(float)]; cv=StratifiedKFold(5,shuffle=True,random_state=0)
            p1=cross_val_predict(LogisticRegression(),dp[:,None],corr,cv=cv,method="predict_proba")[:,1]; p2=cross_val_predict(LogisticRegression(),X,corr,cv=cv,method="predict_proba")[:,1]
            res["lowres"]["cv_auroc_error_pred_dp_only"]=round(auroc(p1,corr),3); res["lowres"]["cv_auroc_error_pred_dp_plus_low"]=round(auroc(p2,corr),3)
        # pointwise + EAGLE gate
        if "eA" in B[0]:
            ed=np.array([d["eA"]-d["eB"] for d in B]); cp=np.where(ed==0,0.5,((ed>0)==(y==1)).astype(float))
            res["pointwise"]=dict(acc=round(cp.mean(),3),tie_rate_rounded=round(np.mean(np.round([d["eA"] for d in B])==np.round([d["eB"] for d in B])),3),agree_with_pairwise=round(((ed>0)==(pa>0.5)).mean(),3))
            gate=[]
            for tau in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
                use=dp>=tau; acc=np.where(use,corr,cp).mean(); gate.append(dict(tau=tau,coverage_pairwise=round(use.mean(),3),acc=round(acc,3)))
            res["eagle_gate"]=gate
            agree=((ed>0)==(pa>0.5)); res["pointwise"]["acc_pairwise_when_agree"]=round(corr[agree].mean(),3); res["pointwise"]["acc_pairwise_when_disagree"]=round(corr[~agree].mean(),3); res["pointwise"]["share_disagree"]=round((~agree).mean(),3)
        # facts look-then-match (CV logistic on fact differences, per all categories with category interaction)
        F=[d for d in B if "fd" in d and not np.isnan(d["fd"]).any()]
        if len(F)>200:
            cats=sorted(set(d["cat"] for d in F)); yy=np.array([d["y"] for d in F]); pf=np.zeros(len(F))
            accs={}
            for c in cats:
                m=np.array([d["cat"]==c for d in F]); X=np.array([d["fd"] for d in F])[m]
                Xs=np.r_[X,-X]; ys=np.r_[yy[m],1-yy[m]]  # symmetrize
                pp=cross_val_predict(LogisticRegression(C=0.3,max_iter=2000),Xs,ys,cv=StratifiedKFold(5,shuffle=True,random_state=0),method="predict_proba")[:,1][:m.sum()]
                pf[m]=pp; accs[c]=round(((pp>0.5)==(yy[m]==1)).mean(),3)
            paf=np.array([d["pa"] for d in F]); cj=((paf>0.5)==(yy==1)).astype(int); cf=((pf>0.5)==(yy==1)).astype(int)
            dpf=np.abs(2*paf-1); fconf=np.abs(2*pf-1)
            res["facts"]=dict(n=len(F),acc_facts_lr=round(cf.mean(),3),acc_judge_same=round(cj.mean(),3),by_cat=accs,agree=round(((pf>0.5)==(paf>0.5)).mean(),3))
            # router: follow facts when fact-confidence high and judge dp low
            rt=[]
            for tau in [0.2,0.4,0.6,0.8]:
                use_f=(dpf<tau)&(fconf>dpf); acc=np.where(use_f,cf,cj).mean(); rt.append(dict(tau=tau,share_facts=round(use_f.mean(),3),acc=round(acc,3)))
            res["facts"]["router"]=rt
            # average of probabilities
            res["facts"]["acc_avg_prob"]=round((((paf+pf)/2>0.5)==(yy==1)).mean(),3)
        # ties
        if "tC" in S[0]:
            tie=np.array([d["tie"] for d in S]).astype(int); tc=np.array([d["tC"] for d in S]); dpa=np.array([d["dp"] for d in S])
            res["ties"]=dict(human_tie_share_in_sample=round(tie.mean(),3),model_tie_pick_rate=round(np.mean([d["tiepick"] for d in S])/2,3),auroc_pC_detect_human_tie=round(auroc(tc,tie),3),auroc_lowdp_detect_human_tie=round(auroc(-dpa,tie),3),mean_pC_on_ties=round(tc[tie==1].mean(),3),mean_pC_on_nonties=round(tc[tie==0].mean(),3))
    else:
        M=[]
        for i in ids:
            r=rows[i]; v=r["votes"]; a,b,e=v.get("a",0),v.get("b",0),v.get("equal",0)
            if a+b==0: continue
            pa,cons,_,_=ptilde(R,"bin",i); d=dict(h=a/(a+b),agree=abs(2*a/(a+b)-1),tieshare=e/(a+b+e),pa=pa,dp=abs(2*pa-1),n=a+b+e,cat=r["category"],cons=cons)
            if ("tri","AB") in R[i]: d["tC"]=ptilde(R,"tri",i)[2]
            if ("low","AB") in R[i]: d["dpl"]=abs(2*ptilde(R,"low",i)[0]-1)
            M.append(d)
        h=np.array([d["h"] for d in M]); pa=np.array([d["pa"] for d in M]); dp=np.array([d["dp"] for d in M]); ag=np.array([d["agree"] for d in M])
        nz=h!=0.5
        res["n_pairs"]=len(M); res["mean_votes"]=round(np.mean([d["n"] for d in M]),2)
        res["acc_vs_majority"]=round(((pa[nz]>0.5)==(h[nz]>0.5)).mean(),3)
        strong=ag>=0.6; res["acc_vs_majority_strong_consensus"]=dict(n=int((strong&nz).sum()),acc=round(((pa[strong&nz]>0.5)==(h[strong&nz]>0.5)).mean(),3))
        res["acc_vs_majority_weak_consensus"]=dict(n=int((~strong&nz).sum()),acc=round(((pa[~strong&nz]>0.5)==(h[~strong&nz]>0.5)).mean(),3))
        res["spearman_pa_h"]=round(spearmanr(pa,h)[0],3); res["spearman_dp_human_agreement"]=round(spearmanr(dp,ag)[0],3)
        # human ceiling: leave-one-vote-out agreement approx = expected prob a random voter matches majority
        res["human_vote_majority_agreement"]=round(np.mean([max(d["h"],1-d["h"]) for d in M]),3)
        if "tC" in M[0]:
            tc=np.array([d["tC"] for d in M]); ts=np.array([d["tieshare"] for d in M])
            res["spearman_pC_human_tieshare"]=round(spearmanr(tc,ts)[0],3); res["spearman_pC_human_disagreement"]=round(spearmanr(tc,-ag)[0],3)
        if "dpl" in M[0]:
            res["spearman_dplow_human_agreement"]=round(spearmanr([d["dpl"] for d in M],ag)[0],3)
        res["swap_consistency"]=round(np.mean([d["cons"] for d in M]),3)
    return res
if __name__=="__main__":
    out={}
    for m in sys.argv[1:]:
        for tag in ["sample","multi"]:
            try: out[f"{m}_{tag}"]=main(m,tag)
            except FileNotFoundError: pass
    json.dump(out,open("summary_pp.json","w"),indent=1,default=float); print(json.dumps(out,indent=1,default=float))
