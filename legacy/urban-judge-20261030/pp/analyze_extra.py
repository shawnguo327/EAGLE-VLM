import json,sys,math,numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict,StratifiedKFold
import analyze_pp as A
def run(model):
    rows,R=A.load(f"raw_sample_{model}.jsonl","sample.jsonl"); X=[];y=[];cats=[]
    for i,r in rows.items():
        if r["choice"]=="equal" or ("score","A") not in R[i]: continue
        ab=R[i][("bin","AB")]["lp"]; ba=R[i][("bin","BA")]["lp"]
        lo=((ab["A"]-ab["B"])+(ba["B"]-ba["A"]))/2   # mean log-odds for left
        ed=A.escore(R,i,"A")-A.escore(R,i,"B")
        fa,fb=A.facts(R,i,"A"),A.facts(R,i,"B")
        fd=list(np.array(fa)-np.array(fb)) if fa and fb else [np.nan]*12
        X.append([lo,ed]+fd); y.append(1 if r["choice"]=="left" else 0); cats.append(r["category"])
    X=np.array(X);y=np.array(y); X=np.nan_to_num(X); cats=np.array(cats); out={}
    lo=X[:,0]; corr=((lo>0)==(y==1)).astype(int)
    out["auroc_abs_logodds_correct"]=round(A.auroc(np.abs(lo),corr),3)
    qb=np.quantile(np.abs(lo),[.2,.4,.6,.8]); b=np.digitize(np.abs(lo),qb)
    out["logodds_quintile_acc"]=[round(corr[b==k].mean(),3) for k in range(5)]
    for name,cols in [("pairwise_lo",[0]),("pointwise",[1]),("pair+point",[0,1]),("facts",list(range(2,14))),("all",list(range(14)))]:
        pred=np.zeros(len(y))
        for c in set(cats):
            m=cats==c; Xs=np.r_[X[m][:,cols],-X[m][:,cols]]; ys=np.r_[y[m],1-y[m]]
            pred[m]=cross_val_predict(LogisticRegression(C=0.3,max_iter=3000),Xs,ys,cv=StratifiedKFold(5,shuffle=True,random_state=0),method="predict_proba")[:,1][:m.sum()]
        out["cv_acc_"+name]=round(((pred>0.5)==(y==1)).mean(),3)
    return out
print(json.dumps({m:run(m) for m in sys.argv[1:]},indent=1))
