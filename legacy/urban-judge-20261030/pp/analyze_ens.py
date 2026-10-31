import json,numpy as np,analyze_pp as A
Ms=["qwen3vl8b","qwen35_9b","qwen25vl7b"]; L={m:A.load(f"raw_sample_{m}.jsonl","sample.jsonl") for m in Ms}
rows=L[Ms[0]][0]; ids=[i for i in rows if rows[i]["choice"]!="equal" and all(("bin","BA") in L[m][1][i] for m in Ms)]
y=np.array([1 if rows[i]["choice"]=="left" else 0 for i in ids])
P=np.array([[A.ptilde(L[m][1],"bin",i)[0] for i in ids] for m in Ms]).T
C=((P>0.5)==(y[:,None]==1))
print("n",len(ids),"acc each",C.mean(0).round(3))
print("error corr",np.corrcoef((~C).T.astype(float)).round(2).tolist())
vote=((P>0.5).sum(1)>=2); cv=(vote==(y==1)); print("majority acc",cv.mean().round(3),"mean-prob acc",(((P.mean(1))>0.5)==(y==1)).mean().round(3))
unan=((P>0.5).all(1))|((P<0.5).all(1)); print("unanimous share",unan.mean().round(3),"acc unanimous",cv[unan].mean().round(3),"acc split",cv[~unan].mean().round(3))
dp=np.abs(2*P[:,0]-1); q=np.quantile(dp,1-unan.mean()); hi=dp>=q
print("single-judge dp top-same-coverage acc",C[hi,0].mean().round(3), "AUROC unanimity",round(A.auroc(unan.astype(float)+0*dp,cv.astype(int)),3),"AUROC mean-dp",round(A.auroc(np.abs(2*P.mean(1)-1),cv.astype(int)),3))
# human-ceiling proxy: trueskill sign
gap=np.array([rows[i]["gap"] for i in ids]); print("acc vs trueskill (majority)",(vote==(gap>0)).mean().round(3))
uw=unan&~cv; uc=unan&cv; vts=((y==1)==(gap>0))
print("unanimous-wrong n",uw.sum(),"vote agrees w/ trueskill there",vts[uw].mean().round(3),"| unanimous-right vote~ts",vts[uc].mean().round(3),"| overall",vts.mean().round(3))
print("unanimous-wrong: models agree with trueskill",(vote[uw]==(gap[uw]>0)).mean().round(3))
