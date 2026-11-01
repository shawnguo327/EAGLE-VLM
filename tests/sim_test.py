import numpy as np, sys
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__file__), '..', 'eaglevlm'))
from eaglev_stats import *
rng=np.random.default_rng(1)
# synthetic judge: true margin m ~ N(0,1); human y ~ Bern(sigmoid(2m)); judge log-odds = 6*m + bias + noise (overconfident -> saturates)
n=3000; m=rng.normal(size=n); y=(rng.random(n)<1/(1+np.exp(-2*m))).astype(int)
lo=6*m+rng.normal(0,4,n); pa=1/(1+np.exp(-lo)); ed=2*m+rng.normal(0,1.5,n)
recs=[dict(pa=pa[i],lo=lo[i],ed=ed[i],y=int(y[i])) for i in range(n)]
S=conf_scores(recs); err=((lo>0).astype(int)!=y).astype(int)
print('acc',1-err.mean(),'sat dp',saturation(S['dp']),'sat lo',saturation(S['abs_logodds']))
for k,v in S.items(): print(k,'auroc',round(auroc(v,1-err),3))
for a in (0.1,0.15,0.2):
  for k in ('dp','abs_logodds'):
    print('alpha',a,k,selective_eval(S[k],err,a))
  print('ladder',a,ladder_eval(recs,a))
# PPI BT: K models, pairs among them
K=8; th=np.linspace(-1,1,K); M=4000
P=rng.integers(0,K,(M,2)); P=P[P[:,0]!=P[:,1]]; M=len(P)
d=th[P[:,0]]-th[P[:,1]]
yh=(rng.random(M)<1/(1+np.exp(-d))).astype(float)
jl=3*d+0.8+rng.normal(0,2,M)  # biased judge (prefers first)
jp=1/(1+np.exp(-jl)); conf=np.abs(jl)
full=bt_fit(P,yh,np.ones(M),K)
for name,pi in (('uniform',np.full(M,0.1)),('routed',routed_propensity(jp,0.1))):
  errs=[]
  for s in range(30):
    lab=rng.random(M)<pi
    t,l=ppi_bt_tuned(P,jp,yh,lab,pi,K,B=8,seed=s); errs.append(np.abs(t-full).mean())
  print(name,'mean|θ-θ_full|',round(np.mean(errs),3))
print('judge-only',round(np.abs(bt_fit(P,jp,np.ones(M),K)-full).mean(),3))
lab=rng.random(M)<0.1; print('human-only 10%',round(np.abs(bt_fit(P[lab],yh[lab],np.ones(lab.sum()),K)-full).mean(),3))

# population check: calibrate on 1500, evaluate true risk on 200k fresh items
def gen(n):
  m=rng.normal(size=n); y=(rng.random(n)<1/(1+np.exp(-2*m))).astype(int); lo=6*m+rng.normal(0,4,n)
  return np.abs(lo), ((lo>0).astype(int)!=y).astype(int)
cp,ep=gen(200000)
for a in (0.1,0.15,0.2):
  v=[];cv=[]
  for s in range(100):
    c,e=gen(1500); tau=calibrate_threshold(c,e,a,0.1); m=cp>=tau; v.append(ep[m].mean()>a if m.any() else False); cv.append(m.mean())
  print('alpha',a,'true-risk violation',np.mean(v),'coverage',round(np.mean(cv),3))
