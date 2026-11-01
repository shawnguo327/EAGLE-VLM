import numpy as np, sys; sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__file__), '..', 'eaglevlm'))
from eaglev_stats import *
rng=np.random.default_rng(2)
K=10; th=np.linspace(-2,2,K)
P=rng.integers(0,K,(6000,2)); P=P[P[:,0]!=P[:,1]]; M=len(P); d=th[P[:,0]]-th[P[:,1]]
yh=(rng.random(M)<1/(1+np.exp(-d))).astype(float)
for jn,jl in (('good',d+rng.normal(0,.3,M)),('biased',1.5*d+0.5+rng.normal(0,1,M))):
  jp=1/(1+np.exp(-jl)); full=bt_fit(P,yh,np.ones(M),K)
  print(jn,'judge-only',round(np.abs(bt_fit(P,jp,np.ones(M),K)-full).mean(),3))
  for name,mk in (('uniform',lambda: np.full(M,.05)),('routed',lambda: routed_propensity(jp,.05))):
    e=[];eh=[];L=[]
    for s in range(20):
      pi=mk(); lab=rng.random(M)<pi
      t,l=ppi_bt_tuned(P,jp,yh,lab,pi,K,B=8,seed=s); e.append(np.abs(t-full).mean()); L.append(l)
      eh.append(np.abs(ppi_bt(P,jp,yh,lab,pi,K,0.0)-full).mean())
    print(' ',name,'PPI',round(np.mean(e),3),'human-IPW',round(np.mean(eh),3),'lam',np.mean(L))
