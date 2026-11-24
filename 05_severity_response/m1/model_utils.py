import numpy as np

def fit(x,y,w,c=None):
    X=np.column_stack([np.ones(len(x)),x] + ([] if c is None else [np.maximum(0,x-c)]))
    b=np.linalg.lstsq(X*np.sqrt(w)[:,None],y*np.sqrt(w),rcond=None)[0]
    rss=np.sum(w*(y-X@b)**2)
    return b,float(len(x)*np.log(max(rss,1e-300)/len(x))+2*(2 if c is None else 4)),float(rss)

def search(x,y,w,mode='historical'):
    o=np.argsort(x);x=x[o];y=y[o];w=w[o];n=len(x)
    lo=max(int(np.floor(.1*n)),4) if mode=='historical' else 4
    hi=min(int(np.ceil(.9*n)),n-4) if mode=='historical' else n-4
    cand=np.unique(x[lo:hi]); b0,a0,r0=fit(x,y,w)
    vals=[fit(x,y,w,c) for c in cand];a=np.array([v[1] for v in vals]);k=int(np.argmin(a))
    return dict(c=float(cand[k]),aic=float(a[k]),linear_aic=a0,delta=a0-a[k],beta=vals[k][0],candidates=cand,aics=a,lo=float(cand.min()),hi=float(cand.max()),n=n)

def weighted_median_sorted(y,block,multiplicity):
    w=multiplicity[block];cs=np.cumsum(w);total=cs[-1] if len(cs) else 0
    if total==0:return np.nan
    # Match np.median for replicated integer observations.
    a=np.searchsorted(cs,(total+1)//2,side='left');b=np.searchsorted(cs,total//2+1,side='left')
    return .5*(y[a]+y[b])

def km_counts(e,c):
    """Discrete observed recovery times; event before censor on ties. Exact step integral."""
    e=np.asarray(e,float);c=np.asarray(c,float)
    risk=np.flip(np.cumsum(np.flip(e+c,axis=-1),axis=-1),axis=-1)
    h=np.divide(e,risk,out=np.zeros_like(e),where=risk>0)
    s=np.cumprod(1-h,axis=-1)
    supported=(risk[...,120]>0)|(s[...,-1]==0)
    p=np.where(supported,1-s[...,120],np.nan)
    rm=np.where(supported,s[...,:120].sum(axis=-1),np.nan)
    return p,rm,risk[...,120],s
