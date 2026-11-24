import numpy as np
from scipy.stats import chi2

def estimates(counts):
    """Last axes: 5 outcomes x121 days. Codes0 recovery,1 competing,2/3 admin,4 missing.
    Aggregate product integral handles tied discrete times without stochastic jitter.
    """
    c=np.asarray(counts,float);tot=c.sum(axis=-2)
    risk=np.flip(np.cumsum(np.flip(tot,axis=-1),axis=-1),axis=-1)
    er=c[...,0,:];en=c[...,1,:]
    hr=np.divide(er,risk,out=np.zeros_like(er),where=risk>0)
    hn=np.divide(en,risk,out=np.zeros_like(en),where=risk>0)
    skm=np.cumprod(1-hr,axis=-1);sall=np.cumprod(1-hr-hn,axis=-1)
    prev=np.concatenate([np.ones_like(sall[...,:1]),sall[...,:-1]],axis=-1)
    cifr=np.cumsum(prev*hr,axis=-1);cifn=np.cumsum(prev*hn,axis=-1)
    kmok=(risk[...,-1]>0)|(skm[...,-1]<=1e-14)
    ajok=(risk[...,-1]>0)|(sall[...,-1]<=1e-14)
    return dict(P120=np.where(kmok,1-skm[...,-1],np.nan),RMPT120=np.where(kmok,skm[...,:120].sum(axis=-1),np.nan),RMPT_Trapezoid=np.where(kmok,(skm[...,:-1]+skm[...,1:]).sum(axis=-1)/2,np.nan),AJ_Recovery120=np.where(ajok,cifr[...,-1],np.nan),AJ_Next120=np.where(ajok,cifn[...,-1],np.nan),N_risk_120=risk[...,-1],KM_S=skm,AJ_S=sall,AJ_Recovery_Curve=cifr,AJ_Next_Curve=cifn,Risk=risk)

def logrank(counts):
    c=np.asarray(counts,float);e=c[:,0,:];allcount=c.sum(axis=1)
    Y=np.flip(np.cumsum(np.flip(allcount,axis=-1),axis=-1),axis=-1)
    n=Y.sum(axis=0);d=e.sum(axis=0);p=np.divide(Y[1],n,out=np.zeros(121),where=n>0)
    O=float(e[1].sum());E=float(np.sum(d*p))
    factor=np.divide(d*(n-d),n-1,out=np.zeros(121),where=n>1)
    V=float(np.sum(factor*p*(1-p)))
    stat=(O-E)**2/V if V>0 else np.nan
    return dict(Logrank_Chi2=stat,Logrank_df=1,Logrank_p=float(chi2.sf(stat,1)) if np.isfinite(stat) else np.nan,Extreme_Observed_Recoveries=O,Extreme_Expected_Recoveries=E,Logrank_Variance=V)

def numerical_tests():
    c=np.zeros((5,121));c[0,10]=1;c[1,20]=1;c[2,120]=2
    f=estimates(c);assert np.isclose(f['P120'],.25);assert np.isclose(f['AJ_Recovery120'],.25);assert np.isclose(f['AJ_Next120'],.25);assert np.isclose(f['RMPT120'],92.5)
    c=np.zeros((5,121));c[1,10]=1;c[0,20]=1;c[2,120]=2
    f=estimates(c);assert np.isclose(f['P120'],1/3);assert np.isclose(f['AJ_Recovery120'],.25)
    c=np.zeros((5,121));c[0,10]=1;c[1,10]=1;c[2,120]=2
    f=estimates(c);assert np.isclose(f['AJ_Recovery120'],.25) and np.isclose(f['AJ_Next120'],.25)
    c=np.zeros((5,121));c[2,20]=1
    f=estimates(c);assert np.isnan(f['P120']) and np.isnan(f['AJ_Recovery120'])
    # Two identical groups have log-rank chi-square zero.
    c=np.zeros((2,5,121));c[:,0,10]=2;c[:,2,120]=3;assert logrank(c)['Logrank_Chi2']==0
    return True
