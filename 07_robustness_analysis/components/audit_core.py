# -*- coding: utf-8 -*-
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import numpy as np
from numba import njit,prange
R=Path(r'.');O=R/'analysis_project/Reviewer2_M1_IndicatorAudit';WORK=O/'qc/work_cache'
VARS=['ZSync','ZuWUE','ZNIRv','Acomp','WithoutSync','WithoutuWUE','WithoutNIRv']
PAIRS=((0,1),(0,2),(1,2),(3,4),(3,5),(3,6))
@njit(cache=True,parallel=True)
def prefix_midrank_lookup(sorted_values,values,buckets,prefix):
 n=len(sorted_values);out=np.empty(len(values),np.float64)
 for i in prange(len(values)):
  v=values[i];b=buckets[i];a=prefix[b];end=prefix[b+1];z=end
  while a<z:
   mid=(a+z)//2
   if sorted_values[mid]<v:a=mid+1
   else:z=mid
  lo=a
  if lo+1>=end or sorted_values[lo+1]!=v:hi=lo+1
  else:
   a=lo+1;z=end
   while a<z:
    mid=(a+z)//2
    if sorted_values[mid]<=v:a=mid+1
    else:z=mid
   hi=a
  out[i]=(lo+hi+1)/(2.*n)
 return out
@njit(cache=True)
def event_values(z,ac,td,onset,term,col,peak,rec,confirm,base):
 n=len(onset);out=np.full((n,44),np.nan,np.float64)
 for i in range(n):
  s=onset[i];t=term[i];c=col[i];r=rec[i];k=confirm[i];start=np.searchsorted(td,td[s]-40)
  out[i,0]=base[i];out[i,1]=ac[r,c];out[i,2]=ac[k,c]
  for j in range(3):
   values=np.empty(s-start,np.float64);dam=np.empty(s-start,np.float64);nn=0
   for q in range(start,s):
    a=z[j,q,c]
    if np.isfinite(a):values[nn]=a;dam[nn]=max(0.,-max(-10.,min(10.,a)));nn+=1
   out[i,3+j]=nn
   if nn>=3:
    out[i,6+j]=np.median(dam[:nn]);out[i,9+j]=np.median(values[:nn])
   a=z[j,r,c];b=z[j,k,c];out[i,12+j]=a;out[i,15+j]=b
   if np.isfinite(a):out[i,18+j]=max(0.,-max(-10.,min(10.,a)))
   if np.isfinite(b):out[i,21+j]=max(0.,-max(-10.,min(10.,b)))
  valid=True
  for j in range(3):
   if not (np.isfinite(out[i,6+j]) and np.isfinite(out[i,9+j]) and np.isfinite(out[i,18+j]) and np.isfinite(out[i,21+j])):valid=False
  out[i,24]=int(valid)
  if valid:
   pattern=0;zp=0
   for j in range(3):
    a=out[i,18+j]<=out[i,6+j] and out[i,21+j]<=out[i,6+j]
    b=out[i,12+j]>=out[i,9+j] and out[i,15+j]>=out[i,9+j]
    out[i,25+j]=int(a);out[i,28+j]=int(b);pattern+=int(a)*(1<<j);zp+=int(b)*(1<<j)
   out[i,31]=pattern;out[i,32]=zp
   out[i,33]=np.sqrt((out[i,18]**2+out[i,19]**2+out[i,20]**2)/3)
   out[i,34]=np.sqrt((out[i,21]**2+out[i,22]**2+out[i,23]**2)/3)
  buf=np.empty(s-start,np.float32);nn=0
  for q in range(start,s):
   a=ac[q,c]
   if np.isfinite(a):buf[nn]=a;nn+=1
  out[i,35]=nn
  if nn>=3:out[i,36]=np.median(buf[:nn])
  out[i,37]=td[r]-td[peak[i]];out[i,38]=td[k]-td[term[i]];out[i,39]=k-r
 return out

@njit(cache=True)
def kth_code(pool,start,end,weights,k):
 lo=np.uint32(0);hi=np.uint32(0x3f800000)
 while lo<hi:
  mid=np.uint32((np.uint64(lo)+np.uint64(hi))//2);total=0
  for b in range(len(weights)):
   if weights[b]==0:continue
   a=start[b];z=end[b]
   while a<z:
    m=(a+z)//2
    if pool[m]<=mid:a=m+1
    else:z=m
   total+=(a-start[b])*weights[b]
  if total>=k:hi=mid
  else:lo=mid+np.uint32(1)
 return lo

@njit(cache=True)
def mixture_quantile_codes(pool,start,end,W,probs):
 out=np.empty((len(W),len(probs),2),np.uint32);fraction=np.empty((len(W),len(probs)),np.float64)
 for i in range(len(W)):
  n=0
  for b in range(len(start)):n+=(end[b]-start[b])*W[i,b]
  for j,p in enumerate(probs):
   rank=(n-1)*p;a=int(np.floor(rank));b=int(np.ceil(rank));fraction[i,j]=rank-a
   out[i,j,0]=kth_code(pool,start,end,W[i],a+1);out[i,j,1]=kth_code(pool,start,end,W[i],b+1)
 return out,fraction

@njit(cache=True)
def reranked_spearman_bootstrap(X,orders,blocks,W):
 n,p=X.shape;out=np.empty((len(W),len(PAIRS)),np.float64);ranks=np.empty((n,p),np.float64)
 for t in range(len(W)):
  weights=W[t,blocks];total=weights.sum();mu=(total+1)/2.
  for j in range(p):
   q=orders[:,j];i=0;cum=0
   while i<n:
    k=i+1
    while k<n and X[q[k],j]==X[q[i],j]:k+=1
    nn=0
    for a in range(i,k):nn+=weights[q[a]]
    rank=cum+(nn+1)/2.
    for a in range(i,k):ranks[q[a],j]=rank
    cum+=nn;i=k
  for j in range(len(PAIRS)):
   aa,bb=PAIRS[j];xy=0.;xx=0.;yy=0.
   for i in range(n):
    w=weights[i];a=ranks[i,aa]-mu;b=ranks[i,bb]-mu;xy+=w*a*b;xx+=w*a*a;yy+=w*b*b
   out[t,j]=xy/np.sqrt(xx*yy) if xx>0 and yy>0 else np.nan
 return out

def sufficient(X,blocks,B):
 s=np.zeros((B,21),np.float64)
 for b in np.unique(blocks):
  v=np.asarray(X[blocks==b],np.float64);s[b,0]=len(v);s[b,1:8]=v.sum(axis=0);s[b,8:15]=np.sum(v*v,axis=0)
  for k,(a,c) in enumerate(PAIRS):s[b,15+k]=np.dot(v[:,a],v[:,c])
 return s

def corr_from_sufficient(s):
 n=s[...,0];mean=np.divide(s[...,1:8],n[...,None],out=np.zeros_like(s[...,1:8]),where=n[...,None]>0)
 var=s[...,8:15]-n[...,None]*mean*mean;out=[]
 for j,(a,b) in enumerate(PAIRS):
  cov=s[...,15+j]-n*mean[...,a]*mean[...,b];den=np.sqrt(np.maximum(0,var[...,a])*np.maximum(0,var[...,b]));out.append(np.divide(cov,den,out=np.full_like(cov,np.nan),where=den>0))
 return np.stack(out,axis=-1)

def add_pixel_sums(target,X,pix):
 target[:,0]+=np.bincount(pix,minlength=len(target))
 for j in range(7):target[:,j+1]+=np.bincount(pix,weights=X[:,j],minlength=len(target))

def write_case(name,X,blocks,pix,registry):
 folder=WORK/name;folder.mkdir(parents=True,exist_ok=True);B=76
 with open(folder/'values.bin','ab') as f:np.asarray(X,np.float32).tofile(f)
 with open(folder/'blocks.bin','ab') as f:np.asarray(blocks,np.uint8).tofile(f)
 registry[name]['stats']+=sufficient(X,blocks,B)
 add_pixel_sums(registry[name]['pixels'],X,pix)
 for j in range(3):
  diff=np.asarray(X[:,4+j],np.float64)-np.asarray(X[:,3],np.float64)
  registry[name]['absdiff'][:,j]+=np.bincount(blocks,weights=abs(diff),minlength=B)
 registry[name]['N']+=len(X)
 # Exactfloat32contributions; squared terms computed from original cappednegative anomalies.
 a=np.maximum(0,-np.clip(X[:,:3].astype(np.float64),-10,10));q=a*a;den=q.sum(axis=1);positive=den>0
 registry[name]['zero']+=np.bincount(blocks[~positive],minlength=B)
 if positive.any():
  c=(q[positive]/den[positive,None]).astype(np.float32);bb=blocks[positive]
  registry[name]['Csum_error']=max(registry[name]['Csum_error'],float(np.max(abs(c.astype(float).sum(axis=1)-1))))
  for b in np.unique(bb):
   cc=c[bb==b]
   with open(folder/f'contrib_block_{int(b):02d}.bin','ab') as f:cc.tofile(f)
   registry[name]['Cstats'][b,0]+=len(cc);registry[name]['Cstats'][b,1:4]+=cc.sum(axis=0,dtype=np.float64)
 return int(positive.sum())
