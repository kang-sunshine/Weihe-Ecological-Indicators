# -*- coding: utf-8 -*-
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import ast,json
import numpy as np
import pandas as pd
from numba import njit
R=Path(r'.');P=R/'analysis_project';O=P/'Reviewer2_M2_EstimationAudit';M1=R/'EI_Reviewer1_Major1_Revision';M2=R/'EI_Reviewer1_Major2_Revision';M3=R/'EI_Reviewer1_Major3_Revision';M5=R/'EI_Reviewer1_Major5_Revision';MIN=R/'EI_Reviewer1_Minor2_SingleIndicator';DB=R/'Weihe database V3'
HAZ=['CDH','HEAT','HR'];WINDOWS=[16,32,56,120];GROUPS=['Low','Moderate','High','Extreme','All'];SEED=20261117

def dirs():
 for s in ['scripts','outputs','outputs/event_details','qc','qc/cache']:(O/s).mkdir(parents=True,exist_ok=True)
def events(h):
 d=pd.read_parquet(MIN/'paired'/f'{h}_Fixed_Clean_Events.parquet',use_threads=False)
 for name,dtype in [('row','int16'),('col','int16'),('onset_idx','int16'),('term_idx','int16'),('event_num','int32'),('valid_resistance_sample','int8')]:
  if name in d:d[name]=d[name].astype(dtype)
 import pyarrow as pa
 pa.default_memory_pool().release_unused()
 return d
def existing(h):return np.load(MIN/'cache'/h/'Acomp.npy',mmap_mode='r')
def groups(h):return np.searchsorted([50,75,90],np.load(MIN/'cache'/h/'Full_ECDF_Q.npy',mmap_mode='r'),side='right')
def blockdata(h):
 b=np.load(MIN/'cache'/h/'Block50.npy',mmap_mode='r');W=np.load(MIN/'cache'/h/'Bootstrap_50km_Weights.npy');return b,W
def ast_function(path,name):
 tree=ast.parse(path.read_text(encoding='utf-8-sig'));node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name);node.decorator_list=[]
 env={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(O/'scripts'/'ast_readonly_functions.py'),'exec'),env);return env[name]
def ci(values):
 v=np.asarray(values);ok=np.isfinite(v);n=int(ok.sum());q=np.quantile(v[ok],[.025,.975]) if n>=200 else [np.nan,np.nan];return dict(CI_Lower=q[0],CI_Upper=q[1],Bootstrap_N=len(v),Bootstrap_Valid=n,Bootstrap_Failed=int(len(v)-n),CI_Status='ESTIMABLE_CONDITIONAL' if n>=200 else 'NOT ESTIMABLE')
def support(mask,pix,b):return dict(N_events=int(mask.sum()),N_pixels=int(np.unique(pix[mask]).size),N_50km_blocks=int(np.unique(b[mask]).size))
def ratio(mask,pop,b,W):
 nn=np.bincount(b[mask],minlength=76);dd=np.bincount(b[pop],minlength=76);den=W@dd;v=np.divide(W@nn,den,out=np.full(len(W),np.nan),where=den>0);return nn.sum()/dd.sum() if dd.sum() else np.nan,v
def paired_mean(x,mask,b,W):
 n=np.bincount(b[mask],minlength=76);s=np.bincount(b[mask],weights=np.asarray(x[mask],float),minlength=76);den=W@n;v=np.divide(W@s,den,out=np.full(len(W),np.nan),where=den>0);return s.sum()/n.sum() if n.sum() else np.nan,v
@njit(cache=True)
def weighted_quantile_sorted(v,block,W,prob):
 out=np.empty(len(W),np.float64)
 for r in range(len(W)):
  total=0
  for j in range(len(v)):total+=W[r,block[j]]
  if total==0:out[r]=np.nan;continue
  target=(total-1)*prob;lo=int(np.floor(target));hi=int(np.ceil(target));frac=target-lo;c=0;a=0.;z=0.;got=False
  for j in range(len(v)):
   c+=W[r,block[j]]
   if not got and c>lo:a=v[j];got=True
   if c>hi:z=v[j];break
  out[r]=a*(1-frac)+z*frac
 return out
def median_boot(v,b,W):
 order=np.argsort(v,kind='stable');return weighted_quantile_sorted(v[order],b[order],W,.5)

@njit(cache=True)
def mix_kth(pool,start,end,weights,k,upper):
 lo=np.uint64(0);hi=upper
 while lo<hi:
  mid=lo+(hi-lo)//np.uint64(2);count=0
  for b in range(len(start)):
   if weights[b]==0:continue
   a=start[b];z=end[b]
   while a<z:
    j=(a+z)//2
    if pool[j]<=mid:a=j+1
    else:z=j
   count+=(a-start[b])*weights[b]
  if count>=k:hi=mid
  else:lo=mid+np.uint64(1)
 return lo
@njit(cache=True)
def mix_quantiles(pool,start,end,W,probs):
 codes=np.empty((len(W),len(probs),2),np.uint64);frac=np.empty((len(W),len(probs)),np.float64);upper=np.uint64(0)
 for b in range(len(start)):
  if end[b]>start[b]:upper=max(upper,pool[end[b]-1])
 for r in range(len(W)):
  n=0
  for b in range(len(start)):n+=(end[b]-start[b])*W[r,b]
  if n<=0:
   for j in range(len(probs)):codes[r,j,0]=0;codes[r,j,1]=0;frac[r,j]=np.nan
   continue
  for j,p in enumerate(probs):
   target=(n-1)*p;lo=int(np.floor(target));hi=int(np.ceil(target));frac[r,j]=target-lo
   codes[r,j,0]=mix_kth(pool,start,end,W[r],lo+1,upper);codes[r,j,1]=mix_kth(pool,start,end,W[r],hi+1,upper)
 return codes,frac
def quantile_boot(v,b,W,probs=np.array([.05,.25,.5,.75,.95])):
 # Shift by fixed cohort minimum; float64 shift may alter last-bit rounding only.
 # Legacy/duration/intensity nonnegative use zero shift and preserve source floats exactly.
 v=np.asarray(v,np.float64);shift=min(0.,float(np.min(v)));x=v-shift
 counts=np.bincount(b,minlength=76);start=np.r_[0,np.cumsum(counts)[:-1]];end=np.cumsum(counts)
 pool=np.concatenate([np.sort(x[b==k]) for k in range(76)]).view(np.uint64)
 codes,f=mix_quantiles(pool,start,end,W.astype(np.int32),np.asarray(probs));vv=codes.view(np.float64);return vv[:,:,0]*(1-f)+vv[:,:,1]*f+shift

@njit(cache=True)
def legacy_audit(ss,tt,cc,td,a,active,m):
 # Historical fields0..11; fixed L12..15,days16..19,exclusionbits20..23.
 out=np.full((len(ss),24),np.nan,np.float32);nt=len(td)
 for i in range(len(ss)):
  s=ss[i];t=tt[i];c=cc[i];b=m[i,0];pk=int(m[i,6]) if np.isfinite(m[i,6]) else -1
  if not np.isfinite(b) or pk<0:
   out[i,4]=9
   for w in range(4):out[i,20+w]=1
   continue
  rend=np.searchsorted(td,td[s]+120,side='right');nxt=-1
  for k in range(t+1,rend):
   if active[k,c]:nxt=k;break
  stop=nxt if nxt>=0 else rend;rec=-1;run=0
  for k in range(pk+1,stop):
   val=a[k,c]
   if not np.isfinite(val):run=0;continue
   run=run+1 if val<=b else 0
   if run>=2:rec=k-1;break
  final=rec if rec>=0 else stop
  reason=0 if rec>=0 else 1 if nxt>=0 else 3 if td[-1]<td[s]+120 else 2
  ll=0.;weighted_days=0;observed_days=0;tail=0;miss=0;first=-1;last=-1
  for k in range(t+1,final):
   val=a[k,c]
   if not np.isfinite(val):miss+=1;continue
   dt=td[k+1]-td[k] if k+1<nt else 8
   if dt<=0:continue
   ll+=max(0.,val-b)*dt;weighted_days+=dt
   if k+1<nt:observed_days+=dt
   else:tail+=dt
   if first<0:first=k
   last=k
  out[i,0]=final;out[i,1]=rec;out[i,2]=rend;out[i,3]=nxt;out[i,4]=reason;out[i,5]=miss;out[i,6]=weighted_days;out[i,7]=ll if weighted_days>0 else np.nan;out[i,8]=first;out[i,9]=last;out[i,10]=observed_days;out[i,11]=tail
  nextday=m[i,16];recday=m[i,10]
  for w in range(4):
   H=16 if w==0 else 32 if w==1 else 56 if w==2 else 120
   flag=0
   if td[-1]<td[t]+H:flag|=2
   if nextday>=0 and nextday<=H:flag|=4
   end=np.searchsorted(td,td[t]+H,side='right')
   for k in range(t,end):
    if not np.isfinite(a[k,c]):flag|=8;break
   out[i,20+w]=flag
   if flag:continue
   limit=min(H,int(recday)) if m[i,8]==0 and recday>=0 else H
   ll=0.;days=0
   for k in range(t+1,nt):
    day=td[k]-td[t]
    if day>=limit:break
    val=a[k,c]
    if not np.isfinite(val):continue
    dt=min(td[k+1]-td[k] if k+1<nt else 8,limit-day)
    if dt>0:ll+=max(0.,val-b)*dt;days+=dt
   out[i,16+w]=days
   if days>0:out[i,12+w]=ll
 return out

@njit(cache=True)
def leakage(ss,cc,td,zsync,pair,ac):
 # counts validSync,structuralFuture,actualFuture,safeSync,baselineZ,safeZ,commonA,safeA,baselineA,safeAmedian
 out=np.full((len(ss),10),np.nan,np.float32);nt=len(td)
 for i in range(len(ss)):
  s=ss[i];c=cc[i];p=np.searchsorted(td,td[s]-40);z=np.empty(s-p,np.float32);zs=np.empty(s-p,np.float32);aa=np.empty(s-p,np.float32);aas=np.empty(s-p,np.float32);n=0;ns=0;na=0;nas=0;nstruct=0;nactual=0
  for k in range(p,s):
   if not np.isfinite(zsync[k,c]):continue
   z[n]=zsync[k,c];n+=1;end=min(nt-1,k+3);struct=td[end]>=td[s];used=False
   if struct:
    nstruct+=1
    for j in range(s,end+1):
     if pair[j,c]:used=True;break
   if used:nactual+=1
   else:zs[ns]=zsync[k,c];ns+=1
   if np.isfinite(ac[k,c]):
    aa[na]=ac[k,c];na+=1
    if not used:aas[nas]=ac[k,c];nas+=1
  out[i,0]=n;out[i,1]=nstruct;out[i,2]=nactual;out[i,3]=ns;out[i,6]=na;out[i,7]=nas
  if n>=3:out[i,4]=np.median(z[:n])
  if ns>=3:out[i,5]=np.median(zs[:ns])
  if na>=3:out[i,8]=np.median(aa[:na])
  if nas>=3:out[i,9]=np.median(aas[:nas])
 return out

@njit(cache=True)
def baseline_peak(a,td,s,c):
 pre=np.searchsorted(td,td[s]-40);end=np.searchsorted(td,td[s]+56,side='right');buf=np.empty(s-pre,np.float32);n=0;npk=0;pk=-1;pv=-np.inf
 for k in range(pre,s):
  if np.isfinite(a[k,c]):buf[n]=a[k,c];n+=1
 for k in range(s,end):
  v=a[k,c]
  if np.isfinite(v):
   npk+=1
   if pk<0 or v>pv:pk=k;pv=v
 b=np.median(buf[:n]) if n>=3 else np.nan
 return b,pv-b if pk>=0 else np.nan,n,npk,pk

@njit(cache=True)
def controls(ss,cc,ids,td,phase,year,a,span,real,hcode):
 nt,nc=a.shape;eligible=np.zeros((nt,nc),np.bool_);pn=np.zeros((nt,nc),np.int8);kn=np.zeros((nt,nc),np.int8)
 phase_indices=np.empty((47,25),np.int16);phase_counts=np.zeros(47,np.int16)
 for k in range(nt):
  p=phase[k];phase_indices[p,phase_counts[p]]=k;phase_counts[p]+=1
 # Eligibility uses catalogue occupancy and finite-value counts only, never damage magnitude.
 for c in range(nc):
  cumulative=np.empty(nt+1,np.int32);cumulative[0]=0
  for k in range(nt):cumulative[k+1]=cumulative[k]+int(span[k,c])
  for s in range(nt):
   if td[s]-40<td[0] or td[s]+56>td[-1]:continue
   left=max(0,np.searchsorted(td,td[s]-40,side='right')-1);end=np.searchsorted(td,td[s]+56,side='right')
   if cumulative[end]>cumulative[left]:continue
   pre=np.searchsorted(td,td[s]-40);nn=0;nk=0
   for k in range(pre,s):nn+=int(np.isfinite(a[k,c]))
   for k in range(s,end):nk+=int(np.isfinite(a[k,c]))
   if nn>=3 and nk>=1:eligible[s,c]=True;pn[s,c]=nn;kn[s,c]=nk
 out=np.full((len(ss),10),np.nan,np.float32)
 for i in range(len(ss)):
  s=ss[i];c=cc[i];N=0;ix=np.empty(25,np.int16);priority=np.empty(25,np.uint64)
  if not np.isfinite(real[i,1]):out[i,0]=0;out[i,1]=0;continue
  for j in range(phase_counts[phase[s]]):
   k=phase_indices[phase[s],j]
   if year[k]==year[s] or not eligible[k,c]:continue
   if pn[k,c]!=real[i,4] or kn[k,c]!=real[i,5]:continue
   v=np.uint64(ids[i]+hcode*50000000)*np.uint64(0x9e3779b185ebca87)^np.uint64(k+20261117)
   v^=v>>np.uint64(30);v*=np.uint64(0xbf58476d1ce4e5b9);v^=v>>np.uint64(27);v*=np.uint64(0x94d049bb133111eb);v^=v>>np.uint64(31)
   ix[N]=k;priority[N]=v;N+=1
  order=np.argsort(priority[:N]);out[i,0]=N;out[i,1]=min(3,N)
  for j in range(min(3,N)):
   k=ix[order[j]];b,d,n,nk,pk=baseline_peak(a,td,k,c);out[i,2+j]=k;out[i,5+j]=d
  if N>0:
   n=min(3,N);out[i,8]=np.mean(out[i,5:5+n]);out[i,9]=real[i,1]-out[i,8]
 return out
