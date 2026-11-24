# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
from numba import njit
R=Path(r'.');O=R/'EI_Reviewer1_Minor2_SingleIndicator';M1=R/'EI_Reviewer1_Major1_Revision';M5=R/'EI_Reviewer1_Major5_Revision'
F=['Baseline','Damage','Peak_Lag_Days','Peak_Value','Pre_N','Peak_N','Peak_Index','Duration','Outcome','Followup_Valid','Recovery_Day','Confirmation_Day','Legacy_Observed','Legacy_Days','Legacy_Historical','Historical_Days','Next_Onset_Day','Peak_After_Next']
@njit(cache=True)
def event_metrics(ss,tt,cc,td,cube,onsets,active):
 n=len(ss);nv=cube.shape[0];nt=len(td);out=np.full((n,nv,18),np.nan,np.float32)
 for i in range(n):
  s=ss[i];t=tt[i];c=cc[i];pre=np.searchsorted(td,td[s]-40);pend=np.searchsorted(td,td[s]+56,side='right');nxt=-1;oldend=np.searchsorted(td,td[s]+120,side='right');oldstop=oldend
  for k in range(t+1,nt):
   dd=td[k]-td[t]
   if dd>120:break
   if onsets[k,c]:nxt=dd;break
  for k in range(t+1,oldend):
   if active[k,c]:oldstop=k;break
  for v in range(nv):
   npre=0;buf=np.empty(s-pre,np.float32);npk=0;pk=-1;pv=-np.inf
   for k in range(pre,s):
    a=cube[v,k,c]
    if np.isfinite(a):buf[npre]=a;npre+=1
   for k in range(s,pend):
    a=cube[v,k,c]
    if np.isfinite(a):
     npk+=1
     if pk<0 or a>pv:pk=k;pv=a
   out[i,v,4]=npre;out[i,v,5]=npk;out[i,v,9]=0;out[i,v,16]=nxt
   if npre<3:continue
   z=np.sort(buf[:npre]);b=float(z[npre//2]) if npre%2 else float((z[npre//2-1]+z[npre//2])/2.)
   out[i,v,0]=b
   if pk<0:continue
   out[i,v,1]=pv-b;out[i,v,2]=td[pk]-td[s];out[i,v,3]=pv;out[i,v,6]=pk;out[i,v,17]=int(nxt>=0 and td[pk]-td[t]>=nxt)
   # M5PRIMARY: unchanged chronology, including no extra positive-damage gate.
   cday=min(120,td[-1]-td[t]);reason=2 if td[-1]-td[t]>=120 else 3
   if nxt>=0 and nxt<=cday:cday=nxt;reason=1
   first=-1;consec=0;rec=-1;confirm=-1;lastvalid=-1;lastscheduled=-1
   for k in range(t,nt):
    day=td[k]-td[t]
    if day>cday or (nxt>=0 and day>=nxt):break
    lastscheduled=day;a=cube[v,k,c]
    if not np.isfinite(a):consec=0;first=-1;continue
    lastvalid=day
    if k<=pk:continue
    if a<=b:
     if consec==0:first=day
     consec+=1
     if consec>=2:rec=first;confirm=day;break
    else:consec=0;first=-1
   dur=rec if rec>=0 else cday
   if rec>=0:reason=0
   elif lastvalid<lastscheduled:reason=4;dur=max(0,lastvalid)
   out[i,v,7]=dur;out[i,v,8]=reason;out[i,v,9]=int(lastvalid>=0);out[i,v,10]=rec;out[i,v,11]=confirm
   leg=0.;days=0
   for k in range(t+1,nt):
    day=td[k]-td[t]
    if day>=dur:break
    a=cube[v,k,c]
    if not np.isfinite(a):continue
    delta=min(td[k+1]-td[k] if k+1<nt else 8,dur-day)
    if delta>0:leg+=max(0.,a-b)*delta;days+=delta
   out[i,v,13]=days
   if days>0:out[i,v,12]=leg
   # Historical L5.3 legacy isolated for exact original replay / secondary bridge.
   first=-1;consec=0;oldrec=-1
   for k in range(pk+1,oldstop):
    a=cube[v,k,c]
    if not np.isfinite(a):consec=0;continue
    if a<=b:consec+=1
    else:consec=0
    if consec>=2:oldrec=k-1;break
   stop=oldrec if oldrec>=0 else oldstop;ll=0.;dd=0
   for k in range(t+1,stop):
    a=cube[v,k,c]
    if not np.isfinite(a):continue
    delta=td[k+1]-td[k] if k+1<nt else 8
    if delta>0:ll+=max(0.,a-b)*delta;dd+=delta
   out[i,v,15]=dd
   if dd>0:out[i,v,14]=ll
 return out
def read_block(ds,name,y0,y1):return np.ma.filled(ds[name][:,y0:y1,:],np.nan).astype(np.float32,copy=False).reshape(1142,-1)
