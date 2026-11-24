# -*- coding: utf-8 -*-
import json,time,gc
import numpy as np
import pandas as pd
from numba import njit
from core import *
from statistics_legacy import describe,PROBS,LABELS

@njit(cache=True)
def weighted_kth(pool,cumulative,start,end,weights,k,upper):
 lo=np.uint64(0);hi=upper
 while lo<hi:
  mid=lo+(hi-lo)//np.uint64(2);total=0
  for b in range(len(start)):
   if weights[b]==0 or start[b]==end[b]:continue
   a=start[b];z=end[b]
   while a<z:
    j=(a+z)//2
    if pool[j]<=mid:a=j+1
    else:z=j
   if a>start[b]:total+=cumulative[a-1]*weights[b]
  if total>=k:hi=mid
  else:lo=mid+np.uint64(1)
 return lo
@njit(cache=True)
def weighted_mixture(pool,cumulative,start,end,W,probs):
 codes=np.empty((len(W),len(probs),2),np.uint64);frac=np.empty((len(W),len(probs)));upper=np.uint64(0)
 for b in range(len(start)):
  if end[b]>start[b]:upper=max(upper,pool[end[b]-1])
 for r in range(len(W)):
  n=0
  for b in range(len(start)):
   if end[b]>start[b]:n+=cumulative[end[b]-1]*W[r,b]
  for j,p in enumerate(probs):
   if n<=0:codes[r,j,0]=0;codes[r,j,1]=0;frac[r,j]=np.nan;continue
   pos=(n-1)*p;lo=int(np.floor(pos));hi=int(np.ceil(pos));frac[r,j]=pos-lo
   codes[r,j,0]=weighted_kth(pool,cumulative,start,end,W[r],lo+1,upper);codes[r,j,1]=weighted_kth(pool,cumulative,start,end,W[r],hi+1,upper)
 return codes,frac

def control_quantiles(v,b,base,W,probs):
 shift=min(0.,float(np.min(v)));chunks=[];cum=[];counts=np.bincount(b,minlength=76)
 for k in range(76):
  m=b==k;vv=v[m].astype(float)-shift;weights=base[m];order=np.argsort(vv);chunks.append(vv[order]);cum.append(np.cumsum(weights[order],dtype=np.int64))
 pool=np.concatenate(chunks).view(np.uint64);cumulative=np.concatenate(cum);start=np.r_[0,np.cumsum(counts)[:-1]];end=np.cumsum(counts);codes,f=weighted_mixture(pool,cumulative,start,end,W.astype(np.int32),np.asarray(probs));values=codes.view(np.float64);return values[:,:,0]*(1-f)+values[:,:,1]*f+shift

def main():
 assert (O/'qc/EXTRACTION_READY.json').exists()
 if (O/'qc/B_READY.json').exists():return
 from statistics_sync import a1
 a1() # complete affected-versus-unaffected safe-baseline denominators before B allocations
 start=time.time();support_rows=[];stats=[];gates=[]
 completed=set()
 if (O/'outputs/B1_Matched_NonEvent_Control.csv').exists() and (O/'outputs/B2_PeakDamage_Background_Comparison.csv').exists():
  saved1=pd.read_csv(O/'outputs/B1_Matched_NonEvent_Control.csv');saved2=pd.read_csv(O/'outputs/B2_PeakDamage_Background_Comparison.csv');completed=set(saved1.Hazard)&set(saved2.Hazard);support_rows.extend(saved1[saved1.Hazard.isin(completed)].to_dict('records'));stats.extend(saved2[saved2.Hazard.isin(completed)].to_dict('records'))
 # Exact event-balanced control quantile verification, including negative raw differences.
 v=np.array([-1.,0.,.2,.4,1.]);b=np.array([0,0,1,1,1]);base=np.array([2,2,3,6,6]);W=np.ones((3,76),np.int32);W[1,0]=2;W[2,1]=0;got=control_quantiles(v,b,base,W,np.array([.05,.5,.9,.95]));err=max(np.max(abs(got[i]-np.quantile(np.repeat(v,base*W[i,b]),[.05,.5,.9,.95]))) for i in range(3));assert err<1e-12
 (O/'qc/Control_Quantile_Engine_Verification.json').write_text(json.dumps(dict(MaxAbs=err,Each_Matched_Event_Total_Integer_Weight=6)),encoding='utf8')
 for h in HAZ:
  if h in completed:print('SKIP completed B',h,flush=True);continue
  e=pd.read_parquet(MIN/'paired'/f'{h}_Fixed_Clean_Events.parquet',columns=['event_num','row','col','onset_idx'],use_threads=False);e=e.astype({'event_num':'int32','row':'int16','col':'int16','onset_idx':'int16'});m=existing(h);q=np.load(MIN/'cache'/h/'Full_ECDF_Q.npy',mmap_mode='r');g=groups(h);b,W=blockdata(h);c=np.load(O/'qc/cache'/f'{h}_Controls.npy',mmap_mode='r');N=len(e);pix=e.row.to_numpy(np.int32)*703+e.col.to_numpy(np.int32);valid=np.isfinite(m[:,1]);matched=valid&(c[:,1]>0);real=m[:,1];delta=c[:,9];ctrlmean=c[:,8];counts=np.nan_to_num(c[:,1]).astype(np.int8)
  for k,label in enumerate(GROUPS):
   pop=(g==k) if k<4 else np.ones(N,bool);basepop=pop&valid;mask=pop&matched;pt,v=ratio(mask,basepop,b,W);pairs=int(counts[mask].sum())
   support_rows.append(dict(Hazard=h,Severity_Group=label,N_Clean_Events=int(pop.sum()),N_Valid_Raw_Damage=int(basepop.sum()),N_Matched_Events=int(mask.sum()),N_Matched_Pixels=int(np.unique(pix[mask]).size),N_Matched_Blocks=int(np.unique(b[mask]).size),N_Unmatched_Valid=int((basepop&~matched).sum()),N_Control_Links=pairs,N_One_Control=int((mask&(counts==1)).sum()),N_Two_Controls=int((mask&(counts==2)).sum()),N_Three_Controls=int((mask&(counts==3)).sum()),Estimate=pt,Metric='Matched_Coverage',Unit='fraction',**support(basepop,pix,b),**ci(v),Matching='exactcalendarphase,samepixel,differentyear,emptymergedcatalogpre40..post56,samefinitepre/peakcounts'))
   if not mask.any():stats.append(dict(Hazard=h,Severity_Group=label,Metric='ALL',Status='NOT ESTIMABLE',Missing_Reason='no strictly matched controls'));continue
   ctx=dict(Hazard=h,Severity_Group=label,Row_Type='PAIRED_EVENT_DISTRIBUTION',Cohort='same strictlymatched real eventIDs',Unit='Acomp_raw_peak_minus_pre40median')
   for field,name in [(real,'Real_Raw_Damage'),(ctrlmean,'Control_PerEvent_Mean_Raw_Damage'),(delta,'Real_Minus_ControlMean')]:describe(field,mask,pix,b,W,dict(ctx,Metric=name),stats)
   controlpositive=np.zeros(N,np.float32)
   for j in range(3):controlpositive+=((j<counts)&(c[:,5+j]>0)).astype(np.float32)
   controlpositive=np.divide(controlpositive,counts,out=np.full(N,np.nan,np.float32),where=counts>0)
   for name,x in [('Real_D_Positive',(real>0).astype(float)),('MatchedControl_D_Positive_EventBalanced',controlpositive),('Real_Exceeds_ControlMean',(delta>0).astype(float))]:
    p,rep=paired_mean(x,mask,b,W);stats.append(dict(ctx,Metric=name,Statistic='Proportion',Estimate=p,Unit='fraction',**support(mask,pix,b),**ci(rep)))
   # Pooled background distribution: each real event contributes total weight6,
   # split2/3/6 across its3/2/1 controls. Recompute thresholds within every block draw.
   vv=[];bb=[];weight=[]
   for j in range(3):
    mm=mask&(counts>j);vv.append(np.asarray(c[mm,5+j],float));bb.append(np.asarray(b[mm],int));weight.append((6//counts[mm]).astype(np.int16))
   vv=np.concatenate(vv);bb=np.concatenate(bb);weight=np.concatenate(weight);allW=np.vstack([np.ones(76,np.int32),W]);quant=control_quantiles(vv,bb,weight,allW,np.r_[PROBS,.9]);cutoffs=quant[0];bootcuts=quant[1:]
   for j,name in enumerate(LABELS+['Q90']):stats.append(dict(ctx,Metric='Matched_Control_Raw_Damage_EventBalanced',Statistic=name,Estimate=cutoffs[j],N_Control_Links=len(vv),**support(mask,pix,b),**ci(bootcuts[:,j])))
   true_by_block=[np.sort(np.asarray(real[mask&(b==z)],float)) for z in range(76)];den=W@np.bincount(b[mask],minlength=76)
   for percentile,j in [(90,5),(95,4)]:
    point=float(np.mean(real[mask]>cutoffs[j]));num=np.zeros(len(W))
    for z,x in enumerate(true_by_block):
     above=len(x)-np.searchsorted(x,bootcuts[:,j],side='right');num+=W[:,z]*above
    rep=np.divide(num,den,out=np.full(len(W),np.nan),where=den>0);stats.append(dict(ctx,Metric='Real_Exceeds_MatchedBackground_P'+str(percentile),Statistic='Proportion',Estimate=point,Unit='fraction',Background_Cutoff=cutoffs[j],Cutoff_Estimator='eventbalancedpooledmatchedbackground, recomputedineachwholeblockreplicate',**support(mask,pix,b),**ci(rep)))
   # Event-specific <=3-control upper quantiles: secondary exploratory diagnostic.
   for percentile in [90,95]:
    subset=np.sort(np.asarray(c[mask,5:8],float),axis=1);kk=counts[mask];pos=(kk-1)*(percentile/100);lo=np.floor(pos).astype(int);hi=np.ceil(pos).astype(int);ii=np.arange(len(pos));cut=subset[ii,lo]+(subset[ii,hi]-subset[ii,lo])*(pos-lo);bb2=b[mask];num=np.bincount(bb2,weights=(real[mask]>cut).astype(float),minlength=76);nn=np.bincount(bb2,minlength=76);rep=(W@num)/(W@nn);stats.append(dict(ctx,Metric='Real_Exceeds_OwnMax3Controls_P'+str(percentile),Statistic='Proportion',Estimate=num.sum()/nn.sum(),Unit='fraction',Limit='secondaryexploratory; only1to3controls sohighquantilesunstable; notfalsepositiverate',**support(mask,pix,b),**ci(rep)))
   print('B STATS',h,label,'matched',int(mask.sum()),'sec',round(time.time()-start,1),flush=True)
  # Matched links stay auditable; no multimillion-row CSV export.
  dates=np.load(M5/'cache/dates.npy').astype('datetime64[D]')
  for first in range(0,N,200000):
   last=min(N,first+200000);indices=np.flatnonzero(matched[first:last])+first
   if not len(indices):continue
   d=e.iloc[indices].copy();d['Hazard']=h;d['Block50']=b[indices];d['Q_Full']=q[indices];d['Real_D_Raw']=real[indices];d['Real_Pre_N']=m[indices,4];d['Real_Peak_N']=m[indices,5];d['N_Eligible_Controls']=c[indices,0];d['N_Selected_Controls']=counts[indices];d['ControlMean_D_Raw']=c[indices,8];d['Real_Minus_ControlMean']=c[indices,9]
   for j in range(3):
    idx=c[indices,2+j];date=np.full(len(indices),'NaT',dtype='datetime64[D]');ok=np.isfinite(idx);date[ok]=dates[idx[ok].astype(int)];d[f'Control_{j+1}_Date']=date;d[f'Control_{j+1}_D_Raw']=c[indices,5+j]
   d.to_parquet(O/f'outputs/event_details/{h}_MatchedControls_{first//200000:03d}.parquet',index=False,compression='zstd')
  pd.DataFrame(support_rows).to_csv(O/'outputs/B1_Matched_NonEvent_Control.csv',index=False,encoding='utf-8-sig');pd.DataFrame(stats).to_csv(O/'outputs/B2_PeakDamage_Background_Comparison.csv',index=False,encoding='utf-8-sig')
  print('B COMPLETE',h,'sec',round(time.time()-start,1),flush=True);del e,m,c,g,b,W,q,pix;gc.collect()
 (O/'qc/B_READY.json').write_text(json.dumps(dict(Complete_ThreeHazards=True,Matching_Not_Relaxed=True,EventBalanced_ControlQuantiles=True,No_Official_Damage_Redefinition=True,Seconds=time.time()-start)),encoding='utf8')
if __name__=='__main__':main()
