# -*- coding: utf-8 -*-
import json,time,gc
import numpy as np
import pandas as pd
from core import *

def a1():
 if (O/'qc/A1_READY.json').exists() and json.loads((O/'qc/A1_READY.json').read_text()).get('Version')==2:return
 out=[];bridges=[]
 for h in HAZ:
  #76blocks x counts/sums. Chunk processing keeps full catalogues out of memory.
  z=np.zeros((76,27));pixel_support={k:np.zeros(414*703,bool) for k in [1,2,4,8,10,13,24]};unknown=0;Nall=0;e=events(h);m=existing(h);lookup=np.full(int(e.event_num.max())+1,-1,np.int32);lookup[e.event_num.to_numpy(int)]=np.arange(len(e),dtype=np.int32);mismatch=0;maxerr=0.;ncompared=0
  for f in sorted((O/'outputs/event_details').glob(h+'_SyncLeak_rows_*.parquet')):
   d=pd.read_parquet(f,use_threads=False);Nall+=len(d);b=d.Block50.to_numpy();unknown+=int((b<0).sum());good=b>=0;d=d.loc[good];b=d.Block50.to_numpy(int);n=d.N_Pre_Sync.to_numpy();eligible=n>=3;actual=d.N_Used_Future_Pairs.to_numpy();struct=d.N_Structural_Future.to_numpy();safe=d.N_Safe_Sync.to_numpy();an=d.N_Pre_Acomp.to_numpy();asn=d.N_Safe_Acomp.to_numpy();dz=d.Safe_ZSync_Pre.to_numpy()-d.Original_ZSync_Pre.to_numpy();da=d.Safe_Acomp_Pre.to_numpy()-d.Original_Acomp_Pre.to_numpy();pZ=np.isfinite(dz);pA=np.isfinite(da)
   vv=[np.ones(len(d)),eligible,eligible&(actual>0),eligible&(struct>0),np.where(eligible,n,0),np.where(eligible,actual,0),eligible&(safe>=3),eligible&((n-struct)>=3),an>=3,(an>=3)&(asn>=3),pZ,np.nan_to_num(dz),np.nan_to_num(abs(dz)),pA,np.nan_to_num(da),np.nan_to_num(abs(da)),np.where(eligible,struct,0),np.where(pZ,d.Original_ZSync_Pre,0),np.where(pZ,d.Safe_ZSync_Pre,0),np.where(pA,d.Original_Acomp_Pre,0),np.where(pA,d.Safe_Acomp_Pre,0),np.where(an>=3,an,0),np.where(an>=3,asn,0)]
   vv += [eligible&(actual>0)&(safe>=3),(an>=3)&(actual>0),(an>=3)&(actual>0)&(asn>=3),eligible&(actual==0)]
   for j,v in enumerate(vv):z[:,j]+=np.bincount(b,weights=np.asarray(v,float),minlength=76)
   for k,mask in [(1,eligible),(2,eligible&(actual>0)),(4,eligible),(8,an>=3),(10,pZ),(13,pA),(24,(an>=3)&(actual>0))]:pixel_support[k][d.Pixel_ID.to_numpy(int)[mask]]=True
   ids=d.Event_Num.to_numpy(int);keep=ids<len(lookup);idx=np.full(len(ids),-1,np.int32);idx[keep]=lookup[ids[keep]];sel=idx>=0;a=d.Original_Acomp_Pre.to_numpy()[sel];ref=m[idx[sel],0];mismatch+=int((np.isfinite(a)!=np.isfinite(ref)).sum());ok=np.isfinite(a)&np.isfinite(ref);maxerr=max(maxerr,float(np.max(abs(a[ok]-ref[ok]))) if ok.any() else 0.);ncompared+=int(sel.sum())
  _,W=blockdata(h);point=z.sum(axis=0);boot=W@z
  def add(metric,num,den,unit='fraction'):
   value=point[num]/point[den] if point[den]>0 else np.nan;rep=np.divide(boot[:,num],boot[:,den],out=np.full(250,np.nan),where=boot[:,den]>0);out.append(dict(Hazard=h,Cohort='ALL_ORIGINAL_CATALOGUE_EVENTS',Metric=metric,N_Catalogue_Events=Nall,N_Unmapped_Spatial_Block=unknown,N_Numerator_Or_Sum=point[num],N_Denominator=point[den],Estimate=value,Unit=unit,N_pixels=int(pixel_support[den].sum()),N_50km_blocks=int((z[:,den]>0).sum()),**ci(rep)))
  for name,num,den in [('Events_With_Actual_Future_Use',2,1),('Events_With_Structural_Future_Window',3,1),('PreSync_Observations_Using_Future_Pairs',5,4),('PreSync_Observations_Window_Reaches_Future',16,4),('Safe_Sync_Baseline_Still_AtLeast3',6,1),('Strict_WindowEndsBeforeOnset_Still_AtLeast3',7,1),('Safe_Acomp_Baseline_Still_AtLeast3',9,8)]:add(name,num,den)
  for name,num,den in [('Affected_Events_SafeSync_RemainAtLeast3',23,2),('Affected_Events_SafeAcomp_RemainAtLeast3',25,24),('Valid_Sync_Baselines_Already_Unaffected',26,1)]:add(name,num,den)
  for name,num,den in [('Paired_SafeMinusOriginal_ZSync_Mean',11,10),('Paired_Absolute_ZSync_Baseline_Change',12,10),('Paired_SafeMinusOriginal_Acomp_Mean',14,13),('Paired_Absolute_Acomp_Baseline_Change',15,13),('Original_ZSync_OnSafePairedSubset',17,10),('Safe_ZSync_OnSafePairedSubset',18,10),('Original_Acomp_OnSafePairedSubset',19,13),('Safe_Acomp_OnSafePairedSubset',20,13)]:add(name,num,den,'dimensionless anomaly')
  bridges.append(dict(Hazard=h,N_Clean_Compared=ncompared,N_Missing_Mismatch=mismatch,MaxAbs=maxerr,Pass=mismatch==0 and maxerr<2e-6));assert mismatch==0 and maxerr<2e-6
  # Pair-safe subset is conditional; no threshold relaxation when most records fail3.
  print('A1',h,'catalogue',Nall,'affected',int(point[2]),'safeSync',int(point[6]),flush=True)
  del e,m;gc.collect()
 pd.DataFrame(out).to_csv(O/'outputs/A1_Sync_PreEvent_Leakage.csv',index=False,encoding='utf-8-sig');pd.DataFrame(bridges).to_csv(O/'qc/A1_Original_Baseline_Bridge.csv',index=False)
 (O/'qc/A1_READY.json').write_text(json.dumps(dict(Completed=True,Version=2,Affected_And_Unaffected_SafeSubsets_Separated=True)),encoding='utf8')

def main():
 assert (O/'qc/SYNC_READY.json').exists() and (O/'qc/OTHER_SYNC_READY.json').exists()
 if (O/'qc/A_READY.json').exists():return
 a1();rows=[];checks=[];start=time.time();estimate=ast_function(M5/'code/survival_m5.py','estimates')
 completed=set()
 if (O/'outputs/A2_TrailingSync_Baseline_Damage_Recovery.csv').exists():
  saved=pd.read_csv(O/'outputs/A2_TrailingSync_Baseline_Damage_Recovery.csv');completed={h for h,d in saved.groupby('Hazard') if d.Scope.nunique()==2};rows.extend(saved[saved.Hazard.isin(completed)].to_dict('records'))
 for h in HAZ:
  if h in completed:print('SKIP completed A2',h,flush=True);continue
  e=pd.read_parquet(MIN/'paired'/f'{h}_Fixed_Clean_Events.parquet',columns=['event_num','row','col','term_idx'],use_threads=False).astype({'event_num':'int32','row':'int16','col':'int16','term_idx':'int16'});orig=existing(h);alt=np.load(O/'qc/cache'/f'{h}_Trailing_Metrics.npy',mmap_mode='r');joint=np.load(O/'qc/cache'/f'{h}_CommonPhase_Metrics.npy',mmap_mode='r');g=groups(h);b,W=blockdata(h);pix=e.row.to_numpy(np.int32)*703+e.col.to_numpy(np.int32);N=len(e)
  for scope,a,v in [('NATIVE_SUPPORT_PAIRED_EVENT_IDS',orig,alt),('EXACT_COMMON_PHASE_SUPPORT',joint[:,0,:],joint[:,1,:])]:
   base=np.isfinite(a[:,1])&np.isfinite(v[:,1]);follow=base&(a[:,9]==1)&(v[:,9]==1)
   for k,label in enumerate(GROUPS):
    pop=(g==k) if k<4 else np.ones(N,bool);mm=base&pop;context=dict(Hazard=h,Severity_Group=label,Scope=scope,N_Source_Clean=int(pop.sum()),N_Common_BaselinePeak=int(mm.sum()),N_Common_Followup=int((follow&pop).sum()))
    for field,name,unit in [(0,'Baseline','Acomp_anomaly'),(1,'Raw_Damage','Acomp_anomaly'),(2,'Peak_Lag','days'),(12,'M5rule_Observed_Legacy','Acomp_anomaly_days'),(14,'Historical_Observed_Legacy','Acomp_anomaly_days')]:
     mask=mm&np.isfinite(a[:,field])&np.isfinite(v[:,field]);delta=np.asarray(v[:,field],float)-np.asarray(a[:,field],float);pt,rep=paired_mean(delta,mask,b,W);r=dict(context,Row_Type='PAIRED_METRIC',Metric=name,Original_Mean=float(np.mean(a[mask,field],dtype=float)) if mask.any() else np.nan,Trailing_Mean=float(np.mean(v[mask,field],dtype=float)) if mask.any() else np.nan,Original_Median=float(np.median(a[mask,field])) if mask.any() else np.nan,Trailing_Median=float(np.median(v[mask,field])) if mask.any() else np.nan,Statistic='Mean_Trailing_Minus_Original',Estimate=pt,Unit=unit,**support(mask,pix,b),**ci(rep));rows.append(r)
     if mask.any():
      qb=quantile_boot(delta[mask],b[mask],W,np.array([.5]))[:,0];rows.append(dict(context,Row_Type='PAIRED_METRIC',Metric=name,Statistic='Median_Paired_Difference',Estimate=float(np.median(delta[mask])),Unit=unit,**support(mask,pix,b),**ci(qb)))
    for condition,name in [((a[:,1]>0)&(v[:,1]<=0),'Positive_To_Nonpositive'),((a[:,1]<=0)&(v[:,1]>0),'Nonpositive_To_Positive')]:
     flag=mm&condition;pt,rep=ratio(flag,mm,b,W);rows.append(dict(context,Row_Type='DAMAGE_CLASS_CHANGE',Metric=name,N_Numerator=int(flag.sum()),Estimate=pt,Unit='fraction',**support(mm,pix,b),**ci(rep)))
    ff=follow&pop;sa=np.where(a[:,8]==0,0,np.where((a[:,8]==2)&(a[:,7]>=120),1,2));sv=np.where(v[:,8]==0,0,np.where((v[:,8]==2)&(v[:,7]>=120),1,2));states=['RECOVERED','OBSERVED_UNRECOVERED120','CENSORED_UNKNOWN120']
    for aa in range(3):
     for vv in range(3):
      flag=ff&(sa==aa)&(sv==vv);pt,rep=ratio(flag,ff,b,W);rows.append(dict(context,Row_Type='RECOVERY_STATUS_TRANSITION',Metric=states[aa]+'_TO_'+states[vv],N_Numerator=int(flag.sum()),Estimate=pt,Unit='fraction',**support(ff,pix,b),**ci(rep)))
    both=ff&(a[:,8]==0)&(v[:,8]==0);delta=v[:,10]-a[:,10];pt,rep=paired_mean(delta,both,b,W);rows.append(dict(context,Row_Type='RECOVERY_DATE_DIFFERENCE',Metric='First_Formal_Recovery_Date_TrailingMinusOriginal',Estimate=pt,Unit='actual_calendar_days',Limit='bothconfirmedrecoveredonly; noimputationforcensoredrecords',**support(both,pix,b),**ci(rep)))
   # Same event IDs for both indicators, same block draws, unchanged M5 estimator.
   counts=np.zeros((76,2,4,5,121),np.int64)
   for ind,z in enumerate([a,v]):
    code=(((b[follow]*4+g[follow])*5+z[follow,8].astype(int))*121+z[follow,7].astype(int));counts[:,ind]=np.bincount(code,minlength=76*4*5*121).reshape(76,4,5,121)
   counts=np.concatenate([counts,counts.sum(axis=2,keepdims=True)],axis=2);point=estimate(counts.sum(axis=0));boot=estimate((W@counts.reshape(76,-1)).reshape(250,2,5,5,121))
   for k,label in enumerate(GROUPS):
    ff=follow&((g==k) if k<4 else True)
    for metric in ['P120','RMPT120']:
     factor=100 if metric=='P120' else 1
     for j,name in enumerate(['ORIGINAL','TRAILING']):rows.append(dict(Hazard=h,Severity_Group=label,Scope=scope,Row_Type='PAIRED_COHORT_KM_GROUP',Metric=metric,Indicator=name,Estimate=float(point[metric][j,k])*factor,N_risk_120=int(point['N_risk_120'][j,k]),Unit='percentage_points_scale_probability' if factor==100 else 'days',**support(ff,pix,b),**ci(boot[metric][:,j,k]*factor)))
     d=(boot[metric][:,1,k]-boot[metric][:,0,k])*factor;rows.append(dict(Hazard=h,Severity_Group=label,Scope=scope,Row_Type='PAIRED_COHORT_KM_DIFFERENCE',Metric=metric,Indicator='TRAILING_MINUS_ORIGINAL',Estimate=float((point[metric][1,k]-point[metric][0,k])*factor),Unit='percentage_points' if factor==100 else 'days',**support(ff,pix,b),**ci(d)))
   # Store explicit dates/metrics with source event keys; no original results overwritten.
   dates=np.load(M5/'cache/dates.npy').astype('datetime64[D]')
   for first in range(0,N,200000):
    last=min(N,first+200000);idx=np.flatnonzero(base[first:last])+first
    if not len(idx):continue
    d=e.iloc[idx].copy();d['Hazard']=h;d['Scope']=scope;d['Block50']=b[idx]
    for name,z in [('Original',a),('Trailing',v)]:
     for field,label in [(0,'Baseline'),(1,'Damage'),(2,'PeakLag'),(7,'Followup_Days'),(8,'Outcome'),(9,'Followup_Valid'),(10,'Recovery_Day'),(11,'Confirmation_Day'),(12,'Legacy_Observed'),(13,'Legacy_Days'),(14,'Legacy_Historical')]:d[name+'_'+label]=z[idx,field]
     for field,label in [(10,'Recovery_Date'),(11,'Confirmation_Date')]:
      day=z[idx,field];arr=np.full(len(idx),'NaT',dtype='datetime64[D]');ok=(z[idx,8]==0)&(day>=0);arr[ok]=dates[e.term_idx.to_numpy()[idx[ok]]]+day[ok].astype('timedelta64[D]');d[name+'_'+label]=arr
    d.to_parquet(O/f'outputs/event_details/{h}_Trailing_{scope}_{first//200000:03d}.parquet',index=False,compression='zstd')
   print('A2',h,scope,'sec',round(time.time()-start,1),flush=True)
  pd.DataFrame(rows).to_csv(O/'outputs/A2_TrailingSync_Baseline_Damage_Recovery.csv',index=False,encoding='utf-8-sig');del e,orig,alt,joint,g,b,W,pix;gc.collect()
 (O/'qc/A_READY.json').write_text(json.dumps(dict(Completed_ThreeHazards=True,Original_M5_No_Positive_Damage_Gate=True,Censored_Unknown_Separate=True,Seconds=time.time()-start)),encoding='utf8')
if __name__=='__main__':main()
