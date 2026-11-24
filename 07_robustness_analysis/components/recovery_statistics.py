# -*- coding: utf-8 -*-
import json,time,gc
import numpy as np
import pandas as pd
from audit_core import O
def main():
 assert (O/'qc/EXTRACTION_READY.json').exists(),'Wait for Acomp and chronology gate before inferential summaries.'
 W=np.load(O/'qc/Spatial_Bootstrap_250_Weights.npy');summary=[];patterns=[];cohorts=[];cirows=[];reps=[];audit=[]
 formal=pd.read_csv(O/'qc/Original_M5_Cohort_Counts.csv')
 for h in ['CDH','HEAT','HR']:
  d=pd.concat([pd.read_parquet(f) for f in sorted((O/'outputs/event_details').glob(h+'_Recovery_rows_*.parquet'))],ignore_index=True)
  assert d.event_num.is_unique and len(d)==int(formal[formal.Event==h].iloc[0].N_Formal_Confirmed_Recovered)
  assert (d.Recovery_Date>d.Peak_Date).all();assert (d.Confirmation_Index_Gap==1).all();assert (d.Confirmation_After_Term_Days<=120).all();assert ((d['next']<0)|(d['confirm']<d['next'])).all()
  assert np.max(abs(d.RMS_First_Rebuilt-d.Acomp_First))<2e-6 and np.max(abs(d.RMS_Confirm_Rebuilt-d.Acomp_Confirm))<2e-6
  good=d.Components_Complete==1;blocks=d.Block50.to_numpy(int)
  for group,mask in [('All_M5',np.ones(len(d),bool)),('High',d.Q_full.to_numpy()<90),('Extreme',d.Q_full.to_numpy()>=90)]:
   complete=mask&good.to_numpy();total=int(mask.sum());n=int(complete.sum());bn=np.bincount(blocks[complete],minlength=76);bd=np.bincount(blocks[mask],minlength=76)
   assert n>0
   cohorts.append(dict(Event=h,Severity_Group=group,N_Formal_Acomp_Recovered=total,N_Components_Complete=n,N_Excluded_Missing_Baseline_Or_Confirmation=total-n,Exclusion_Rate=(total-n)/total,N_Valid_Pixels=d.loc[complete,'Pixel_ID'].nunique(),N_50km_Blocks=np.count_nonzero(bn),N_Whole_Bootstrap_Blocks=76,Scope='original M5 High/Extreme, all confirmed recoveries; All_M5 does not includeLow/Moderate'))
   bden=W@bn;valid=bden>0;exclude_boot=np.divide(W@(bd-bn),W@bd,out=np.full(len(W),np.nan),where=(W@bd)>0)
   cirows.append(dict(Event=h,Severity_Group=group,Criterion='COMMON_SUPPORT',Metric='Missing_Component_Exclusion_Rate',Estimate=(total-n)/total,CI_lower=float(np.nanquantile(exclude_boot,.025)),CI_upper=float(np.nanquantile(exclude_boot,.975)),N_events=total,N_50km_Blocks=np.count_nonzero(bd),Bootstrap_N=250,Bootstrap_valid_n=np.isfinite(exclude_boot).sum(),Bootstrap_Block_km=50,Seed=20261159))
   mismatches={}
   for criterion,field in [('NEGATIVE_COMPONENT_BASELINE','Pattern_A'),('RAW_Z_MEDIAN_SENSITIVITY','Pattern_Z')]:
    pat=d[field].to_numpy()[complete].astype(int);bb=blocks[complete];tab=np.bincount(bb*8+pat,minlength=76*8).reshape(76,8);boot=W@tab
    assert tab.sum()==n
    for code in range(8):
     num=tab[:,code];value=num.sum()/n;v=np.divide(W@num,bden,out=np.full(250,np.nan),where=valid);ci=np.nanquantile(v,[.025,.975]);np_rec=bin(code).count('1')
     patterns.append(dict(Event=h,Severity_Group=group,Criterion=criterion,Pattern_Code=code,Sync_Recovered=int(bool(code&1)),uWUE_Recovered=int(bool(code&2)),NIRv_Recovered=int(bool(code&4)),N_Components_Recovered=np_rec,N_events=int(num.sum()),N_Denominator=n,Proportion=value,CI_lower=ci[0],CI_upper=ci[1],N_50km_Blocks=np.count_nonzero(bn),Bootstrap_N=250,Bootstrap_valid_n=int(valid.sum()),Unit='fraction of complete confirmedAcomp recoveries'))
    for count in [3,2,1,0]:
     codes=[v for v in range(8) if bin(v).count('1')==count];num=tab[:,codes].sum(axis=1);v=np.divide(W@num,bden,out=np.full(250,np.nan),where=valid);ci=np.nanquantile(v,[.025,.975]);summary.append(dict(Event=h,Severity_Group=group,Criterion=criterion,Category='Exactly_'+str(count)+'_Components_Recovered',N_events=int(num.sum()),N_Denominator=n,Proportion=num.sum()/n,CI_lower=ci[0],CI_upper=ci[1],N_50km_Blocks=int(np.count_nonzero(bn)),Bootstrap_N=250,Bootstrap_valid_n=int(valid.sum())))
    for name,codes in [('At_Least_One_Not_Synchronous',list(range(7))),('Sync_Not_Synchronous',[p for p in range(8) if not p&1]),('uWUE_Not_Synchronous',[p for p in range(8) if not p&2]),('NIRv_Not_Synchronous',[p for p in range(8) if not p&4])]:
     num=tab[:,codes].sum(axis=1);v=np.divide(W@num,bden,out=np.full(250,np.nan),where=valid);ci=np.nanquantile(v,[.025,.975]);r=dict(Event=h,Severity_Group=group,Criterion=criterion,Metric=name,Estimate=num.sum()/n,CI_lower=ci[0],CI_upper=ci[1],N_events=n,N_Numerator=int(num.sum()),N_50km_Blocks=int(np.count_nonzero(bn)),Bootstrap_N=250,Bootstrap_valid_n=int(valid.sum()),Bootstrap_Block_km=50,Seed=20261159,Interpretation='not synchronous at composite confirmation times, not neverrecovered duringfollowup')
     cirows.append(r)
     if name=='At_Least_One_Not_Synchronous':
      mismatches[criterion]=(r,v)
      for i,value in enumerate(v):reps.append(dict(Event=h,Severity_Group=group,Criterion=criterion,Replicate=i,At_Least_One_Not_Synchronous=value))
   a,aa=mismatches['NEGATIVE_COMPONENT_BASELINE'];b,bb=mismatches['RAW_Z_MEDIAN_SENSITIVITY'];dif=bb-aa;lo,hi=np.nanquantile(dif,[.025,.975]);cirows.append(dict(Event=h,Severity_Group=group,Criterion='PAIRED_Z_MINUS_NEGATIVE_RULE',Metric='Difference_AtLeastOneNotSynchronous',Estimate=b['Estimate']-a['Estimate'],CI_lower=lo,CI_upper=hi,N_events=n,N_50km_Blocks=int(np.count_nonzero(bn)),Bootstrap_N=250,Bootstrap_valid_n=int(np.isfinite(dif).sum()),Bootstrap_Block_km=50,Seed=20261159))
  # An audit sample includes actual year-end5/6day pairs, never substitutes8days.
  sample=pd.concat([d.head(12),d[d.Actual_Confirmation_Gap_Days!=8].head(20)]).drop_duplicates('event_num');audit.append(sample)
  print('B STATS',h,'NRecovered',len(d),'Complete',int(good.sum()),'PatternA counts',d.Pattern_A.value_counts().to_dict(),flush=True)
  del d;gc.collect()
 for data,name in [(cohorts,'B1_CompositeRecovery_Cohort.csv'),(summary,'B2_ComponentRecovery_Concordance.csv'),(patterns,'B3_Recovery_Patterns.csv'),(cirows,'B4_SpatialBootstrap_CI.csv')]:pd.DataFrame(data).to_csv(O/'outputs'/name,index=False,encoding='utf-8-sig')
 pd.DataFrame(reps).to_csv(O/'qc/Recovery_Bootstrap_Replicates.csv.gz',index=False,compression='gzip');pd.concat(audit).to_csv(O/'qc/Recovery_Event_Audit_Sample.csv',index=False,encoding='utf-8-sig')
 (O/'qc/B_READY.json').write_text(json.dumps(dict(All_3_Hazards=True,No_New_KM_AJ_Fit=True,Uncertainty='exact paired whole50km block binomial proportions on fixedM5 recovery cohort'),indent=2),encoding='utf8')
if __name__=='__main__':main()
