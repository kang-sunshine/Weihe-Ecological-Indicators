from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
O=Path(r'.\EI_Reviewer1_Major3_Revision')
BLOCKS=None

def stats(x):
 x=np.asarray(x);x=x[np.isfinite(x)]
 if not len(x):return dict(N_Valid=0,Mean=np.nan,Median=np.nan,SD=np.nan,Q25=np.nan,Q75=np.nan)
 return dict(N_Valid=len(x),Mean=float(x.mean()),Median=float(np.median(x)),SD=float(np.std(x)),Q25=float(np.quantile(x,.25)),Q75=float(np.quantile(x,.75)))

def paired(a,b,mask):
 good=mask&np.isfinite(a)&np.isfinite(b);aa=a[good];bb=b[good];st=stats(bb-aa)
 detail={}
 if BLOCKS is not None and len(aa):
  sums=np.bincount(BLOCKS[good],weights=bb-aa);counts=np.bincount(BLOCKS[good],minlength=len(sums));bm=sums[counts>0]/counts[counts>0]
  detail=dict(N_Spatial_Blocks=len(bm),Mean_Of_Block_Deltas=float(bm.mean()),SE_Of_Block_Deltas=float(bm.std(ddof=1)/np.sqrt(len(bm))))
 return dict(N_Common=int(good.sum()),A_Mean=float(aa.mean()) if len(aa) else np.nan,B_Mean=float(bb.mean()) if len(bb) else np.nan,A_Median=float(np.median(aa)) if len(aa) else np.nan,B_Median=float(np.median(bb)) if len(bb) else np.nan,Delta_Mean_B_minus_A=st['Mean'],Delta_Median_B_minus_A=st['Median'],Delta_Q25=st['Q25'],Delta_Q75=st['Q75'],Pearson=float(np.corrcoef(aa,bb)[0,1]) if len(aa)>2 and np.std(aa)>0 and np.std(bb)>0 else np.nan,Fraction_B_Greater=float(np.mean(bb>aa)) if len(aa) else np.nan,Fraction_AbsDelta_ge8=float(np.mean(np.abs(bb-aa)>=8)) if len(aa) else np.nan,**detail)

def km(d,reason):
 d=np.asarray(d,dtype=int);r=np.asarray(reason,dtype=int);s=1.;area=0.;last=0;endpoint=True
 for t in np.unique(d):
  area+=s*(t-last);risk=int((d>=t).sum());recs=int(((d==t)&(r==0)).sum());s*=1-recs/risk;last=t
  if risk==int((d==t).sum()) and t<120 and s>0:endpoint=False
 area+=s*(120-last)
 return (1-s if endpoint else np.nan,area if endpoint else np.nan,endpoint,int((d>=120).sum()))

def main():
 global BLOCKS
 meta=json.loads((O/'cache/EXTRACTION_READY.json').read_text());V=meta['Variants'];fields=meta['Fields'];m=np.load(O/'cache/Event_Metrics.npy',mmap_mode='r');ev=pd.read_parquet(O/'paired/CDH_Original_Clean_Cohort.parquet');N=len(ev);allmask=np.ones(N,bool)
 BLOCKS=((ev.row.to_numpy().astype(int)//20)*36+(ev.col.to_numpy().astype(int)//20)).astype(int)
 orig=ev.valid_resistance_sample.eq(1).to_numpy()&ev.Acomp_damage.gt(0).to_numpy()&np.isfinite(m[:,0,1]);sourceerr=np.abs(m[:,0,1]-ev.Acomp_damage.to_numpy());pp=np.isfinite(sourceerr)
 validation=dict(N_Clean=N,N_Formal=int(orig.sum()),Acomp_Damage_Reproduction_MaxAbs=float(sourceerr[pp].max()),Acomp_Baseline_Reproduction_MaxAbs=float(np.nanmax(np.abs(m[:,0,0]-ev.Acomp_pre))),Acomp_Lag_Reproduction_MaxAbs=float(np.nanmax(np.abs(m[:,0,2]-ev.Acomp_peak_lag_days))))
 assert validation['Acomp_Damage_Reproduction_MaxAbs']<1e-6 and validation['Acomp_Lag_Reproduction_MaxAbs']==0
 sync=[]
 strict=np.load(O/'cache/Strict_Sync_Event_Metrics.npy',mmap_mode='r') if (O/'cache/STRICT_SYNC_READY.json').exists() else None
 for support,mat,pairs in [('INDICATOR_NATIVE_SUPPORT',m,[(5,6,'Sync'),(0,4,'Acomp')])]+([('EXACT_COMMON_PHASE_SUPPORT',strict,[(2,3,'Sync'),(0,1,'Acomp')])] if strict is not None else []):
  for a,b,name in pairs:
   valid=np.isfinite(mat[:,a,1])&np.isfinite(mat[:,b,1]);base=dict(Status='COMPLETED',Indicator=name,Support=support,N_Input_Events=N,N_A_Valid=int(np.isfinite(mat[:,a,1]).sum()),N_B_Valid=int(np.isfinite(mat[:,b,1]).sum()),N_Common_Events=int(valid.sum()),N_A_Only=int((np.isfinite(mat[:,a,1])&~np.isfinite(mat[:,b,1])).sum()),N_B_Only=int((~np.isfinite(mat[:,a,1])&np.isfinite(mat[:,b,1])).sum()),Selection='same CDH clean-pre IDs; pre40d median >=3; peak0..56d; source Sync pairs>=5; phase baseline2000-2020 n>=10 std>1e-6',Limit='retrospective fixed seasonal climatology; trailing input uses no future raw observations; centered A retains future observations by definition')
   for f,unit in [(0,'baseline_badness_Z_or_RMS'),(1,'damage_Z_or_RMS'),(2,'days'),(9,'observations')]:sync.append(dict(base,Row_Type='PAIRED_RESPONSE',Metric=fields[f],Units=unit,**paired(mat[:,a,f],mat[:,b,f],valid)))
 # Negative baseline is badness=-Z for signed metrics; provide actual Sync Z baseline too.
 for row in sync:
  if row.get('Metric')!='Peak_Lag_Days':row.pop('Fraction_AbsDelta_ge8',None)
  if row['Indicator']=='Sync' and row.get('Metric')=='Baseline':row['Actual_Sync_Z_A_Mean']=-row['A_Mean'];row['Actual_Sync_Z_B_Mean']=-row['B_Mean'];row['Actual_Sync_Z_Delta_Mean']=-row['Delta_Mean_B_minus_A']
 if strict is not None:
  order=np.isfinite(strict[:,2,2])&np.isfinite(strict[:,3,2])&np.isfinite(m[:,7,2])
  for j,name in [(2,'centered'),(3,'trailing')]:
   a=strict[order,j,2];b=m[order,7,2];sync.append(dict(Row_Type='SYNC_UWUE_PEAK_ORDER',Status='COMPLETED',Indicator='Sync_'+name,Support='EXACT_COMMON_PHASE_SYNC; unchanged_uWUE',N_Common=int(order.sum()),Fraction_Sync_Before_uWUE=float(np.mean(a<b)),Fraction_Sync_Same_uWUE=float(np.mean(a==b)),Fraction_Sync_After_uWUE=float(np.mean(a>b)),Limit='event peak order within fixed0..56d window; not proof of physiological causality'))
 pd.DataFrame(sync).to_csv(O/'M3_Sync_Window_Comparison.csv',index=False,encoding='utf-8-sig')
 uw=[];valid=np.isfinite(m[:,7,1])&np.isfinite(m[:,8,1])
 for f,unit in [(0,'negative_Z'),(1,'Z'),(2,'days')]:uw.append(dict(Row_Type='PAIRED_RESPONSE',Status='COMPLETED',A='uWUE',B='GPP_ET',Metric=fields[f],Units=unit,N_Input_Events=N,**paired(m[:,7,f],m[:,8,f],valid),Limit='same accepted original uWUE support incl VPD filter and0..50 QC; each metric has own phase baseline; direct VPD multiplier removed, selection dependence remains'))
 tr=np.load(O/'paired/uWUE_Trajectory_Aggregates.npy')
 for j,off in enumerate(range(-5,16)):
  for v,var in enumerate(['uWUE','GPP_ET']):
   n,s,ss,days=tr[v,j];uw.append(dict(Row_Type='EVENT_TRAJECTORY',Status='COMPLETED',Indicator=var,Offset_Phases=off,Mean_Actual_Lag_Days=days/n if n else np.nan,N_Valid=int(n),Mean=s/n if n else np.nan,SD=np.sqrt(max(0,ss/n-(s/n)**2)) if n else np.nan,Units='Z_change_from_own_pre_median',Limit='same paired finite event observations per offset; descriptive, not next-event-isolated trajectory'))
 lg=np.load(O/'cache/uWUE_Log_Components.npy',mmap_mode='r');good=np.isfinite(lg).all(axis=1)&valid
 for j,col in enumerate(meta['Log_Columns']):uw.append(dict(Row_Type='ALGEBRAIC_LOG_CHANGE',Status='COMPLETED',Indicator=col,Units='natural_log',**stats(lg[good,j]),Mean_Absolute=float(np.mean(np.abs(lg[good,j]))),Fraction_Aligned_With_Total=float(np.mean(lg[good,j]*lg[good,3]>0)),Limit='positive raw values at original uWUE Z-damage peak; reference mean log across same accepted pre40d observations; no physiological/causal decomposition'))
 uw.append(dict(Row_Type='ALGEBRAIC_IDENTITY',Status='COMPLETED',N_Valid=int(good.sum()),MaxAbs_Residual=float(np.max(np.abs(lg[good,0]+lg[good,1]+lg[good,2]-lg[good,3]))),Limit='stored total constructed by exact log identity, not an independent model validation'))
 for row in uw:
  if row.get('Metric')!='Peak_Lag_Days':row.pop('Fraction_AbsDelta_ge8',None)
 pd.DataFrame(uw).to_csv(O/'M3_uWUE_VPD_Comparison.csv',index=False,encoding='utf-8-sig')
 ab=[];indices=[0,1,2,3,9,10];common=orig.copy()
 for k in indices:common&=np.isfinite(m[:,k,1])
 for k in indices:
  for f,unit in [(0,'RMS_or_capped_negative_Z'),(1,'RMS_or_capped_negative_Z'),(2,'days')]:ab.append(dict(Row_Type='COMMON_EVENT_RESPONSE',Status='COMPLETED',Variant=V[k],Metric=fields[f],Units=unit,N_Input_Events=N,N_Original_Formal=int(orig.sum()),N_Common_Events=int(common.sum()),**stats(m[common,k,f]),Limit='same event IDs; own pre40d median and0..56d peak; NIRv-only/GPP-only capped negative Z; own component temporal validity retained'))
  ab.append(dict(Row_Type='PAIRED_DAMAGE',Status='COMPLETED',Variant=V[k],Reference='Acomp_original',**paired(m[:,0,1],m[:,k,1],common),Limit='same events; all standardized components but differing RMS aggregation; correlation is not independent physiological information'))
  ab.append(dict(Row_Type='DAMAGE_DISCORDANCE',Status='COMPLETED',Variant=V[k],N_Common_Events=int(common.sum()),Positive_Fraction=float(np.mean(m[common,k,1]>0)),Acomp_Positive_Variant_Nonpositive_Fraction=float(np.mean(m[common,k,1]<=0)),Limit='original formal cohort is selected for Acomp_damage>0; not an unconditional population fraction'))
 # One shared damaging and observable cohort across all four primary ablations.
 recovery=common.copy()
 for k in [0,1,2,3]:recovery&=(m[:,k,1]>0)&(m[:,k,7]==1)&np.isfinite(m[:,k,5])
 legacycommon=recovery.copy()
 for k in [0,1,2,3]:legacycommon&=np.isfinite(m[:,k,4])
 for k in [0,1,2,3]:
  d=m[recovery,k,5];r=m[recovery,k,6];P,R,estim,nrisk=km(d,r)
  ab.append(dict(Row_Type='COMMON_RECOVERY',Status='COMPLETED',Variant=V[k],N_Common_Events=int(recovery.sum()),P120_KM_Net=P,RMPT120=R,Endpoint_Estimable=estim,N_Risk120=nrisk,Next_Event_Interruption_Fraction=float(np.mean(r==1)),Incomplete_Fraction=float(np.mean(r==4)),Unrecovered_Fraction=float(np.mean(r!=0)),Limit='same positive-damage and observable event intersection; M5 primary postpeak/termination-origin/catalogue-next/2-confirmation/120d rules; no bootstrap CI; KM net probability, not AJ CIF'))
  ab.append(dict(Row_Type='COMMON_OBSERVED_LEGACY',Status='COMPLETED',Variant=V[k],Units='badness_anomaly_days',N_Common_Events=int(legacycommon.sum()),N_Recovery_Cohort=int(recovery.sum()),**stats(m[legacycommon,k,4]),Limit='same finite-legacy event intersection across4 variants; observed positive excess integral post termination; stops at recovery/interruption/horizon; missing intervals omitted; not a censor-adjusted total burden, not historical L5.3 legacy'))
 diag=np.load(O/'cache/Event_Diagnostics.npy',mmap_mode='r');ab.append(dict(Row_Type='COMPONENT_OPPOSITION_AT_ACOMP_PEAK',Status='COMPLETED',N_Common_Events=int(common.sum()),Fraction=float(np.mean(diag[common,1]==1)),Limit='Sync/uWUE signed Z with NIRv capped negative Z at original peak; at least one deterioration and one improvement vs own baseline; not measured at recovery'))
 ab.append(dict(Row_Type='COMPONENT_OPPOSITION_AT_RECOVERY',Status='NOT COMPLETED',Limit='RMS threshold can permit one component below its own baseline while others improve; per-component state at recovery not separately extracted; no empirical fraction supplied'))
 for row in ab:row.pop('Fraction_AbsDelta_ge8',None)
 pd.DataFrame(ab).to_csv(O/'M3_Composite_Ablation.csv',index=False,encoding='utf-8-sig')
 (O/'audit/Event_Metric_Reproduction.json').write_text(json.dumps(validation,indent=2),encoding='utf8')
 print('M3 SUMMARIES READY',json.dumps(validation),flush=True)
if __name__=='__main__':main()
