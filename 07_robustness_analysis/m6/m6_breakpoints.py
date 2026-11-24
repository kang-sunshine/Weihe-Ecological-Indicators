# -*- coding: utf-8 -*-
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from pyproj import Transformer
O=Path(r'.\EI_Reviewer1_Major6_Revision');M1=Path(r'.\EI_Reviewer1_Major1_Revision');M4=Path(r'.\EI_Reviewer1_Major4_Revision');D=Path(r'.\Weihe database V3')
def fit(x,y,w,c=None):
 X=np.column_stack([np.ones(len(x)),x]+([] if c is None else [np.maximum(0,x-c)]));beta=np.linalg.lstsq(X*np.sqrt(w)[:,None],y*np.sqrt(w),rcond=None)[0];rss=np.sum(w*(y-X@beta)**2);return beta,float(len(x)*np.log(max(rss,1e-300)/len(x))+2*(2 if c is None else 4))
def historical_joint_sample(q,eventnum,rng):
 q=q[np.argsort(eventnum[q],kind='stable')];selected=np.empty(0,np.int64)
 for chunk in range(5):
  current=q[((eventnum[q]-1)//1000000)==chunk]
  if len(current)>20000:current=current[rng.choice(len(current),20000,replace=False)]
  selected=np.concatenate([selected,current])
  if len(selected)>20000:selected=selected[rng.choice(len(selected),20000,replace=False)]
 return selected
def main():
 assert (O/'cache/EXTRACTION_READY.json').exists();ev=pd.read_parquet(O/'paired/CDH_Fixed_Original_Clean_Events.parquet');met=np.load(O/'cache/CDH_Event_Metrics.npy',mmap_mode='r');s=ev.climate_cumulative_severity.to_numpy();year=ev.Onset_Year.to_numpy();edges=np.load(M4/'paired/CDH_Original_Historical_Bin_Edges.npy');bn=np.clip(np.searchsorted(edges,s,side='right')-1,0,len(edges)-2);sourcecand=pd.read_csv(M1/'R1M1_Breakpoint_Candidate_AIC.csv');sourcecand=sourcecand[(sourcecand.Event=='CDH')&(sourcecand.Analysis=='SAVED_BIN_historical')];saved=pd.read_csv(D/'L5_EventEcoFunction/L5_4_ThresholdAnalysis/L5_4A_Severity_Response_Thresholds.csv');grid=json.loads((M4/'audit/Grid_Metadata.json').read_text())['Axes'];lon=np.array(grid['x'])[ev.col.to_numpy(int)];lat=np.array(grid['y'])[ev.row.to_numpy(int)];proj=json.loads((M4/'code/configuration.json').read_text())['Projection'];x,y=Transformer.from_crs(4326,proj,always_xy=True).transform(lon,lat);b50=np.floor(x/50000).astype(int)*10000+np.floor(y/50000).astype(int);b100=np.floor(x/100000).astype(int)*10000+np.floor(y/100000).astype(int);clim=np.load(M4/'cache/Reconstructed_ERA5_0p1_Cell_ID.npy',mmap_mode='r')[ev.row.to_numpy(int),ev.col.to_numpy(int)];pixels=ev.row.to_numpy(int)*703+ev.col.to_numpy(int)
 rows=[];curves=[];binsout=[];countout=[];fullAll=np.load(O/'cache/Full_Severity_ALL_2000_2024.npy',mmap_mode='r')
 comparisons=[('LOYO','REF_2000_2020',year<=2020,1,2),('LOYO','OUT_2021_2024',year>=2021,1,2),('LOYO','ALL_2000_2024',(year>=2000),1,2),('EARLY','EARLY_APPLY_2011_2024',year>=2011,3,4)]
 # Historical saved-bin fits remain separately identified, never overwritten.
 historicalbin=pd.read_csv(D/'L5_EventEcoFunction/L5_4_ThresholdAnalysis/L5_4A_Severity_Response_Binned.csv');prob=pd.read_csv(D/'L5_EventEcoFunction/L5_4_ThresholdAnalysis/L5_4A_Severity_DamageProbability_Binned.csv')
 for response in ['Damage_probability','Acomp_damage','Acomp_legacy']:
  h=(prob[prob.event_type=='CDH'] if response=='Damage_probability' else historicalbin[(historicalbin.event_type=='CDH')&(historicalbin.response==response)]);xx=h['severity_mean' if response=='Damage_probability' else 'severity_median'].to_numpy();yy=h['damage_probability' if response=='Damage_probability' else 'response_median'].to_numpy();ww=np.sqrt(h.n.to_numpy());cand=np.sort(sourcecand[sourcecand.Response==response].Candidate_S.unique());a0=fit(xx,yy,ww)[1];values=[fit(xx,yy,ww,c) for c in cand];j=np.argmin([q[1] for q in values]);his=saved[(saved.event_type=='CDH')&(saved.response==response)].iloc[0];assert abs(cand[j]-his.breakpoint_severity)<1e-12 and abs(a0-values[j][1]-his.delta_AIC_linear_minus_segmented)<1e-8
  rows.append(dict(Event='CDH',Comparison='HISTORICAL_SAVED_BIN',Period='ALL_2000_2024',Response=response,Stage='HISTORICAL_SAVED_BIN_REPRODUCTION',Status='VERIFIED',S_star=cand[j],Q_Full_Comparable=100*np.searchsorted(fullAll,cand[j],side='right')/len(fullAll),Full_ECDF_N=len(fullAll),Full_ECDF_Reference='all original CDH catalogue events onset2000-2024, clean-pre not imposed',Delta_AIC=a0-values[j][1],Search_Min=cand.min(),Search_Max=cand.max(),Candidate_N=len(cand),Upper_Edge_Hit=bool(cand[j]==cand.max()),Lower_Edge_Hit=bool(cand[j]==cand.min()),N_Bins=len(xx),N_Events=int(h.n.sum()),Original_Source=str(D/'L5_EventEcoFunction/L5_4_ThresholdAnalysis/L5_4A_Severity_Response_Thresholds.csv'),Limit='saved historical sampled bin medians; not newly standardized response; no new uncertainty claim'))
 for comparison,period,periodmask,a,b in comparisons:
  full=np.load(O/'cache'/f'Full_Severity_{period}.npy',mmap_mode='r')
  for response,f in [('Damage_probability',1),('Acomp_damage',1),('Acomp_legacy',4)]:
   finite=periodmask&np.isfinite(met[:,0,f])&np.isfinite(met[:,a,f])&np.isfinite(met[:,b,f]);native=periodmask&np.isfinite(met[:,0,f])
   if response=='Acomp_damage':finite&=(met[:,0,1]>0)&(met[:,a,1]>0)&(met[:,b,1]>0);native&=met[:,0,1]>0
   # Shared positivity for magnitude implements the original conditional-damage
   # estimand on identical events. Fixed-original-positive signed test below.
   common=finite;selectedS=np.sort(s[common]);cand=np.sort(sourcecand[sourcecand.Response==response].Candidate_S.unique());pairid=f'{comparison}_{period}_{response}';nr=int(common.sum());basefits=[]
   stages=[('ORIGINAL_NATIVE_ALL_ELIGIBLE',0,native),('ORIGINAL_NATIVE_AT_COMMON_EVENTS',0,common),('ORIGINAL_COMMON_PHASE',a,common),('ALTERNATIVE_COMMON_PHASE',b,common)]
   if response=='Acomp_damage':
    fixed=periodmask&np.isfinite(met[:,0,1])&np.isfinite(met[:,a,1])&np.isfinite(met[:,b,1])&(met[:,0,1]>0)
    stages.extend([('ORIGINAL_FIXED_ORIGINAL_POSITIVE',a,fixed),('ALTERNATIVE_FIXED_ORIGINAL_POSITIVE_SIGNED',b,fixed)])
   for stage,k,mask in stages:
    rng=np.random.default_rng(20260908)
    X=[];Y=[];NN=[]
    for bid in range(len(edges)-1):
     q=np.flatnonzero(mask&(bn==bid));n=len(q)
     if n<500:continue
     use=q if response=='Damage_probability' else historical_joint_sample(q,ev.event_num.to_numpy(),rng);vals=met[use,k,f];xx=float(s[use].mean() if response=='Damage_probability' else np.median(s[use]));yy=float(np.mean(vals>0) if response=='Damage_probability' else np.median(vals));X.append(xx);Y.append(yy);NN.append(n);binsout.append(dict(Pair_ID=pairid,Stage=stage,Bin_ID=bid,N=n,N_Sampled=len(use),Severity_X=xx,Response_Y=yy,RSS_Weight=np.sqrt(n),N_50km_Blocks=len(np.unique(b50[q])),N_Climate_Cells=len(np.unique(clim[q])),Sample_Event_ID_SHA256=hashlib.sha256(ev.event_num.to_numpy()[use].tobytes()).hexdigest(),Subsample_Seed=20260908,Sampling='original1M-ID chunk joint bounded20k cap, same sampled event IDs across paired variants; probability uses all events'))
    xx=np.asarray(X);yy=np.asarray(Y);ww=np.sqrt(NN)
    if len(xx)<12:
     rows.append(dict(Event='CDH',Comparison=comparison,Period=period,Response=response,Stage=stage,Status='NOT COMPLETED',N_Events=int(mask.sum()),Missing_Reason='fewer than12 original bins reach unchanged MIN_N500; no search expansion or favorable replacement'));continue
    # Keep the EXACT archived candidate grid. All candidates are re-evaluated;
    # S* is free, no new S values or wider search are introduced.
    a0=fit(xx,yy,ww)[1];values=[fit(xx,yy,ww,c) for c in cand];aics=np.array([v[1] for v in values]);best=int(np.argmin(aics));star=cand[best];qfull=100*np.searchsorted(full,star,side='right')/len(full);qan=100*np.searchsorted(selectedS,star,side='right')/len(selectedS)
    r=dict(Event='CDH',Pair_ID=pairid,Comparison=comparison,Period=period,Response=response,Stage=stage,Status='COMPLETED',S_star=star,Q_Full_Comparable=qfull,Full_ECDF_N=len(full),Full_ECDF_Reference=f'all original CDH catalogue events restricted to onset years of{period}; same population for both standardizations; no clean-pre filter',Q_Common_Analysis_Cohort=qan,Analysis_ECDF_N=len(selectedS),Analysis_ECDF_Reference='same common eligible response cohort, positive in BOTH versions only for conditional damage magnitude',Q_Full_2000_2024_Auxiliary=100*np.searchsorted(fullAll,star,side='right')/len(fullAll),Delta_AIC=a0-aics[best],Linear_AIC=a0,Segmented_AIC=aics[best],Support_Level='STRONG' if a0-aics[best]>=10 else ('MODERATE' if a0-aics[best]>=4 else 'UNSUPPORTED'),Search_Min=cand.min(),Search_Max=cand.max(),Candidate_N=len(cand),Candidate_Grid='EXACT_ORIGINAL_SAVED_BIN_historical_VALUES',Upper_Edge_Hit=bool(star==cand.max()),Lower_Edge_Hit=bool(star==cand.min()),N_Bins=len(xx),N_Events=int(mask.sum()),N_Events_In_Used_Bins=int(np.sum(NN)),N_Pixels=len(np.unique(pixels[mask])),N_Climate_Support_Cells=len(np.unique(clim[mask])),N_50km_Blocks=len(np.unique(b50[mask])),N_100km_Blocks=len(np.unique(b100[mask])),Pre_Slope=values[best][0][1],Post_Slope=values[best][0][1]+values[best][0][2],Limit='original edges/MIN_N500/sqrtN RSS/k2-vs-k4 AIC; exact common phases/events; point candidates not independent ecological critical points; no new breakpoint BootstrapCI')
    rows.append(r);basefits.append(r)
    for v,aa in zip(cand,aics):curves.append(dict(Pair_ID=pairid,Stage=stage,Candidate_S=v,AIC=aa,Delta_AIC_Linear=a0-aa,Delta_AIC_From_Best=aa-aics[best]))
   ref=next((q for q in basefits if q['Stage']=='ORIGINAL_COMMON_PHASE'),None)
   if ref:
    fixedref=next((q for q in basefits if q['Stage']=='ORIGINAL_FIXED_ORIGINAL_POSITIVE'),ref)
    for q in basefits:
     r=fixedref if 'FIXED_ORIGINAL_POSITIVE' in q['Stage'] else ref
     q['Paired_Original_Stage']=r['Stage'];q['Delta_S_From_Paired_Original']=q['S_star']-r['S_star'];q['Delta_Q_From_Paired_Original']=q['Q_Full_Comparable']-r['Q_Full_Comparable'];q['Delta_AIC_From_Paired_Original']=q['Delta_AIC']-r['Delta_AIC']
   countout.append(dict(Pair_ID=pairid,N_Period_Clean_Events=int(periodmask.sum()),N_Native_Eligible=int(native.sum()),N_Common_Eligible=int(common.sum()),Fraction_Native_Retained=float((native&common).sum()/native.sum()) if native.any() else np.nan,N_Native_Positive_New_Nonpositive=int((periodmask&np.isfinite(met[:,a,1])&np.isfinite(met[:,b,1])&(met[:,a,1]>0)&(met[:,b,1]<=0)).sum())))
 # Low-cost fixed-original-positive signed magnitude cohort checks, no new search.
 pd.DataFrame(rows).to_csv(O/'M6_Breakpoint_Stability.csv',index=False,encoding='utf-8-sig');pd.DataFrame(curves).to_csv(O/'supplement/Breakpoint_Candidate_AIC.csv',index=False);pd.DataFrame(binsout).to_csv(O/'supplement/Breakpoint_Binned_Responses.csv',index=False);pd.DataFrame(countout).to_csv(O/'supplement/Breakpoint_Comparison_Cohorts.csv',index=False)
 (O/'cache/BREAKPOINT_READY.json').write_text(json.dumps(dict(N_Rows=len(rows),No_Range_Expansion=True,Candidate_Grid='archived complete original candidate sets; all knots searched',N_New_Bootstrap=0),indent=2),encoding='utf8');print('BREAKPOINT COMPLETE',len(rows),flush=True)
if __name__=='__main__':main()
