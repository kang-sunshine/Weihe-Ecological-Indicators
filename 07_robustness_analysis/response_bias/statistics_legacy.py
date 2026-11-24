# -*- coding: utf-8 -*-
import json,time,gc
import numpy as np
import pandas as pd
from core import *
PROBS=np.array([.05,.25,.5,.75,.95]);LABELS=['Q05','Q25','Median','Q75','Q95']

def describe(values,mask,pix,b,W,context,rows):
 mask=mask&np.isfinite(values);sup=support(mask,pix,b)
 if not mask.any():rows.append(dict(**context,**sup,Statistic='Mean',Estimate=np.nan,CI_Status='NOT ESTIMABLE',Missing_Reason='no finite observed values'));return
 v=np.asarray(values[mask],float);bb=np.asarray(b[mask],int);pt,rep=paired_mean(values,mask,b,W);row=dict(**context,**sup,Statistic='Mean',Estimate=pt,**ci(rep));rows.append(row)
 boot=quantile_boot(v,bb,W,PROBS);point=np.quantile(v,PROBS)
 for j,name in enumerate(LABELS):rows.append(dict(**context,**sup,Statistic=name,Estimate=point[j],**ci(boot[:,j])))

def correlation(x,y,mask,pix,b,W,context):
 mask=mask&np.isfinite(x)&np.isfinite(y);z=np.zeros((76,6));xx=np.asarray(x[mask],float);yy=np.asarray(y[mask],float);bb=b[mask]
 for j,v in enumerate([np.ones(len(xx)),xx,yy,xx*xx,yy*yy,xx*yy]):z[:,j]=np.bincount(bb,weights=v,minlength=76)
 def f(s):
  n=s[...,0];mx=s[...,1]/np.maximum(n,1);my=s[...,2]/np.maximum(n,1);den=np.sqrt(np.maximum(0,s[...,3]-n*mx*mx)*np.maximum(0,s[...,4]-n*my*my));return np.divide(s[...,5]-n*mx*my,den,out=np.full_like(den,np.nan),where=den>0)
 pt=f(z.sum(axis=0));rep=f(W@z);return dict(**context,**support(mask,pix,b),Statistic='Pearson_L_vs_ObservedDays',Estimate=float(pt),**ci(rep))

def bounded_sample(old,new,rng):
 x=new[np.isfinite(new)]
 if len(x)>500000:x=x[rng.choice(len(x),500000,replace=False)]
 if len(old)==0:return x.copy() if len(x)<=500000 else x[rng.choice(len(x),500000,replace=False)]
 z=np.r_[old,x];return z if len(z)<=500000 else z[rng.choice(len(z),500000,replace=False)]

def edges_replay():
 rng=np.random.default_rng(20260908);out={};rows=[]
 for h in HAZ:
  files=sorted((M1/'cache'/h).glob('raw_severity_*.npz'));sample=np.empty(0)
  for k in range(0,len(files),10):
   zz=[np.load(p) for p in files[k:k+10]];severity=np.concatenate([v['severity'] for v in zz]);clean=np.concatenate([v['clean'] for v in zz]);sample=bounded_sample(sample,severity[clean&np.isfinite(severity)],rng)
  edges=np.unique(np.quantile(sample,np.linspace(0,1,81)));edges[0]=-np.inf;edges[-1]=np.inf;out[h]=edges
  for i,s in enumerate(edges):rows.append(dict(Hazard=h,Edge=i,S=s,Reference='original order-dependent500kclean-pre severity reservoir; exact originalseed/pass1/chunk regrouping'))
 pd.DataFrame(rows).to_csv(O/'qc/Historical_80Bin_Edges.csv',index=False);return out

def fit_segmented(x,y,n,lower,upper):
 order=np.argsort(x);x=x[order];y=y[order];n=n[order];w=np.sqrt(n)
 def fit(c=None):
  X=np.column_stack([np.ones(len(x)),x]+([] if c is None else [np.maximum(0,x-c)]));beta=np.linalg.lstsq(X*np.sqrt(w)[:,None],y*np.sqrt(w),rcond=None)[0];rss=np.sum(w*(y-X@beta)**2);aic=len(x)*np.log(max(rss,1e-300)/len(x))+2*(2 if c is None else 4);return beta,aic
 if len(x)<12:return dict(Status='NOT ESTIMABLE',Reason='fewer than12 supportedbins')
 lo=max(int(np.floor(.1*len(x))),4);hi=min(int(np.ceil(.9*len(x))),len(x)-4);cand=np.unique(x[lo:hi]);cand=cand[(cand>=lower)&(cand<=upper)]
 if len(cand)==0:return dict(Status='NOT ESTIMABLE',Reason='no candidate inside unchanged historical S bounds')
 b0,a0=fit();vals=[fit(c) for c in cand];k=int(np.argmin([z[1] for z in vals]));beta,a=vals[k];delta=a0-a
 return dict(Status='ESTIMABLE_CANDIDATE_ONLY',S_star=cand[k],Delta_AIC=delta,Linear_AIC=a0,Segmented_AIC=a,Pre_Slope=beta[1],Post_Slope=beta[1]+beta[2],Slope_Change=beta[2],N_Bins=len(x),Search_Min=float(cand.min()),Search_Max=float(cand.max()),Historical_S_Min=lower,Historical_S_Max=upper,Lower_Edge_Hit=k==0,Upper_Edge_Hit=k==len(cand)-1,Support_Level='STRONG' if delta>=10 else 'MODERATE' if delta>=4 else 'WEAK_OR_NO_SUPPORT',Breakpoint_CI=np.nan,CI_Reason='no repeat of350 spatial breakpointbootstrap; point sensitivity only')

def main():
 assert (O/'qc/EXTRACTION_READY.json').exists();start=time.time();rows=[];fixed=[];bins=[];fits=[];binchecks=[];replicas=[];denoms=[];edges=edges_replay()
 historical=pd.read_csv(DB/'L5_EventEcoFunction/L5_4_ThresholdAnalysis/L5_4A_Severity_Response_Binned.csv');registry=pd.read_csv(M1/'R1M1_Original_Breakpoint_Registry.csv');cutoffs={}
 completed=set()
 progress=[('outputs/C1_Legacy_StopReasons_Duration.csv',rows),('outputs/C2_FixedWindow_Legacy.csv',fixed),('outputs/C3_SeverityLegacy_Sensitivity.csv',bins),('outputs/C3_Paired_Breakpoint_Fits.csv',fits),('qc/Historical_Bin_Edge_Verification.csv',binchecks),('qc/C2_Spatial_Contrast_Replicates.csv.gz',replicas),('qc/Analysis_Denominators.csv',denoms)]
 if all((O/p).exists() for p,_ in progress):
  tables=[pd.read_csv(O/p) for p,_ in progress];completed=set.intersection(*(set(d.Hazard.unique()) for d in tables))
  for d,(_,target) in zip(tables,progress):target.extend(d[d.Hazard.isin(completed)].to_dict('records'))
 # Exact float64 quantile engine validation with ties and whole-block multiplicities.
 vv=np.array([0.,0.,.1,.3,1.,3.,4.]);bb=np.array([0,0,1,1,1,2,2]);ww=np.ones((3,76),np.int32);ww[1,0]=2;ww[2,1]=0;got=quantile_boot(vv,bb,ww,PROBS);error=max(np.max(abs(got[k]-np.quantile(np.repeat(vv,ww[k,bb]),PROBS))) for k in range(3));assert error<1e-12
 (O/'qc/Quantile_Engine_Verification.json').write_text(json.dumps(dict(MaxAbs_vs_Explicit_WholeBlock=np.float64(error).item(),Original_Float64_Nonnegative_Values_Retained=True)),encoding='utf8')
 for h in HAZ:
  if h in completed:print('SKIP completed C',h,flush=True);continue
  e=events(h);m=existing(h);c=np.load(O/'qc/cache'/f'{h}_Legacy_Audit.npy',mmap_mode='r');g=groups(h);b,W=blockdata(h);pix=e.row.to_numpy(np.int32)*703+e.col.to_numpy(np.int32);valid=np.isfinite(m[:,1]);old=e.Acomp_legacy_anomaly_days.to_numpy();cur=m[:,12];N=len(e);full=np.load(M1/'cache'/h/'all_severity_sorted.npy',mmap_mode='r');meta=json.loads((M1/'cache'/h/'READY.json').read_text());source_N=meta['rows'];fullcuts=[0]+[int(np.searchsorted(full,full[int(np.ceil(q*len(full)))-1],side='left')) for q in [.5,.75,.9]]+[len(full)]
  for group in range(5):
   pop=(g==group) if group<4 else np.ones(N,bool);label=GROUPS[group];base=dict(Hazard=h,Severity_Group=label,N_Catalogue_Group=int(fullcuts[group+1]-fullcuts[group]) if group<4 else source_N,N_Clean_Source=int(pop.sum()),N_Baseline_Peak=int((pop&valid).sum()),N_Catalogue_All=source_N)
   denoms.append(base)
   for version,L,days,reason,tail in [('HISTORICAL_L53_ONSET120_ACTIVE_INTERRUPT',old,c[:,10],c[:,4],c[:,11]),('MINOR2_M5RULE_TERM120_CATALOG_INTERRUPT',cur,m[:,13],np.where(valid,m[:,8],9),np.zeros(N))]:
    context=dict(base,Version=version,Row_Type='DISTRIBUTION')
    for field,value,unit in [('Legacy_L',L,'Acomp_anomaly_days'),('Actual_Observed_Days',days,'actual_calendar_days')]:
     describe(value,pop&valid,pix,b,W,dict(context,Metric=field,Unit=unit),rows)
    intensity=np.divide(L,days,out=np.full(N,np.nan),where=np.isfinite(L)&(days>0)&(tail==0));describe(intensity,pop&valid,pix,b,W,dict(context,Metric='L_Over_Actual_Observed_Days',Unit='Acomp_anomaly',Limit='auxiliary intensity; not equivalent toL; excludeszero duration andsyntheticterminalfallback'),rows)
    rows.append(correlation(L,days,pop&valid&(tail==0),pix,b,W,dict(base,Version=version,Row_Type='CORRELATION',Metric='L_and_observation_duration',Unit='correlation')))
    names={0:'CONFIRMED_RECOVERY',1:'NEXT_ACTIVE_PHASE' if version.startswith('HISTORICAL') else 'NEXT_CATALOGUE_ONSET',2:'ONSET_120_BOUND' if version.startswith('HISTORICAL') else 'TERM_120_ADMIN',3:'RECORD_END',4:'MISSING_OR_INCOMPLETE',9:'INVALID_BASELINE_OR_PEAK'}
    for code,name in names.items():
     flag=pop&(reason==code);pt,v=ratio(flag,pop,b,W);rows.append(dict(base,Version=version,Row_Type='STOP_REASON',Metric=name,N_Numerator=int(flag.sum()),N_Denominator=int(pop.sum()),Estimate=pt,Unit='fraction',**support(pop,pix,b),**ci(v)))
    quality={'ZERO_EFFECTIVE_DAYS':valid&(days==0),'LEGACY_UNDEFINED':valid&~np.isfinite(L),'SYNTHETIC_TERMINAL_8DAY_TAIL':valid&(tail>0)}
    if version.startswith('HISTORICAL'):quality['MISSING_PHASES_SKIPPED_NOT_A_STOP']=valid&(c[:,5]>0);quality['HISTORICAL_RECOVERY_BEFORE_TERMINATION']=valid&(c[:,1]>=0)&(c[:,1]<e.term_idx.to_numpy())
    for name,cond in quality.items():
     flag=pop&cond;pt,v=ratio(flag,pop&valid,b,W);rows.append(dict(base,Version=version,Row_Type='DATA_QUALITY',Metric=name,N_Numerator=int(flag.sum()),N_Denominator=int((pop&valid).sum()),Estimate=pt,Unit='fraction',**support(pop&valid,pix,b),**ci(v)))
   print('C1',h,label,'seconds',round(time.time()-start,1),flush=True)
  # Four fixed windows. Eligibility is separate from finite integral availability.
  for j,H in enumerate(WINDOWS):
   eligible=c[:,20+j]==0;L=c[:,12+j];finite=eligible&np.isfinite(L)
   boots={};means={};medians={};n_group={}
   for k,label in enumerate(GROUPS):
    pop=(g==k) if k<4 else np.ones(N,bool);mask=pop&finite;pt,v=ratio(pop&eligible,pop,b,W);context=dict(Hazard=h,Severity_Group=label,Window_Days=H,Row_Type='COVERAGE',Metric='Complete_Window_Eligibility',N_Clean_Source=int(pop.sum()),N_Eligible=int((pop&eligible).sum()),N_Finite_L=int(mask.sum()),N_Eligible_No_Integral_Interval=int((pop&eligible&~np.isfinite(L)).sum()),Estimate=pt,Unit='fraction',**support(pop,pix,b),**ci(v));fixed.append(context)
    for bit,label2 in [(1,'INVALID_BASELINE_PEAK'),(2,'RECORD_DOES_NOT_REACH_H'),(4,'NEW_CATALOGUE_EVENT_WITHIN_H_INCLUSIVE'),(8,'MISSING_SCHEDULED_OBSERVATION')]:
     flag=pop&((c[:,20+j].astype(np.int16)&bit)!=0);pt,v=ratio(flag,pop,b,W);fixed.append(dict(Hazard=h,Severity_Group=label,Window_Days=H,Row_Type='EXCLUSION_NOT_MUTUALLY_EXCLUSIVE',Metric=label2,N_Numerator=int(flag.sum()),N_Denominator=int(pop.sum()),Estimate=pt,Unit='fraction',**support(pop,pix,b),**ci(v)))
    context=dict(Hazard=h,Severity_Group=label,Window_Days=H,Row_Type='GROUP_L',Metric='Fixed_Window_L',Unit='Acomp_anomaly_days',Cohort='complete_window_with_finite_integral; earlyrecovery noobservedinterval remainsmissing')
    describe(L,mask,pix,b,W,context,fixed)
    if mask.any():
     mean,boot=paired_mean(L,mask,b,W);med=np.median(L[mask]);medboot=quantile_boot(L[mask],b[mask],W,np.array([.5]))[:,0];boots[k]=(boot,medboot);means[k]=mean;medians[k]=med;n_group[k]=int(mask.sum())
   if 2 in boots and 3 in boots:
    for name,a,pt in [('Mean',0,means[3]-means[2]),('Median',1,medians[3]-medians[2])]:
     v=boots[3][a]-boots[2][a];fixed.append(dict(Hazard=h,Severity_Group='Extreme_minus_High',Window_Days=H,Row_Type='DIRECT_SPATIAL_CONTRAST',Metric='Fixed_Window_L',Statistic=name,Estimate=pt,N_High=n_group[2],N_Extreme=n_group[3],Unit='Acomp_anomaly_days',**support(finite&((g==2)|(g==3)),pix,b),**ci(v)))
     for r,value in enumerate(v):replicas.append(dict(Hazard=h,Window_Days=H,Statistic=name,Replicate=r,Extreme_Minus_High=value))
  # Historical bin-edge identity verified by original archived bin population counts.
  edge=edges[h];bi=np.clip(np.searchsorted(edge,e.climate_cumulative_severity.to_numpy(),side='right')-1,0,len(edge)-2);ref=historical[(historical.event_type==h)&(historical.response=='Acomp_legacy')];actual=np.bincount(bi[np.isfinite(old)],minlength=len(edge)-1)
  mismatches=int(sum(actual[int(r.bin_id)]!=int(r.n) for r in ref.itertuples()));binchecks.append(dict(Hazard=h,Archived_Supported_Bins=len(ref),Count_Mismatches=mismatches,Pass=mismatches==0));assert mismatches==0,'Original bin version cannot be verified'
  for H in [0]+WINDOWS:
   j=WINDOWS.index(H) if H else 0;mask=np.isfinite(old) if H==0 else np.isfinite(old)&(c[:,20+j]==0)&np.isfinite(c[:,12+j]);fitdata=[]
   for k in range(len(edge)-1):
    mm=mask&(bi==k);n=int(mm.sum())
    if not n:
     bins.append(dict(Hazard=h,Window_Days=H,Bin=k,N_events=0,Status='NOT ESTIMABLE',Missing_Reason='no paired complete support'));continue
    s=e.climate_cumulative_severity.to_numpy()[mm];orig=old[mm];fixedv=c[mm,12+j] if H else orig;sm=float(np.median(s));bins.append(dict(Hazard=h,Window_Days=H,Bin=k,S_Lower=edge[k],S_Upper=edge[k+1],S_Median=sm,Q_Full=100*np.searchsorted(full,sm,side='right')/len(full),Original_L_Median=float(np.median(orig)),Fixed_L_Median=float(np.median(fixedv)),**support(mm,pix,b),Status='SUPPORTED_BIN' if n>=500 else 'BELOW_ORIGINAL_MIN500',Cohort='originalfiniteall' if H==0 else 'samepairedoriginal/fixedfinitecompleteeventIDs',Median_Estimator='exactall_event_median; modelusesoriginal20000cap separately'))
    if H in [32,56] and n>=500:
     # Same historical 20k response sample cap, with shared IDs for paired responses.
     rng=np.random.default_rng(SEED+H+k);ix=rng.choice(n,20000,replace=False) if n>20000 else np.arange(n);fitdata.append((float(np.median(s[ix])),float(np.median(orig[ix])),float(np.median(fixedv[ix])),n))
   if H in [32,56]:
    reg=registry[(registry.Event==h)&registry.Response.eq('Acomp_legacy')].iloc[0]
    for response,col in [('Historical_L_on_same_window_subset',1),('Fixed_Window_L',2)]:
     data=np.array(fitdata);result=fit_segmented(data[:,0],data[:,col],data[:,3],reg.Search_Min,reg.Search_Max) if len(fitdata) else dict(Status='NOT ESTIMABLE',Reason='no supportedbins')
     if 'S_star' in result:result['Q_Full']=100*np.searchsorted(full,result['S_star'],side='right')/len(full)
     fits.append(dict(Hazard=h,Window_Days=H,Response=response,Historical_S_star=reg.S_star,Historical_Delta_AIC=reg.Delta_AIC,N_Paired_Events=int(mask.sum()),N_Paired_Pixels=int(np.unique(pix[mask]).size),N_Paired_Blocks=int(np.unique(b[mask]).size),**result))
  # Thin provenance-rich event details; no raw time series are duplicated.
  for k in range(0,N,200000):
   stop=min(N,k+200000);sl=slice(k,stop);d=e.iloc[sl][['event_num','row','col','onset_idx','term_idx','climate_cumulative_severity','Acomp_pre','Acomp_damage','Acomp_legacy_anomaly_days']].copy();d['Hazard']=h;d['Q_Full']=np.load(MIN/'cache'/h/'Full_ECDF_Q.npy',mmap_mode='r')[sl];d['Severity_Group']=[GROUPS[v] for v in g[sl]];d['Block50']=b[sl]
   def indexdates(v):
    v=np.asarray(v);out=np.full(len(v),'NaT',dtype='datetime64[D]');ok=np.isfinite(v)&(v>=0)&(v<len(dates));out[ok]=dates[v[ok].astype(int)];return out
   dates=np.load(M5/'cache/dates.npy').astype('datetime64[D]');d['Termination_Date']=indexdates(e.term_idx.to_numpy()[sl]);d['Historical_Integration_First_Valid_Date']=indexdates(c[sl,8]);d['Historical_Integration_Last_Valid_Date']=indexdates(c[sl,9]);d['Historical_Stop_Date']=indexdates(c[sl,0]);d['Historical_Recovery_Date']=indexdates(c[sl,1]);d['Historical_Next_Active_Date']=indexdates(c[sl,3]);d['Historical_Stop_Reason_Code']=c[sl,4];d['Historical_Weighted_Days']=c[sl,6];d['Historical_Actual_Observed_Days']=c[sl,10];d['Historical_Synthetic_Tail_Days']=c[sl,11];d['Historical_Missing_Phases_Skipped']=c[sl,5]
   d['M5Rule_L']=m[sl,12];d['M5Rule_Observed_Days']=m[sl,13];d['M5Rule_Outcome']=m[sl,8];d['M5Rule_Followup_Days']=m[sl,7];d['M5Rule_Recovery_Day']=m[sl,10];d['M5Rule_Confirmation_Day']=m[sl,11];d['Next_Catalogue_Onset_Day']=m[sl,16];d['Record_End_Date']=dates[-1]
   for j,H in enumerate(WINDOWS):d[f'L_{H}']=c[sl,12+j];d[f'Days_{H}']=c[sl,16+j];d[f'ExclusionBits_{H}']=c[sl,20+j]
   d.to_parquet(O/f'outputs/event_details/{h}_Legacy_{k//200000:03d}.parquet',index=False,compression='zstd')
  pd.DataFrame(rows).to_csv(O/'outputs/C1_Legacy_StopReasons_Duration.csv',index=False,encoding='utf-8-sig');pd.DataFrame(fixed).to_csv(O/'outputs/C2_FixedWindow_Legacy.csv',index=False,encoding='utf-8-sig');pd.DataFrame(bins).to_csv(O/'outputs/C3_SeverityLegacy_Sensitivity.csv',index=False,encoding='utf-8-sig');pd.DataFrame(fits).to_csv(O/'outputs/C3_Paired_Breakpoint_Fits.csv',index=False,encoding='utf-8-sig');pd.DataFrame(binchecks).to_csv(O/'qc/Historical_Bin_Edge_Verification.csv',index=False);pd.DataFrame(replicas).to_csv(O/'qc/C2_Spatial_Contrast_Replicates.csv.gz',index=False,compression='gzip');pd.DataFrame(denoms).to_csv(O/'qc/Analysis_Denominators.csv',index=False)
  print('C COMPLETE',h,'seconds',round(time.time()-start,1),flush=True);del e,m,c,g,b,W,pix,old,cur,full;gc.collect()
 # Exact archival KM/AJ, interruption and spatial uncertainty reuse; no survival recomputation.
 comparison=pd.read_csv(M5/'M5_KM_CompetingRisk_Comparison.csv');comparison=comparison[comparison.Analysis.eq('PRIMARY_POSTPEAK_CATALOG_INCLUSIVE')].copy();comparison['Source_File']=str(M5/'M5_KM_CompetingRisk_Comparison.csv');comparison['Row_Type']='KM_AJ_ARCHIVED_GROUP_OR_CONTRAST'
 source=pd.read_csv(M5/'M5_Figure6ef_SourceData.csv');source['Source_File']=str(M5/'M5_Figure6ef_SourceData.csv');source['Row_Type']='ARCHIVED_RECOVERY_AND_CENSOR_COUNTS'
 pd.concat([comparison,source],ignore_index=True,sort=False).to_csv(O/'outputs/C4_KM_AJ_ReuseSummary.csv',index=False,encoding='utf-8-sig')
 (O/'qc/C_READY.json').write_text(json.dumps(dict(Complete_ThreeHazards=True,Fixed_Windows=WINDOWS,Original_L_Reproduced=True,Historical_Bin_Counts_Verified=True,No_M5_KM_AJ_Refit=True,Seconds=time.time()-start)),encoding='utf8')
if __name__=='__main__':main()
