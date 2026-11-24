# -*- coding: utf-8 -*-
from pathlib import Path
import json
import numpy as np
import pandas as pd
O=Path(r'.\EI_Reviewer1_Major6_Revision');M1=Path(r'.\EI_Reviewer1_Major1_Revision');M4=Path(r'.\EI_Reviewer1_Major4_Revision')
def moments(v):
 n,s,ss,cap,zero,tail=v
 return dict(N_Phase_Pixels=int(n),Mean=s/n if n else np.nan,SD=np.sqrt(max(0,ss/n-(s/n)**2)) if n else np.nan,Fraction_AbsGt10=cap/n if n else np.nan,Fraction_Zero=zero/n if n else np.nan,Fraction_AbsGt4p5=tail/n if n else np.nan)
def describe(x):
 x=np.asarray(x);x=x[np.isfinite(x)]
 if not len(x):return dict(N_Valid=0,Mean=np.nan,SD=np.nan,Median=np.nan,Q25=np.nan,Q75=np.nan)
 return dict(N_Valid=len(x),Mean=float(x.mean()),SD=float(x.std()),Median=float(np.median(x)),Q25=float(np.quantile(x,.25)),Q75=float(np.quantile(x,.75)))
def main():
 meta=json.loads((O/'cache/EXTRACTION_READY.json').read_text());ev=pd.read_parquet(O/'paired/CDH_Fixed_Original_Clean_Events.parquet');m=np.load(O/'cache/CDH_Event_Metrics.npy',mmap_mode='r');z=np.load(O/'cache/Extraction_Moments_Checkpoint.npz');mom=z['Indicator_Moments'];year=np.load(O/'cache/year.npy');dates=pd.DatetimeIndex(np.load(O/'cache/dates.npy'));onsetyear=ev.Onset_Year.to_numpy();pixel=(ev.row.to_numpy(int)*703+ev.col.to_numpy(int));maps=np.load(M4/'cache/Reconstructed_ERA5_0p1_Cell_ID.npy',mmap_mode='r');clim=maps[ev.row.to_numpy(int),ev.col.to_numpy(int)];out=[[],[]]
 # Original native phase support from existing immutable Acomp memmap.
 orig=np.load(M1/'cache/Acomp_by_row.npy',mmap_mode='r');domain=np.load(M4/'cache/Climate_Valid_Mask.npy',mmap_mode='r');nativephase=np.zeros(25,np.int64)
 for y0 in range(0,414,12):
  y1=min(414,y0+12);a=np.transpose(orig[y0:y1],(1,0,2));mask=domain[y0:y1]
  for yr in range(2000,2025):nativephase[yr-2000]+=int((np.isfinite(a[year==yr])&mask).sum())
 for pair,name in [(0,'LOYO'),(1,'EARLY')]:
  ranges=[('REF_2000_2020',2000,2020),('OUT_2021_2024',2021,2024)] if pair==0 else [('APPLY_2011_2024',2011,2024)]
  ranges+=[(f'YEAR_{y}',y,y) for y in range(2000 if pair==0 else 2011,2025)]
  for label,start,end in ranges:
   mm=mom[pair,start-2000:end-1999].sum(axis=0)
   for k,ind in enumerate(['ZSync','ZuWUE','ZNIRv','Acomp_state']):
    for variant in [0,1]:out[pair].append(dict(Row_Type='COMMON_PHASE_INDICATOR',Comparison=name,Period=label,Variant='ORIGINAL_COMMON_PHASE' if variant==0 else name,Indicator=ind,Units='Z' if k<3 else 'RMS_of_capped_negative_Z',**moments(mm[k,variant]),Reference='exact common finite all3-component phase support; native raw inputs fixed',Limit='per-observation-year LOYO;2021-2024 original reference untouched; state SD is phase-pixel dispersion'))
   out[pair].append(dict(Row_Type='ANALYTIC_PHASE_SUPPORT',Comparison=name,Period=label,N_Original_Phase_Pixels=int(nativephase[start-2000:end-1999].sum()),N_Common_Phase_Pixels=int(mm[3,0,0]),Fraction_Original_Phases_Retained=mm[3,0,0]/nativephase[start-2000:end-1999].sum(),Limit='common phase domain in fixed167546-climate-pixel basin; new-only phases outside original support not included in paired inference'))
  a,b=(1,2) if pair==0 else (3,4)
  evscopes=[('REF_2000_2020',onsetyear<=2020),('OUT_2021_2024',onsetyear>=2021),('OUT_2022_2024_NEGATIVE_CONTROL',onsetyear>=2022),('YEAR2021_PRE_CROSSES2020',(onsetyear==2021)&(ev.Pre_Window_Start_Year.to_numpy()==2020))] if pair==0 else [('APPLY_2011_2024',onsetyear>=2011)]
  evscopes.extend([(f'EVENT_YEAR_{y}',onsetyear==y) for y in range(2000 if pair==0 else 2011,2025)])
  for label,scope in evscopes:
   good=scope&np.isfinite(m[:,a,1])&np.isfinite(m[:,b,1]);native=scope&np.isfinite(m[:,0,1]);N=int(good.sum());base=dict(Comparison=name,Period=label,N_Input_Clean_Events=int(scope.sum()),N_Original_Native_Valid=int(native.sum()),N_Common_Events=N,N_Common_Pixels=len(np.unique(pixel[good])),N_Common_Climate_Support_Cells=len(np.unique(clim[good])),N_Original_Native_Pixels=len(np.unique(pixel[native])))
   out[pair].append(dict(base,Row_Type='COMMON_EVENT_SUPPORT',Fraction_Original_Valid_Retained=float((native&good).sum()/native.sum()) if native.any() else np.nan,Limit='same event IDs and finite phase support; legacy comparison uses additional common finite-legacy intersection'))
   for f,field,unit in [(0,'Event_Pre_Baseline','RMS'),(1,'Acomp_Damage','RMS_increment'),(2,'Peak_Lag_Days','days'),(4,'Historical_Legacy','RMS_anomaly_days')]:
    use=good&np.isfinite(m[:,a,f])&np.isfinite(m[:,b,f]);aa=m[use,a,f];bb=m[use,b,f];delta=describe(bb-aa)
    out[pair].append(dict(base,Row_Type='PAIRED_EVENT_RESPONSE',Indicator=field,Units=unit,N_Common_Response=int(use.sum()),N_Response_Pixels=len(np.unique(pixel[use])),N_Response_Climate_Cells=len(np.unique(clim[use])),Original_Mean=float(aa.mean()) if len(aa) else np.nan,Alternative_Mean=float(bb.mean()) if len(bb) else np.nan,Original_SD=float(aa.std()) if len(aa) else np.nan,Alternative_SD=float(bb.std()) if len(bb) else np.nan,Original_Median=float(np.median(aa)) if len(aa) else np.nan,Alternative_Median=float(np.median(bb)) if len(bb) else np.nan,Delta_Mean=delta['Mean'],Delta_Median=delta['Median'],Delta_Q25=delta['Q25'],Delta_Q75=delta['Q75'],Fraction_AbsDelta_ge8=float(np.mean(np.abs(bb-aa)>=8)) if len(aa) and f==2 else np.nan,Limit='legacy is exact historical L5.3 rule for both versions; not M5 corrected observed legacy; no interpolated endpoints'))
   old=m[good,a,1];new=m[good,b,1];out[pair].append(dict(base,Row_Type='DAMAGE_CLASSIFICATION',Original_Positive_N=int((old>0).sum()),Alternative_Positive_N=int((new>0).sum()),Original_Positive_Fraction=float(np.mean(old>0)) if N else np.nan,Alternative_Positive_Fraction=float(np.mean(new>0)) if N else np.nan,Positive_To_Nonpositive=int(((old>0)&(new<=0)).sum()),Nonpositive_To_Positive=int(((old<=0)&(new>0)).sum()),Limit='positive damage means own0..56d maximum exceeds ownpre40d median; numerical comparison, not significance'))
 for i,n in enumerate(['M6_LOYO_Indicator_Comparison.csv','M6_EarlyBaseline_Comparison.csv']):pd.DataFrame(out[i]).to_csv(O/n,index=False,encoding='utf-8-sig')
 # Every phase actual valid-year histogram, not filled calendar counts.
 nh=z['N_Histogram'];supp=z['Support'];rh=[];support=[]
 for k,var in enumerate(['Sync','uWUE','NIRv']):
  for baseline in range(2):
   for p in range(46):
    hist=nh[k,baseline,p];n=int(hist.sum());cs=np.cumsum(hist);med=int(np.searchsorted(cs,(n+1)//2));active=np.flatnonzero(hist);support.append(dict(Indicator=var,Baseline='2000_2020' if baseline==0 else '2000_2010',Phase=p+1,N_Basin_Pixels=n,Valid_Years_Min=int(active.min()),Valid_Years_Median=med,Valid_Years_Max=int(active.max()),N_Count_ge10=int(supp[k,baseline,p,1]),N_Supported_SD=int(supp[k,baseline,p,2]),N_ge10_but_SD_Unsupported=int(supp[k,baseline,p,3]),N_Missing_Count=int(n-supp[k,baseline,p,1])))
    for count,value in enumerate(hist):
     if value:rh.append(dict(Indicator=var,Baseline='2000_2020' if baseline==0 else '2000_2010',Phase=p+1,Actual_Valid_Reference_Years=count,N_Basin_Pixels=int(value)))
 pd.DataFrame(rh).to_csv(O/'supplement/Reference_Valid_Year_Histograms.csv',index=False);pd.DataFrame(support).to_csv(O/'supplement/Reference_Phase_Support.csv',index=False)
 loyo=z['N_LOYO_Histogram'];nrows=[]
 for k,var in enumerate(['Sync','uWUE','NIRv']):
  for yr in range(2000,2021):
   for p in range(46):
    hist=loyo[k,yr-2000,p];total=int(hist.sum());active=np.flatnonzero(hist)
    nrows.append(dict(Indicator=var,Target_Year=yr,Phase=p+1,N_Basin_Pixels=total,Reference_Valid_Years_Min=int(active.min()) if total else np.nan,Reference_Valid_Years_Median=int(np.searchsorted(np.cumsum(hist),(total+1)//2)) if total else np.nan,Reference_Valid_Years_Max=int(active.max()) if total else np.nan,N_Reference_Count_ge10=int(hist[10:].sum()),Target_Phase_Observed=bool(total),Missing_Reason='NONE' if total else 'targetcalendarphaseabsent in2000; no synthetic observation or LOYO statistic'))
 pd.DataFrame(nrows).to_csv(O/'supplement/LOYO_Target_Year_Phase_Reference_Counts.csv',index=False)
 sd=np.load(O/'cache/SD_Ratio_Samples.npy');rows=[]
 for k,var in enumerate(['Sync','uWUE','NIRv']):
  vals=sd[sd[:,0]==k,2];rows.append(dict(Indicator=var,**describe(vals),Q05=float(np.quantile(vals,.05)),Q95=float(np.quantile(vals,.95)),Min=float(vals.min()),Max=float(vals.max()),Sampling='fixed geographic pixel index modulo37, all phases with both reference SD supported; earlySD/originalSD'))
 pd.DataFrame(rows).to_csv(O/'supplement/Early_Reference_SD_Ratios.csv',index=False)
 # Strict negative control, not diluted with reference-period years.
 check=(onsetyear>=2022)&np.isfinite(m[:,1,1])&np.isfinite(m[:,2,1]);err=float(np.max(np.abs(m[check,1,1]-m[check,2,1]))) if check.any() else np.nan;assert err==0
 (O/'audit/Negative_Control.json').write_text(json.dumps(dict(Post2020_Phase_Exact=True,Event_2022_2024_N=int(check.sum()),Event_2022_2024_Damage_MaxAbs=err,Year2021_PreWindowCrosses2020_N=int(((onsetyear==2021)&(ev.Pre_Window_Start_Year.to_numpy()==2020)).sum())),indent=2),encoding='utf8');print('COMPARE COMPLETE',len(out[0]),len(out[1]),'rows',flush=True)
if __name__=='__main__':main()
