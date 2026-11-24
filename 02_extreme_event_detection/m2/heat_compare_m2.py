# -*- coding: utf-8 -*-
from pathlib import Path
import json,time,gc
import numpy as np
import pandas as pd
OUT=Path(r'.\EI_Reviewer1_Major2_Revision');DB=Path(r'.\Weihe database V3');M1=Path(r'.\EI_Reviewer1_Major1_Revision')
def describe(v):
    v=np.asarray(v);good=v[np.isfinite(v)]
    if not len(good):return dict(N_Valid=0,N_Missing=len(v),Mean=np.nan,SD=np.nan,Min=np.nan,Q05=np.nan,Q25=np.nan,Median=np.nan,Q75=np.nan,Q95=np.nan,Q99=np.nan,Max=np.nan)
    q=np.quantile(good,[0,.05,.25,.5,.75,.95,.99,1]);return dict(N_Valid=len(good),N_Missing=len(v)-len(good),Mean=float(good.mean()),SD=float(good.std()),**{n:float(x) for n,x in zip(['Min','Q05','Q25','Median','Q75','Q95','Q99','Max'],q)})
def main():
    while not (OUT/'cache/OVERLAP_READY.json').exists():time.sleep(3)
    metadata=json.loads((M1/'cache/HEAT/READY.json').read_text());n=metadata['rows'];nc=metadata['clean']
    flag=np.load(OUT/'cache/HEAT_Membership_By_EventNum.npy',mmap_mode='r');full=np.load(M1/'cache/HEAT/all_severity_sorted.npy',mmap_mode='r')
    sev=np.lib.format.open_memmap(OUT/'cache/HEAT_All_Severity.npy',mode='w+',dtype='float64',shape=(n,));flags=np.lib.format.open_memmap(OUT/'cache/HEAT_All_ClassFlags.npy',mode='w+',dtype='uint8',shape=(n,))
    clean=np.lib.format.open_memmap(OUT/'cache/HEAT_Clean_Responses.npy',mode='w+',dtype='float64',shape=(nc,4));cf=np.lib.format.open_memmap(OUT/'cache/HEAT_Clean_ClassFlags.npy',mode='w+',dtype='uint8',shape=(nc,));validres=np.lib.format.open_memmap(OUT/'cache/HEAT_Clean_ValidResistance.npy',mode='w+',dtype='bool',shape=(nc,))
    seen=np.zeros(n+1,bool);at=0;ct=0;start=time.time()
    path=DB/'L5_EventEcoFunction/L5_3_SeverityResilience/L5_3_HEAT_Event_Severity_Resilience_Metrics.csv.gz'
    cols=['event_id','climate_cumulative_severity','clean_pre_any_event','valid_resistance_sample','Acomp_damage','Acomp_legacy_anomaly_days','Acomp_peak_lag_days']
    for j,d in enumerate(pd.read_csv(path,usecols=cols,chunksize=100000)):
        ids=d.event_id.str.rsplit('_',n=1).str[-1].astype(int).to_numpy();assert (ids>0).all() and (ids<=n).all() and not seen[ids].any() and len(np.unique(ids))==len(ids)
        seen[ids]=True;f=flag[ids];v=d.climate_cumulative_severity.to_numpy(float);nn=len(d);sev[at:at+nn]=v;flags[at:at+nn]=f;at+=nn
        good=d.clean_pre_any_event.eq(1).to_numpy()&np.isfinite(v);m=int(good.sum())
        clean[ct:ct+m]=np.column_stack([v[good],d.Acomp_damage.to_numpy(float)[good],d.Acomp_peak_lag_days.to_numpy(float)[good],d.Acomp_legacy_anomaly_days.to_numpy(float)[good]]);cf[ct:ct+m]=f[good];validres[ct:ct+m]=d.valid_resistance_sample.to_numpy()[good]==1;ct+=m
        if j%30==0:print('HEAT L53',j,at,ct,'sec',round(time.time()-start,1),flush=True)
    assert at==n and ct==nc and seen[1:].all()
    for a in [sev,flags,clean,cf,validres]:a.flush()
    rows=[];sample=[]
    for cohort,f,mat in [('ALL_ORIGINAL_HEAT',flags,None),('CLEAN_PRE_FINITE_SEVERITY',cf,clean)]:
        masks={'ALL_HEAT':np.ones(len(f),bool),'HEAT_ONLY_NO_CDH_MERGED_INTERVAL':(f&2)>0,'HEAT_WITH_CDH_MERGED_OVERLAP':(f&2)==0,'HEAT_ONLY_KNOWN_SOIL':((f&2)>0)&((f&4)>0)}
        for name,mask in masks.items():
            N=int(mask.sum());sample.append(dict(Cohort=cohort,Group=name,N_Events=N,Membership='original HEAT ID; no CDH merged interval overlap' if 'ONLY' in name else 'all or CDH-overlap complement',Severity_Reference='original HEAT severity unchanged',Source=str(path)))
            v=np.asarray(sev[mask] if mat is None else mat[mask,0]);result=describe(v);mid=result['Median'];q=100*np.searchsorted(full,mid,side='right')/len(full) if np.isfinite(mid) else np.nan
            rows.append(dict(Cohort=cohort,Group=name,Response='climate_cumulative_severity',N_Events=N,Units='degree_Celsius days',**result,Median_Severity_in_Full_HEAT_ECDF_Q=q,Interpretation='descriptive original severity; no rescaling within subgroup'))
            if mat is not None:
                for col,response,units in [(1,'Acomp_damage','dimensionless anomaly'),(2,'Acomp_peak_lag_days','actual days from onset'),(3,'Acomp_legacy_anomaly_days','anomaly-days')]:
                    use=mask&validres if col==1 else mask
                    result=describe(mat[use,col]);rows.append(dict(Cohort=cohort,Group=name,Response=response,N_Events=N,Units=units,**result,Median_Severity_in_Full_HEAT_ECDF_Q=np.nan,Interpretation='damage requires valid_resistance;legacy finite subset with original recovery/censor-dependent integration'))
                good=mask&validres&np.isfinite(mat[:,1]);num=int((mat[good,1]>0).sum());den=int(good.sum());rows.append(dict(Cohort=cohort,Group=name,Response='Positive_Acomp_Damage_Probability',N_Events=N,N_Valid=den,N_Missing=N-den,Mean=num/den if den else np.nan,Units='proportion',Positive_N=num,Interpretation='descriptive positive-damage fraction, not a survival recovery probability'))
    pd.DataFrame(sample).to_csv(OUT/'M2_HEAT_Only_Sample_Counts.csv',index=False,encoding='utf-8-sig');pd.DataFrame(rows).to_csv(OUT/'M2_HEAT_Only_Response_Comparison.csv',index=False,encoding='utf-8-sig')
    (OUT/'cache/HEAT_COMPARE_READY.json').write_text(json.dumps(dict(N_All=at,N_Clean=ct,Unique_ID_Join=True,Any_Missing_ID=False,Metric_Columns=['Severity','Acomp_damage','Acomp_peak_lag_days','Acomp_legacy_anomaly_days']),indent=2),encoding='utf8')
    print('HEAT COMPARISON READY',at,ct,flush=True)
if __name__=='__main__':main()
