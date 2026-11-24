"""Read L5.3 once in 100k-row chunks; regroup by ten for original 1M-row replay.
Never write source directories.
"""
from pathlib import Path
import os,json,time,gc
import numpy as np
import pandas as pd
import xarray as xr
import pyarrow as pa
import pyarrow.parquet as pq
OUT=Path(r'.\EI_Reviewer1_Major1_Revision'); BASE=Path(r'.\Weihe database V3')
L5=BASE/'L5_EventEcoFunction'; CACHE=OUT/'cache'
CHUNK=100_000
ds=xr.open_dataset(BASE/'L4_EcoFunction/Weihe_L4_EcoFunction_StateSpace_1000m_2000_2024_V3_1.nc')
dates=pd.DatetimeIndex(ds.time.values); date_map={t.strftime('%Y-%m-%d'):i for i,t in enumerate(dates)}
nx=ds.sizes['x']; ny=ds.sizes['y']; ds.close()
cols=['event_id','row','col','x','y','onset_time','termination_time','clean_pre_any_event','climate_cumulative_severity','Acomp_pre','Acomp_damage','Acomp_peak_lag_days','Acomp_legacy_anomaly_days','valid_resistance_sample','Acomp_recovery_found','Acomp_recovery_days_after_termination','next_event_days_after_termination','recovery_censored_by_next_event']
for et in ['CDH','HEAT','HR']:
    folder=CACHE/et; folder.mkdir(exist_ok=True)
    if (folder/'READY.json').exists(): print(et,'cache already ready',flush=True);continue
    stats=dict(event_type=et,rows=0,clean=0,finite_severity=0,legacy_valid=0,recovery_base=0,id_duplicates=0,pixel_onset_duplicates=0,coord_mismatches=0,onset_min='9999',onset_max='',term_min='9999',term_max='')
    p=L5/'L5_3_SeverityResilience'/f'L5_3_{et}_Event_Severity_Resilience_Metrics.csv.gz'
    ids=[]; keys=[]; start=time.time()
    for j,df in enumerate(pd.read_csv(p,usecols=cols,chunksize=CHUNK,low_memory=False)):
        sev=df.climate_cumulative_severity.to_numpy(float); clean=df.clean_pre_any_event.eq(1).to_numpy()
        np.savez(folder/f'raw_severity_{j:03d}.npz',severity=sev,clean=clean)
        stats['rows']+=len(df); stats['finite_severity']+=int(np.isfinite(sev).sum()); stats['clean']+=int((clean&np.isfinite(sev)).sum())
        stats['legacy_valid']+=int((clean&np.isfinite(sev)&np.isfinite(df.Acomp_legacy_anomaly_days)).sum())
        stats['recovery_base']+=int((clean&np.isfinite(sev)&np.isfinite(df.Acomp_pre)&np.isfinite(df.Acomp_peak_lag_days)).sum())
        eid=df.event_id.str.rsplit('_',n=1).str[-1].astype('int64').to_numpy();ids.append(eid)
        oi=df.onset_time.map(date_map).fillna(-1).to_numpy(np.int32)
        ti=df.termination_time.map(date_map).fillna(-1).to_numpy(np.int32)
        if (oi<0).any() or (ti<0).any(): raise ValueError('Unmapped dates')
        keys.append(((df.row.to_numpy(np.int64)*nx+df.col.to_numpy(np.int64))*len(dates)+oi))
        for label,c in [('onset','onset_time'),('term','termination_time')]:
            stats[label+'_min']=min(stats[label+'_min'],df[c].min()); stats[label+'_max']=max(stats[label+'_max'],df[c].max())
        d=df.loc[clean&np.isfinite(sev)].copy()
        d['onset_idx']=oi[clean&np.isfinite(sev)];d['term_idx']=ti[clean&np.isfinite(sev)]
        d['event_num']=eid[clean&np.isfinite(sev)]
        d=d.drop(columns=['event_id','onset_time','termination_time','clean_pre_any_event'])
        pq.write_table(pa.Table.from_pandas(d,preserve_index=False),folder/f'clean_{j:03d}.parquet',compression='zstd')
        if j%10==0: print(et,j,'rows',stats['rows'],'clean',stats['clean'],'seconds',round(time.time()-start,1),flush=True)
        del df,d;gc.collect()
    allid=np.concatenate(ids);allkey=np.concatenate(keys)
    stats['id_duplicates']=int(len(allid)-len(np.unique(allid)));stats['pixel_onset_duplicates']=int(len(allkey)-len(np.unique(allkey)))
    stats['id_min']=int(allid.min());stats['id_max']=int(allid.max())
    (folder/'READY.json').write_text(json.dumps(stats,indent=2),encoding='utf8')
    del ids,keys,allid,allkey;gc.collect()
    print('READY',stats,flush=True)
