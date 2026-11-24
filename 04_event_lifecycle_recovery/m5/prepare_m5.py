# -*- coding: utf-8 -*-
"""Read existing catalogue onsets, not event re-identification. All source access read-only."""
from pathlib import Path
import json,hashlib,sys,time
import numpy as np
import pandas as pd
import xarray as xr
OUT=Path(r'.\EI_Reviewer1_Major5_Revision');M1=Path(r'.\EI_Reviewer1_Major1_Revision');DB=Path(r'.\Weihe database V3')
for folder in ['audit','cache','event_records']: (OUT/folder).mkdir(parents=True,exist_ok=True)
inventory=[]
for root in [M1,DB,Path(r'.\Weihe database V2'),Path(r'.\analysis_project')]:
    for p in root.rglob('*'):
        if p.is_file():inventory.append(dict(Path=str(p),Bytes=p.stat().st_size,Mtime_ns=p.stat().st_mtime_ns))
pd.DataFrame(inventory).to_csv(OUT/'audit/Inputs_Before.csv',index=False)
ds=xr.open_dataset(DB/'L4_EcoFunction/Weihe_L4_EcoFunction_StateSpace_1000m_2000_2024_V3_1.nc')
dates=ds.time.values.astype('datetime64[D]');td=(dates-dates[0]).astype(int).astype(np.int32);shape=(ds.sizes['y'],ds.sizes['time'],ds.sizes['x']);ds.close()
np.save(OUT/'cache/dates.npy',dates);np.save(OUT/'cache/time_days.npy',td)
onset_path=OUT/'cache/Catalogue_Onsets_Union.npy'
if not (OUT/'cache/ONSETS_READY.json').exists():
    onset=np.lib.format.open_memmap(onset_path,mode='w+',dtype='bool',shape=shape);onset[:]=False
    rows=[];start=time.time()
    for et in ['CDH','HEAT','HR']:
        p=DB/'L5_EventEcoFunction/L5_2_EventLifecycle'/f'L5_2_{et}_Independent_Event_Catalog.csv.gz'
        n=0;bad=0
        for j,d in enumerate(pd.read_csv(p,usecols=['row','col','onset_index','termination_index'],chunksize=100000)):
            r=d.row.to_numpy(np.int32);c=d.col.to_numpy(np.int32);o=d.onset_index.to_numpy(np.int32);t=d.termination_index.to_numpy(np.int32)
            assert (r>=0).all() and (r<shape[0]).all() and (c>=0).all() and (c<shape[2]).all() and (o>=0).all() and (t<shape[1]).all() and (o<=t).all()
            onset[r,o,c]=True;n+=len(d)
            if j%30==0:print('catalogue',et,j,n,'sec',round(time.time()-start,1),flush=True)
        expected=json.loads((M1/'cache'/et/'READY.json').read_text())['rows'];assert n==expected
        rows.append(dict(Event=et,Rows=n,Matches_L53_Count=True,Source=str(p)))
    onset.flush();del onset
    pd.DataFrame(rows).to_csv(OUT/'audit/Catalogue_Onset_Counts.csv',index=False)
    (OUT/'cache/ONSETS_READY.json').write_text(json.dumps(dict(shape=shape,definition='union of existing L5.2 event onsets; next must be strictly after current termination index',rows=rows),indent=2),encoding='utf8')
# Reuse precisely the existing M1 block identity, mapped to pixels without reprojection.
for et in ['CDH','HEAT','HR']:
    row=np.load(M1/'cache'/et/'row.npy',mmap_mode='r');col=np.load(M1/'cache'/et/'col.npy',mmap_mode='r')
    for scale in [20,50]:
        b=np.load(M1/'cache'/et/f'blocks_{scale}.npy',mmap_mode='r');grid=np.full((shape[0],shape[2]),-1,np.int32)
        for start in range(0,len(row),100000):
            sl=slice(start,start+100000);grid[row[sl].astype(int),col[sl].astype(int)]=b[sl]
        np.save(OUT/'cache'/f'{et}_pixel_blocks_{scale}.npy',grid)
params=dict(seed=20261109,bootstrap_pilot=50,bootstrap_total=250,bootstrap_scales_km=[20,50],time_origin='existing L52/L53 climate termination timestamp (last active phase anchor)',primary_horizon='0 through 120 actual calendar days inclusive',recovery='first of 2 adjacent finite Acomp<=baseline observations; both strictly after stored ecological peak; confirmation strictly before next catalogue onset; within120',next_event='earliest existing catalogue onset strictly greater than current termination; union of hazards; ongoing episodes ignored',primary_missing='internal missing resets run; trailing missing censored at last finite pre-stop phase; no valid trajectory excluded at origin and separately counted',sensitivity_missing='censor at last valid observation before first missing scheduled phase',groups='High 75<=Q_full<90; Extreme 90<=Q_full<=100; float64 right-continuous upper-rank complete-hazard ECDF; ties unsplit',output=str(OUT),python=sys.executable)
(OUT/'code/configuration.json').write_text(json.dumps(params,indent=2),encoding='utf8')
print('M5 preparation ready',flush=True)
