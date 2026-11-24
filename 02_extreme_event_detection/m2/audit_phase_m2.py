# -*- coding: utf-8 -*-
from pathlib import Path
import ast,json,hashlib,gc,time,sys
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset,date2num
OUT=Path(r'.\EI_Reviewer1_Major2_Revision');DB=Path(r'.\Weihe database V3');SRC=Path(r'.\analysis_project\analysis_code')
for n in ['audit','cache','catalogues','sensitivity']: (OUT/n).mkdir(parents=True,exist_ok=True)
inventory=[]
for root in [DB,Path(r'.\Weihe database V2'),Path(r'.\analysis_project'),Path(r'.\EI_Reviewer1_Major1_Revision'),Path(r'.\EI_Reviewer1_Major5_Revision'),Path(r'.\EI_Figure6_Redraw_DataPack')]:
    for p in root.rglob('*'):
        if p.is_file():inventory.append(dict(Path=str(p),Bytes=p.stat().st_size,Mtime_ns=p.stat().st_mtime_ns))
pd.DataFrame(inventory).to_csv(OUT/'audit/Inputs_Before.csv',index=False)
names=['Step L5.0 - Build Event Masks from L3.py','Step_L5_2_Reconstruct_Event_Lifecycles_Robust_V1.py','Step_L5_3_Build_Event_Severity_Resilience_Metrics_Robust_V1.py','Step_L5_3_Build_Event_Severity_Resilience_Metrics_Robust_V2_FAST.py','Build_Weihe_L3_V3_1_Final_NetCDF.py']
locations=[];manifest=[]
interesting={'phase8_from_dates','reconstruct_segments','climate_severity_for_event','calculate_row_events','actual_days','future_window_end_index'}
variables={'heat_p','cdh_p','hr_p','tmax_q90','rootsm_q10','precip_q95','heat_rel','dry_rel','intensity','cumulative','BASELINE_START','BASELINE_END','PRECIP_MIN_MM','MAX_GAP_PHASES','MIN_ACTIVE_PHASES','EPS','dt','htden','smden'}
for name in names:
    p=SRC/name;raw=p.read_bytes();s=raw.decode('utf-8-sig');(OUT/'audit'/name).write_bytes(raw);tree=ast.parse(s)
    manifest.append(dict(Source=str(p),Snapshot=str(OUT/'audit'/name),SHA256=hashlib.sha256(raw).hexdigest()))
    for n in ast.walk(tree):
        symbol=None
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in interesting:symbol='function '+n.name
        if isinstance(n,ast.Assign):
            targets=[t.id for t in n.targets if isinstance(t,ast.Name)]
            if any(t in variables for t in targets):symbol='assignment '+','.join(targets)
        if symbol:
            lines=s.splitlines();locations.append(dict(Source=str(p),Symbol=symbol,Start_Line=n.lineno,End_Line=n.end_lineno,Excerpt='\n'.join(lines[n.lineno-1:min(n.end_lineno,n.lineno+18)])))
pd.DataFrame(locations).to_csv(OUT/'M2_Source_Code_Locations.csv',index=False,encoding='utf-8-sig');pd.DataFrame(manifest).to_csv(OUT/'audit/Source_Code_Manifest.csv',index=False)
cl=xr.open_dataset(DB/'Weihe_L3_ClimateStress_1000m_2000_2024_V3_1_FINAL.nc');ms=xr.open_dataset(DB/'L5_EventEcoFunction/Weihe_L5_0_EventMasks_1000m_2000_2024_V3_1.nc');th=xr.open_dataset(DB/'L5_EventEcoFunction/Weihe_L5_0_EventThresholds_2000_2020_V3_1.nc')
dates=pd.DatetimeIndex(cl.time.values);td=np.asarray((dates-dates[0]).days,np.int32);nt=len(td);ny=cl.sizes['y'];nx=cl.sizes['x'];shape=(ny,nt,nx)
assert np.array_equal(cl.time.values,ms.time.values) and cl.sizes['y']==ms.sizes['y'] and cl.sizes['x']==ms.sizes['x']
phase=np.minimum((dates.dayofyear.to_numpy()-1)//8,45).astype(int)
dt=np.r_[np.diff(td),8].astype(int);assert (dt>0).all()
np.save(OUT/'cache/dates.npy',dates.values.astype('datetime64[D]'));np.save(OUT/'cache/time_days.npy',td);np.save(OUT/'cache/phase_dt_days.npy',dt)
meta={k:dict(attrs=dict(d.attrs),variables={v:dict(d[v].attrs) for v in d.variables},sizes=dict(d.sizes)) for k,d in [('L3',cl),('Masks',ms),('Thresholds',th)]}
(OUT/'audit/NetCDF_Metadata.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False,default=str),encoding='utf8')
params=dict(shape=shape,original_event_types=['CDH','HEAT','HR'],nominal_phase_days=8,actual_dt_counts={int(k):int(v) for k,v in zip(*np.unique(np.diff(td),return_counts=True))},start=str(dates[0]),end=str(dates[-1]),baseline='2000-2020 inclusive by phase',block_y=8,python=sys.executable)
(OUT/'code/configuration.json').write_text(json.dumps(params,indent=2),encoding='utf8')
maps={et:np.lib.format.open_memmap(OUT/'cache'/f'native_{et}.npy',mode='w+',dtype='bool',shape=shape) for et in ['CDH','HEAT','HR']}
unknown=np.lib.format.open_memmap(OUT/'cache/heat_unknown_soil.npy',mode='w+',dtype='bool',shape=shape)
original_counts=pd.read_csv(DB/'L5_EventEcoFunction/L5_0_EventMask_Counts.csv');print(original_counts.to_string(index=False),flush=True)
# Store only the ~8M active-CDH component tuples for exact distribution diagnostics.
ncdh=int(original_counts.loc[original_counts.event_variable.eq('CDH_event'),'event_cells'].iloc[0])
components=np.lib.format.open_memmap(OUT/'cache/CDH_Active_Phase_Components.npy',mode='w+',dtype='float64',shape=(ncdh,8));pos=0
counters={k:0 for k in ['all_grid_phases','valid_H','valid_C','valid_R','H','C','R','HC','HR','CR','HCR','C_not_H','H_only','H_only_known_soil','H_only_unknown_soil','H_mismatch','C_mismatch','R_mismatch','p95_candidates','p95_removed_by_floor','p95_lt1_observed','HR_under_low_threshold','tden_floor_C','smden_epsilon_C','zero_joint_C']}
dominance={k:0 for k in ['valid_joint','heat_ge90','heat_ge99','dry_ge90','dry_ge99','heat_gt_dry','dry_gt_heat','equal_components']};threshold_ranges=[]
weighted={k:0. for k in ['phase_days','heat_squared_share_days','valid_joint_days']}
outnc=Dataset(str(OUT/'sensitivity/HEAT_Only_Native_Phases.nc'),'w');outnc.createDimension('time',nt);outnc.createDimension('y',ny);outnc.createDimension('x',nx)
tvar=outnc.createVariable('time','i4',('time',));tvar.units='days since 1970-01-01';tvar.calendar='standard';tvar[:]=date2num(dates.to_pydatetime(),tvar.units,tvar.calendar)
for n in ['y','x']:
    v=outnc.createVariable(n,'f8',(n,));v[:]=cl[n].values;v.units='degrees_north' if n=='y' else 'degrees_east'
vonly=outnc.createVariable('HEAT_only_mask_difference','i1',('time','y','x'),zlib=True,complevel=4,chunksizes=(16,8,min(nx,352)));vknown=outnc.createVariable('HEAT_only_known_soil','i1',('time','y','x'),zlib=True,complevel=4,chunksizes=(16,8,min(nx,352)))
outnc.definition='Existing HEAT & NOT existing CDH, sensitivity only; known_soil additionally requires finite RootSM and q10; no new event runs detected.'
start=time.time()
for y0 in range(0,ny,8):
    y1=min(y0+8,ny);sel=dict(y=slice(y0,y1))
    T=cl.Tmax.isel(**sel).transpose('time','y','x').values;SM=cl.RootSM.isel(**sel).transpose('time','y','x').values;P=cl.Precip.isel(**sel).transpose('time','y','x').values
    tq=th.Tmax_q90.isel(**sel).transpose('phase','y','x').values;sq=th.RootSM_q10.isel(**sel).transpose('phase','y','x').values;pq=th.Precip_q95.isel(**sel).transpose('phase','y','x').values
    for name,arr in [('Tq90',tq),('SMq10',sq),('Pq95',pq)]:
        vals=arr[np.isfinite(arr)];threshold_ranges.append(dict(Variable=name,N=len(vals),Sum=float(vals.sum(dtype='float64')),Min=float(vals.min()) if len(vals) else np.nan,Max=float(vals.max()) if len(vals) else np.nan))
    tq=tq[phase];sq=sq[phase];pq=pq[phase]
    C=ms.CDH_event.isel(**sel).transpose('time','y','x').fillna(0).values>0;H=ms.HEAT_event.isel(**sel).transpose('time','y','x').fillna(0).values>0;R=ms.HR_event.isel(**sel).transpose('time','y','x').fillna(0).values>0
    for et,v in [('CDH',C),('HEAT',H),('HR',R)]:maps[et][y0:y1]=v.transpose(1,0,2)
    validH=np.isfinite(T)&np.isfinite(tq);validS=np.isfinite(SM)&np.isfinite(sq);validR=np.isfinite(P)&np.isfinite(pq);preR=validR&(P>=pq)
    only=H&~C;uk=H&~validS;unknown[y0:y1]=uk.transpose(1,0,2);vonly[:,y0:y1,:]=only;vknown[:,y0:y1,:]=only&validS
    vals=dict(all_grid_phases=C.size,valid_H=validH.sum(),valid_C=(validH&validS).sum(),valid_R=validR.sum(),H=H.sum(),C=C.sum(),R=R.sum(),HC=(H&C).sum(),HR=(H&R).sum(),CR=(C&R).sum(),HCR=(H&C&R).sum(),C_not_H=(C&~H).sum(),H_only=only.sum(),H_only_known_soil=(only&validS).sum(),H_only_unknown_soil=(only&~validS).sum(),H_mismatch=(H!=(validH&(T>=tq))).sum(),C_mismatch=(C!=(validH&validS&(T>=tq)&(SM<=sq))).sum(),R_mismatch=(R!=(preR&(P>=1))).sum(),p95_candidates=preR.sum(),p95_removed_by_floor=(preR&(P<1)).sum(),p95_lt1_observed=(validR&(pq<1)).sum(),HR_under_low_threshold=(R&(pq<1)).sum())
    for k,v in vals.items():counters[k]+=int(v)
    tx=T[C].astype(float);sx=SM[C].astype(float);tt=tq[C].astype(float);ss=sq[C].astype(float)
    he=np.maximum(tx-tt,0);de=np.maximum(ss-sx,0);denh=np.maximum(np.abs(tt),1);dend=np.maximum(np.abs(ss),1e-8);h=he/denh;d=de/dend;norm=h*h+d*d;good=norm>0;share=np.divide(h*h,norm,out=np.full(len(h),np.nan),where=good)
    days=np.broadcast_to(dt[:,None,None],C.shape)[C]
    assert pos+len(h)<=ncdh;components[pos:pos+len(h)]=np.column_stack([he,de,denh,dend,h,d,share,days]);pos+=len(h)
    counters['tden_floor_C']+=int((np.abs(tt)<1).sum());counters['smden_epsilon_C']+=int((np.abs(ss)<1e-8).sum());counters['zero_joint_C']+=int((~good).sum())
    for k,value in dict(valid_joint=good.sum(),heat_ge90=(share>=.9).sum(),heat_ge99=(share>=.99).sum(),dry_ge90=(share<=.1).sum(),dry_ge99=(share<=.01).sum(),heat_gt_dry=(h[good]>d[good]).sum(),dry_gt_heat=(d[good]>h[good]).sum(),equal_components=(d[good]==h[good]).sum()).items():dominance[k]+=int(value)
    weighted['phase_days']+=float(days.sum());weighted['heat_squared_share_days']+=float(np.nansum(share*days));weighted['valid_joint_days']+=float(days[good].sum())
    if y0%40==0:print('native audit rows',y0,'CDH',pos,'seconds',round(time.time()-start,1),flush=True)
    del T,SM,P,tq,sq,pq,H,C,R,validH,validS,validR,preR,only,uk,tx,sx,tt,ss,he,de,denh,dend,h,d,norm,share,days;gc.collect()
assert pos==ncdh
outnc.close();components.flush()
for m in maps.values():m.flush()
unknown.flush();pd.DataFrame(threshold_ranges).to_csv(OUT/'audit/Threshold_Block_Statistics.csv',index=False)
# Exact threshold distribution on finite phase-pixel cells; this is not a distribution of daily rain.
qq=th.Precip_q95.values;finite=qq[np.isfinite(qq)].astype(float);qv=np.quantile(finite,[0,.01,.05,.25,.5,.75,.95,.99,1]);np.save(OUT/'cache/HR_P95_Finite_Values.npy',finite)
records=[dict(Measure='P95_'+str(k),Value=float(v),Numerator=np.nan,Denominator=len(finite),Definition='finite baseline phase-pixel P95 threshold, mm per phase') for k,v in zip(['min','q01','q05','q25','q50','q75','q95','q99','max'],qv)]
for name,n,d,definition in [('P95_below1_fraction',int((finite<1).sum()),len(finite),'threshold pairs where max(P95,1mm) is increased'),('P95_equal1_fraction',int((finite==1).sum()),len(finite),'1mm floor equals threshold'),('Observed_valid_phases_under_P95lt1',counters['p95_lt1_observed'],counters['valid_R'],'observed valid precipitation phase-pixels with threshold below1'),('P95_candidates_removed_by_1mm',counters['p95_removed_by_floor'],counters['p95_candidates'],'P>=P95 candidates excluded because P<1'),('Realized_HR_under_P95lt1',counters['HR_under_low_threshold'],counters['R'],'actual HR phases whose local threshold is below1')]:records.append(dict(Measure=name,Value=n/d if d else np.nan,Numerator=n,Denominator=d,Definition=definition))
pd.DataFrame(records).to_csv(OUT/'M2_HR_P95_And_1mm_Floor.csv',index=False,encoding='utf-8-sig')
columns=['heat_excess_C','soil_deficit_m3_m3','heat_denominator_C','soil_denominator_m3_m3','heat_relative','soil_relative','heat_squared_share','phase_duration_days'];dist=[]
for j,name in enumerate(columns):
    v=np.asarray(components[:,j]);valid=v[np.isfinite(v)];qv=np.quantile(valid,[0,.01,.05,.25,.5,.75,.95,.99,1]);dist.append(dict(Variable=name,N=len(valid),N_missing=int(len(v)-len(valid)),Mean=float(valid.mean()),SD=float(valid.std()),**{n:float(value) for n,value in zip(['Min','Q01','Q05','Q25','Median','Q75','Q95','Q99','Max'],qv)},Population='all existing native CDH pixel-phases'))
pd.DataFrame(dist).to_csv(OUT/'M2_CDH_Component_Distributions.csv',index=False,encoding='utf-8-sig')
dom=[dict(Measure=k,Count=v,Denominator=dominance['valid_joint'],Fraction=v/dominance['valid_joint'] if dominance['valid_joint'] else np.nan) for k,v in dominance.items()]
dom.append(dict(Measure='zero_joint_intensity',Count=counters['zero_joint_C'],Denominator=counters['C'],Fraction=counters['zero_joint_C']/counters['C']))
dom.append(dict(Measure='phase_day_weighted_heat_squared_share',Count=weighted['heat_squared_share_days'],Denominator=weighted['valid_joint_days'],Fraction=weighted['heat_squared_share_days']/weighted['valid_joint_days']))
pd.DataFrame(dom).to_csv(OUT/'M2_CDH_Component_Dominance.csv',index=False,encoding='utf-8-sig')
(OUT/'cache/NATIVE_READY.json').write_text(json.dumps(dict(counters=counters,shape=shape,component_columns=columns,component_n=pos,dominance=dominance,weighted=weighted),indent=2),encoding='utf8')
cl.close();ms.close();th.close();print('NATIVE AUDIT READY',counters,flush=True)
