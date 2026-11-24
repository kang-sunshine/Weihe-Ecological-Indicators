# -*- coding: utf-8 -*-
from pathlib import Path
import json,ast,time,gc
import numpy as np
import pandas as pd
from pyproj import Transformer
O=Path(r'.\EI_Reviewer1_Major8_Revision');R=O.parent;M1=R/'EI_Reviewer1_Major1_Revision';M4=R/'EI_Reviewer1_Major4_Revision'
def main():
 assert (O/'cache/M8B/LANDCOVER_READY.json').exists();start=time.time();lc=np.load(O/'cache/M8B/Landcover_Cohorts.npz');group=lc['Main_Group'];mode=lc['Main_Mode'];pur=lc['Main_Purity'];exact=lc['Exact_ThreePeriod'];broad=lc['Broad_ThreePeriod'];domain=np.load(M4/'cache/Climate_Valid_Mask.npy');area=np.load(O/'cache/Pixel_Area_km2.npy');grid=json.loads((O/'audit/Target_Grid.json').read_text());cfg=json.loads((O/'code/M8B_configuration.json').read_text());proj=json.loads((O/'code/configuration.json').read_text())['Projection'];x,y=Transformer.from_crs(4326,proj,always_xy=True).transform(*np.meshgrid(grid['X'],grid['Y']));b50=np.floor(x/50000).astype(np.int64)*10000+np.floor(y/50000).astype(np.int64);b100=np.floor(x/100000).astype(np.int64)*10000+np.floor(y/100000).astype(np.int64);clim=np.load(M4/'cache/Reconstructed_ERA5_0p1_Cell_ID.npy');ub,bi=np.unique(b50[domain],return_inverse=True);blockgrid=np.full(domain.shape,-1,np.int16);blockgrid[domain]=bi;np.savez_compressed(O/'cache/M8B/Spatial_Grid.npz',Block50=b50,Block100=b100,Climate=clim,Block50_Index=blockgrid,Block50_Labels=ub)
 src=(M1/'audit/Step_L5_4A_Severity_Response_Thresholds_FAST.py').read_text(encoding='utf-8-sig');node=next(v for v in ast.parse(src).body if isinstance(v,ast.FunctionDef) and v.name=='bounded_sample');env={'np':np,'RNG':np.random.default_rng(20260908)};exec(compile(ast.Module(body=[node],type_ignores=[]),'<original bounded_sample>','exec'),env)
 support=[];qc=[];model=pd.read_parquet(O/'paired/Frozen_Model_Pixels_With_Terrain.parquet');mp=np.zeros(domain.shape,bool);mp[model.row.to_numpy(int),model.col.to_numpy(int)]=True
 for h in ['CDH','HEAT','HR']:
  cache=M1/'cache'/h;meta=json.loads((cache/'READY.json').read_text());N=meta['rows'];folder=O/'cache/M8B'/h;folder.mkdir(exist_ok=True);rawgroup=np.full(N,255,np.uint8);raws=np.empty(N,np.float64);rawpixcounts=np.zeros(domain.size,np.int64);cum=0;sample=np.empty(0)
  files=sorted(cache.glob('raw_severity_*.npz'))
  for j in range(0,len(files),10):
   zz=[np.load(f) for f in files[j:j+10]];s=np.concatenate([z['severity'] for z in zz]);clean=np.concatenate([z['clean'] for z in zz]);raws[cum:cum+len(s)]=s;cum+=len(s);sample=env['bounded_sample'](sample,s[clean&np.isfinite(s)],500000)
  assert cum==N;edges=np.unique(np.quantile(sample,np.linspace(0,1,81)));edges[0]=-np.inf;edges[-1]=np.inf;np.save(folder/'Original_Historical_Bin_Edges.npy',edges)
  if h=='CDH':assert np.allclose(edges,np.load(M4/'paired/CDH_Original_Historical_Bin_Edges.npy'),rtol=0,atol=1e-13)
  for f in sorted((R/f'EI_Reviewer1_Major2_Revision/catalogues/{h}').glob('*.parquet')):
   df=pd.read_parquet(f,columns=['Event_Num','row','col']);ids=df.Event_Num.to_numpy(int)-1;r=df.row.to_numpy(int);c=df.col.to_numpy(int);assert (rawgroup[ids]==255).all();rawgroup[ids]=group[r,c];rawpixcounts+=np.bincount(r*group.shape[1]+c,minlength=domain.size)
  assert (rawgroup!=255).all();np.save(folder/'Full_Event_Group.npy',rawgroup)
  # Full ECDF reference is ALL finite original events at corresponding2001type,
  # not the clean or positive-response subset. Ties right-continuous float64.
  for gid in [0,1,2,3,4]:np.save(folder/f'Full_Severity_Group{gid}.npy',np.sort(raws[(rawgroup==gid)&np.isfinite(raws)]))
  # Exact-class consistency/CDH and purecrop+mosaic sensitivity remain distinct.
  if h=='CDH':
   maskraw=np.zeros(N,bool);mosaic=np.zeros(N,bool)
   for f in sorted((R/f'EI_Reviewer1_Major2_Revision/catalogues/{h}').glob('*.parquet')):
    df=pd.read_parquet(f,columns=['Event_Num','row','col']);ids=df.Event_Num.to_numpy(int)-1;r=df.row.to_numpy(int);c=df.col.to_numpy(int);maskraw[ids]=exact[r,c];mosaic[ids]=np.isin(mode[r,c],[12,14])
   for gid in [1,2,3]:np.save(folder/f'Full_Severity_Exact3_Group{gid}.npy',np.sort(raws[(rawgroup==gid)&maskraw&np.isfinite(raws)]))
   np.save(folder/'Full_Severity_Crop12plus14.npy',np.sort(raws[mosaic&np.isfinite(raws)]))
   del maskraw,mosaic
  eventnum=np.concatenate([pd.read_parquet(p,columns=['event_num']).event_num.to_numpy(np.int64) for p in sorted(cache.glob('clean_*.parquet'))]);r=np.load(cache/'row.npy',mmap_mode='r').astype(np.int32);c=np.load(cache/'col.npy',mmap_mode='r').astype(np.int32);s=np.load(cache/'climate_cumulative_severity.npy',mmap_mode='r');damage=np.load(cache/'Acomp_damage.npy',mmap_mode='r');legacy=np.load(cache/'Acomp_legacy_anomaly_days.npy',mmap_mode='r');vr=np.load(cache/'valid_resistance_sample.npy',mmap_mode='r');clean_g=group[r,c];assert np.array_equal(clean_g,rawgroup[eventnum-1]);assert np.array_equal(s,raws[eventnum-1]);np.save(folder/'Clean_Event_Num.npy',eventnum);np.save(folder/'Clean_Group.npy',clean_g);counts={'RAW_ORIGINAL_CATALOGUE':rawpixcounts.reshape(domain.shape),'CLEAN_PRE_FINITE_SEVERITY':np.bincount(r*703+c,minlength=domain.size).reshape(domain.shape)}
  for name,m in [('FINITE_DAMAGE_RESPONSE',np.isfinite(damage)),('POSITIVE_VALID_DAMAGE',np.isfinite(damage)&(damage>0)&(vr==1)),('FINITE_HISTORICAL_LEGACY',np.isfinite(legacy))]:counts[name]=np.bincount(r[m].astype(int)*703+c[m],minlength=domain.size).reshape(domain.shape)
  np.savez_compressed(folder/'Event_Pixel_Counts.npz',**counts)
  groupdefs=[(str(gid),str(name),group==int(gid)) for gid,name in cfg['Groups'].items()]+[(f'IGBP_{k}',f'Other_IGBP_{k}',mode==k) for k in [8,9,11,13,14,15,16,17]]+[('0','NoData_or_Unclassified',mode==0)]
  for label,name,gm in groupdefs:
   for cohort,extra in [('MAIN_2001',np.ones(domain.shape,bool)),('THREE_PERIOD_EXACT_CLASS',exact),('THREE_PERIOD_BROAD_GROUP',broad),('PURITY_GE0p75',pur>=.75)]:
    m=domain&gm&extra;npx=int(m.sum())
    for response,v in counts.items():
     present=m&(v>0);values=v[m];bb,ww=np.unique(b50[m],return_counts=True) if npx else ([],[])
     event_blocks=np.bincount(blockgrid[present],weights=v[present],minlength=len(ub)) if present.any() else np.zeros(len(ub));total=event_blocks.sum();hhi=np.sum((event_blocks/total)**2) if total else np.nan
     support.append(dict(Event=h,Vegetation=name,Group_Label=label,Cohort=cohort,Metric=response,Landcover_Year=2001,N_Type_Domain_Pixels=npx,Type_Footprint_Area_km2=float(area[m].sum()),Type_Area_Fraction=float(area[m].sum()/area[domain].sum()),N_Events=int(values.sum()),N_Response_Pixels=int(present.sum()),N_Formal_Model_Pixels=int((m&mp).sum()),N_Distributed_Climate_Support_Cells=int(len(np.unique(clim[present]))) if present.any() else 0,N_50km_Blocks=int(len(np.unique(b50[present]))) if present.any() else 0,N_100km_Blocks=int(len(np.unique(b100[present]))) if present.any() else 0,Max_50km_Event_Share=float(event_blocks.max()/total) if total else np.nan,Effective_50km_Support_InvHHI=1/hhi if hhi>0 else np.nan,N_Mixed_Purity_Lt0p5=int((m&(pur<.5)).sum()),Mixed_Fraction=float((m&(pur<.5)).sum()/npx) if npx else np.nan,Status='COMPUTED' if values.sum()>0 else 'NO_EVENTS',Support_Definition='ERA5 reconstructed0.1degree distributedsupport, not independent nativeGaussian observations; spatial blocks from M4 projection',Response_Definition='clean-pre+finiteS; probability finiteDamage; magnitude valid_resistance1&positiveDamage; legacy finite originalL5.3legacy; raw catalog separate'))
  # Exact original per-bin event counts independently validate reconstructed edges.
  saved=pd.read_csv(R/'Weihe database V3/L5_EventEcoFunction/L5_4_ThresholdAnalysis/L5_4A_Severity_DamageProbability_Binned.csv');saved=saved[saved.event_type==h];bn=np.clip(np.searchsorted(edges,s,side='right')-1,0,len(edges)-2);ns=np.bincount(bn[np.isfinite(damage)],minlength=len(edges)-1);assert all(ns[int(z.bin_id)]==z.n for z in saved.itertuples())
  qc.append(dict(Event=h,N_Raw=N,N_Clean=len(eventnum),Raw_ECDF_Mapped_By_Event_ID=True,Clean_ID_Severity_And_Type_Exact=True,N_Original_Bins=len(edges)-1,Original_Probability_Bin_Counts_Reproduced=True,N_Raw_Class_NoData=int((rawgroup==0).sum())));print('EVENTS',h,N,len(eventnum),'sec',round(time.time()-start,1),flush=True);del raws,rawgroup,r,c,s,damage,legacy,vr,eventnum,clean_g,counts;gc.collect()
 pd.DataFrame(support).to_csv(O/'M8_Landcover_Support.csv',index=False,encoding='utf-8-sig');pd.DataFrame(qc).to_csv(O/'audit/M8B/Event_Cohort_And_Bin_Reproduction.csv',index=False);(O/'cache/M8B/EVENTS_READY.json').write_text(json.dumps(dict(N_Hazards=3,N_Support_Rows=len(support),No_Event_Reidentification=True,Seconds=time.time()-start),indent=2),encoding='utf8');print('EVENTS READY',flush=True)
if __name__=='__main__':main()
