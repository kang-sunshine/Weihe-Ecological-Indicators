# -*- coding: utf-8 -*-
from pathlib import Path
import json
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from pyproj import Transformer
O=Path(r'.\EI_Reviewer1_Major6_Revision');D=Path(r'.\Weihe database V3');M4=Path(r'.\EI_Reviewer1_Major4_Revision')
def catalogue(mask,T,SM,tq,sq,td):
 records=[]
 for col in range(mask.shape[1]):
  active=np.flatnonzero(mask[:,col])
  for seg in np.split(active,np.flatnonzero(np.diff(active)>2)+1):
   if not len(seg):continue
   h=np.maximum(0,T[seg,col]-tq[seg,col])/np.maximum(1,np.abs(tq[seg,col]));d=np.maximum(0,sq[seg,col]-SM[seg,col])/np.maximum(1e-8,np.abs(sq[seg,col]));delta=np.r_[np.diff(td),8][seg];severity=float(np.sum(np.sqrt(h*h+d*d)*delta));records.append(dict(Sample_Cell_Index=col,Onset_Index=int(seg[0]),Termination_Index=int(seg[-1]),Active_Phases=len(seg),Span_Phases=int(seg[-1]-seg[0]+1),Severity=severity))
 return pd.DataFrame(records)
def main():
 scale=json.loads((M4/'audit/Climate_Scale_Results.json').read_text());assert scale['Regular_Support_Partition_Exactly_Reproduced'];parent=np.load(M4/'cache/Reconstructed_ERA5_0p1_Cell_ID.npy');mask=parent>=0;grid=json.loads((M4/'audit/Grid_Metadata.json').read_text())['Axes'];lon=np.array(grid['x']);lat=np.array(grid['y']);tr=Transformer.from_crs(4326,json.loads((M4/'code/configuration.json').read_text())['Projection'],always_xy=True);xx,yy=np.meshgrid(lon,lat);reps=[]
 for pid in np.unique(parent[mask]):
  r,c=np.where(parent==pid);j=np.argmin((r-np.mean(r))**2+(c-np.mean(c))**2);xr,yr=tr.transform(lon[c[j]],lat[r[j]]);reps.append(dict(Parent_ID=int(pid),row=int(r[j]),col=int(c[j]),Longitude=lon[c[j]],Latitude=lat[r[j]],X_m=xr,Y_m=yr,Block_ix=int(np.floor(xr/50000)),Block_iy=int(np.floor(yr/50000))))
 reps=pd.DataFrame(reps);groups=reps.groupby(['Block_ix','Block_iy']).agg(X_m=('X_m','mean'),Y_m=('Y_m','mean'),N_Cells=('Parent_ID','size')).reset_index();groups=groups[groups.N_Cells>=5].reset_index(drop=True);xy=groups[['X_m','Y_m']].to_numpy();chosen=[int(np.argmin(np.sum((xy-xy.mean(axis=0))**2,axis=1)))]
 while len(chosen)<5:
  distance=np.min(np.stack([np.sum((xy-xy[i])**2,axis=1) for i in chosen]),axis=0);distance[chosen]=-1;chosen.append(int(np.argmax(distance)))
 sample=groups.iloc[chosen].copy();sample['Sample_Block']=np.arange(5);selected=reps.merge(sample[['Block_ix','Block_iy','Sample_Block']],on=['Block_ix','Block_iy']);selected.to_csv(O/'paired/Climate_Sample_Cells.csv',index=False);sample.to_csv(O/'paired/Climate_Sample_Blocks.csv',index=False)
 dates=pd.DatetimeIndex(np.load(O/'cache/dates.npy'));td=np.load(O/'cache/time_days.npy');phase=np.load(O/'cache/phase.npy');year=np.load(O/'cache/year.npy');raw=Dataset(D/'Weihe_L3_ClimateStress_1000m_2000_2024_V3_1_FINAL.nc');threshold=Dataset(D/'L5_EventEcoFunction/Weihe_L5_0_EventThresholds_2000_2020_V3_1.nc');native=Dataset(D/'L5_EventEcoFunction/Weihe_L5_0_EventMasks_1000m_2000_2024_V3_1.nc');results=[];allcat=[];hist=[];validation=[]
 for block,sub in selected.groupby('Sample_Block'):
  sub=sub.reset_index(drop=True);r0=int(sub.row.min());r1=int(sub.row.max()+1);c0=int(sub.col.min());c1=int(sub.col.max()+1);rr=sub.row.to_numpy()-r0;cc=sub.col.to_numpy()-c0;n=len(sub)
  def take(ds,name):return np.ma.filled(ds[name][:,r0:r1,c0:c1].astype(np.float64),np.nan)[:,rr,cc]
  T=take(raw,'Tmax');SM=take(raw,'RootSM');P=take(raw,'Precip');original=[take(threshold,name).astype(np.float32) for name in ['Tmax_q90','RootSM_q10','Precip_q95']];orig=[q[phase-1].copy() for q in original];loyo=[q.copy() for q in orig];early=[q.copy() for q in orig]
  for p in range(1,47):
   idx=np.flatnonzero(phase==p);base=np.flatnonzero((phase==p)&(year<=2020));be=np.flatnonzero((phase==p)&(year<=2010));future=np.flatnonzero((phase==p)&(year>=2011))
   for k,(values,q) in enumerate(zip([T,SM,P],[.9,.1,.95])):
    qb=np.nanquantile(values[base].astype(np.float32),q,axis=0).astype(np.float32);assert np.array_equal(qb,original[k][p-1]);qe=np.nanquantile(values[be].astype(np.float32),q,axis=0).astype(np.float32);early[k][future]=qe
    for target in base:
     ref=base[base!=target];loyo[k][target]=np.nanquantile(values[ref].astype(np.float32),q,axis=0).astype(np.float32)
    nb=np.isfinite(values[base]).sum(axis=0);ne=np.isfinite(values[be]).sum(axis=0);hist.append(dict(Sample_Block=block,Variable=['Tmax','RootSM','Precip'][k],Phase=p,N_Cells=n,Original_N_Years_Min=int(nb.min()),Original_N_Years_Max=int(nb.max()),Early_N_Years_Min=int(ne.min()),Early_N_Years_Max=int(ne.max()),Missing_Calendar_2000_Not_Filled=bool(p<=8)))
  masks=[(T>=q[0])&(SM<=q[1])&np.isfinite(T+SM+q[0]+q[1]) for q in [orig,loyo,early]];stored=take(native,'CDH_event')==1;assert np.array_equal(masks[0],stored);assert np.array_equal(masks[0][year>=2021],masks[1][year>=2021]);validation.append(dict(Block=block,N_Cells=n,N_Phases=len(year),Original_Mask_Mismatch=0,Original_Threshold_Mismatch=0,LOYO_Post2020_Phase_Mismatch=0))
  cats=[catalogue(m,T,SM,q[0],q[1],td) for m,q in zip(masks,[orig,loyo,early])]
  for variant,cat in zip(['ORIGINAL','LOYO_CLIMATE','EARLY_CLIMATE'],cats):
   cat['Variant']=variant;cat['Sample_Block']=block;cat['Onset_Year']=year[cat.Onset_Index.to_numpy(int)];cat['Population']='NEW_LOCAL_SAMPLE_REIDENTIFICATION; not fixed ecological response cohort';allcat.append(cat)
  for name,k,period,start,end in [('LOYO',1,'REF_2000_2020',2000,2020),('LOYO',1,'OUT_2021_2024',2021,2024),('EARLY',2,'APPLY_2011_2024',2011,2024)]:
   scope=(year>=start)&(year<=end);a=masks[0][scope];b=masks[k][scope];finite=np.isfinite(T[scope])&np.isfinite(SM[scope]);inter=int((a&b).sum());union=int((a|b).sum());ca=cats[0];cb=cats[k];ca=ca[(ca.Onset_Year>=start)&(ca.Onset_Year<=end)];cb=cb[(cb.Onset_Year>=start)&(cb.Onset_Year<=end)];sa=set(map(tuple,ca[['Sample_Cell_Index','Onset_Index','Termination_Index']].to_numpy()));sb=set(map(tuple,cb[['Sample_Cell_Index','Onset_Index','Termination_Index']].to_numpy()));shared=len(sa&sb)
   results.append(dict(Row_Type='NATIVE_PHASE_AND_NEW_EVENT_POPULATION',Hazard='CDH',Comparison=name,Period=period,Sample_Block=block,N_Climate_Cells=n,N_Valid_Cell_Phases=int(finite.sum()),Original_Active_Phases=int(a.sum()),Alternative_Active_Phases=int(b.sum()),Original_Active_Fraction=float(a.sum()/finite.sum()),Alternative_Active_Fraction=float(b.sum()/finite.sum()),Active_Phase_Jaccard=inter/union if union else np.nan,Gained_Active_Phases=int((b&~a).sum()),Lost_Active_Phases=int((a&~b).sum()),Original_Merged_Events=len(ca),Alternative_Merged_Events=len(cb),Exact_Interval_Matched_Events=shared,Merged_Event_Exact_Interval_Jaccard=shared/len(sa|sb) if len(sa|sb) else np.nan,Original_Severity_Median=float(ca.Severity.median()),Alternative_Severity_Median=float(cb.Severity.median()),Original_Severity_Q90=float(ca.Severity.quantile(.9)),Alternative_Severity_Q90=float(cb.Severity.quantile(.9)),Severity_Comparison='unpaired newly identified sample-event distributions; not fixed-original-catalogue ecological response pairing',Sampling='5occupied50km blocks selected geographically before threshold outcomes; sourcecell representatives, not121duplicate pixels',Status='COMPLETED_LOCAL_SAMPLE'))
  print('CLIMATE SAMPLE block',block,'cells',n,flush=True)
 raw.close();threshold.close();native.close();pd.DataFrame(results).to_csv(O/'M6_EventThreshold_Sensitivity.csv',index=False,encoding='utf-8-sig');pd.concat(allcat,ignore_index=True).to_parquet(O/'paired/Climate_Local_Reidentified_Events.parquet',index=False);pd.DataFrame(hist).to_csv(O/'supplement/Climate_Sample_Reference_Years.csv',index=False);pd.DataFrame(validation).to_csv(O/'audit/Climate_Threshold_Reproduction.csv',index=False);(O/'cache/CLIMATE_READY.json').write_text(json.dumps(dict(N_Blocks=5,N_Source_Cells=len(selected),All_Available_Years=True,Full_Region_Reidentification=False,New_Sample_Events_Separate_From_Fixed_Catalogue=True),indent=2),encoding='utf8');print('CLIMATE SAMPLE COMPLETE',len(selected),'cells',flush=True)
if __name__=='__main__':main()
