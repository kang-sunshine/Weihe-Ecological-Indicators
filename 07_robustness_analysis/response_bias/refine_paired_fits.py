# -*- coding: utf-8 -*-
"""Preserve historical bounded replacement sampling for paired bin summaries."""
import json
import numpy as np
import pandas as pd
from core import *
from statistics_legacy import fit_segmented

def bounded_matrix(old,new,rng,maximum=20000):
 if len(new)>maximum:new=new[rng.choice(len(new),maximum,replace=False)]
 if not len(old):return new.copy()
 both=np.concatenate([old,new]);return both if len(both)<=maximum else both[rng.choice(len(both),maximum,replace=False)]

def main():
 assert (O/'qc/C_READY.json').exists()
 if (O/'qc/PAIRED_FIT_READY.json').exists():return
 registry=pd.read_csv(M1/'R1M1_Original_Breakpoint_Registry.csv');reference=pd.read_csv(DB/'L5_EventEcoFunction/L5_4_ThresholdAnalysis/L5_4A_Severity_Response_Binned.csv');edges=pd.read_csv(O/'qc/Historical_80Bin_Edges.csv');out=[];checks=[]
 for hi,h in enumerate(HAZ):
  reg=registry[(registry.Event==h)&registry.Response.eq('Acomp_legacy')].iloc[0];src=reference[(reference.event_type==h)&reference.response.eq('Acomp_legacy')];v=fit_segmented(src.severity_median.to_numpy(),src.response_median.to_numpy(),src.n.to_numpy(),reg.Search_Min,reg.Search_Max)
  errS=abs(v['S_star']-reg.S_star);errA=abs(v['Delta_AIC']-reg.Delta_AIC);assert errS<1e-10 and errA<1e-8
  checks.append(dict(Hazard=h,Historical_S_Difference=errS,Historical_DeltaAIC_Difference=errA,Pass=True))
  e=events(h);c=np.load(O/'qc/cache'/f'{h}_Legacy_Audit.npy',mmap_mode='r');s=e.climate_cumulative_severity.to_numpy();old=e.Acomp_legacy_anomaly_days.to_numpy();edge=edges[edges.Hazard==h].sort_values('Edge').S.to_numpy();bi=np.clip(np.searchsorted(edge,s,side='right')-1,0,len(edge)-2);sourcechunk=(e.event_num.to_numpy(np.int64)-1)//1000000;full=np.load(M1/'cache'/h/'all_severity_sorted.npy',mmap_mode='r');b,_=blockdata(h);pix=e.row.to_numpy(np.int32)*703+e.col.to_numpy(np.int32)
  for H in [32,56]:
   j=WINDOWS.index(H);paired=np.isfinite(old)&(c[:,20+j]==0)&np.isfinite(c[:,12+j]);data=[];rng=np.random.default_rng(SEED+hi*1000+H)
   for k in range(len(edge)-1):
    ix=np.flatnonzero(paired&(bi==k));n=len(ix)
    if n<500:continue
    pool=np.empty((0,3))
    for chunk in np.unique(sourcechunk[ix]):
     ids=ix[sourcechunk[ix]==chunk];new=np.column_stack([s[ids],old[ids],c[ids,12+j]]);pool=bounded_matrix(pool,new,rng)
    med=np.median(pool,axis=0);data.append((med[0],med[1],med[2],n))
   data=np.array(data)
   for name,col in [('Historical_L_on_same_window_subset',1),('Fixed_Window_L',2)]:
    f=fit_segmented(data[:,0],data[:,col],data[:,3],reg.Search_Min,reg.Search_Max) if len(data) else dict(Status='NOT ESTIMABLE',Reason='no bin with500paired events')
    if 'S_star' in f:f['Q_Full']=100*np.searchsorted(full,f['S_star'],side='right')/len(full)
    out.append(dict(Hazard=h,Window_Days=H,Response=name,Historical_S_star=reg.S_star,Historical_Delta_AIC=reg.Delta_AIC,N_Paired_Events=int(paired.sum()),N_Paired_Pixels=np.unique(pix[paired]).size,N_Paired_Blocks=np.unique(b[paired]).size,Bin_Estimator='original bounded replacement,max20000,original1Msourcechunk order; same sampledIDs forpaired x/original/fixed',Seed=SEED+hi*1000+H,Sampling_Limit='paired adaptation usesjointIDs instead ofindependent marginalx/y reservoirs; not claimedexacthistoricalfitonchangedcohort',**f))
 pd.DataFrame(out).to_csv(O/'outputs/C3_Paired_Breakpoint_Fits.csv',index=False,encoding='utf-8-sig');pd.DataFrame(checks).to_csv(O/'qc/Segmented_Implementation_Reproduction.csv',index=False)
 (O/'qc/PAIRED_FIT_READY.json').write_text(json.dumps(dict(Original_Parameter_Counts_And_Weights_Reproduced=True,Original_Edges_And_S_Constraints_Retained=True,Paired_Bounded_Response_Sampling=True,No_Breakpoint_Bootstrap_Repeated=True)),encoding='utf8')
if __name__=='__main__':main()
