# -*- coding: utf-8 -*-
"""Separate observation-window effects from different per-window sample selection."""
import json,gc
import numpy as np
import pandas as pd
from core import *
from statistics_legacy import fit_segmented
from refine_paired_fits import bounded_matrix
def main():
 assert (O/'qc/PAIRED_FIT_READY.json').exists()
 if (O/'qc/COMMON_WINDOW_READY.json').exists():return
 outbins=pd.read_csv(O/'outputs/C3_SeverityLegacy_Sensitivity.csv');outfits=pd.read_csv(O/'outputs/C3_Paired_Breakpoint_Fits.csv');fixed=pd.read_csv(O/'outputs/C2_FixedWindow_Legacy.csv');outbins['Comparison_Cohort']=np.where(outbins.Window_Days==0,'ORIGINAL_FULL','PAIR_WITH_EACH_WINDOW');outfits['Comparison_Cohort']='PAIR_WITH_EACH_WINDOW';fixed['Comparison_Cohort']='EACH_WINDOW_COMPLETE_COHORT';br=[];fr=[];cr=[];den=[];edges=pd.read_csv(O/'qc/Historical_80Bin_Edges.csv');reg=pd.read_csv(M1/'R1M1_Original_Breakpoint_Registry.csv')
 for hi,h in enumerate(HAZ):
  e=events(h);c=np.load(O/'qc/cache'/f'{h}_Legacy_Audit.npy',mmap_mode='r');old=e.Acomp_legacy_anomaly_days.to_numpy();common=np.isfinite(old)
  for j in [0,1,2]:common&=(c[:,20+j]==0)&np.isfinite(c[:,12+j])
  b,W=blockdata(h);g=groups(h);pix=e.row.to_numpy(np.int32)*703+e.col.to_numpy(np.int32);severity=e.climate_cumulative_severity.to_numpy();edge=edges[edges.Hazard==h].sort_values('Edge').S.to_numpy();bi=np.clip(np.searchsorted(edge,severity,side='right')-1,0,len(edge)-2);full=np.load(M1/'cache'/h/'all_severity_sorted.npy',mmap_mode='r');r=reg[reg.Event.eq(h)&reg.Response.eq('Acomp_legacy')].iloc[0];sourcechunk=(e.event_num.to_numpy(np.int64)-1)//1000000
  den.append(dict(Hazard=h,Comparison_Cohort='COMMON_16_32_56',N_Common=int(common.sum()),N_Clean=len(e),N_Pixels=np.unique(pix[common]).size,N_Blocks=np.unique(b[common]).size))
  for H in [0,16,32,56]:
   j=WINDOWS.index(H) if H else 0;L=old if H==0 else c[:,12+j]
   for k in range(len(edge)-1):
    mask=common&(bi==k);n=int(mask.sum())
    if not n:continue
    sm=float(np.median(severity[mask]));br.append(dict(Hazard=h,Window_Days=H,Bin=k,S_Lower=edge[k],S_Upper=edge[k+1],S_Median=sm,Q_Full=100*np.searchsorted(full,sm,side='right')/len(full),Original_L_Median=float(np.median(old[mask])),Fixed_L_Median=float(np.median(L[mask])),Comparison_Cohort='COMMON_16_32_56',**support(mask,pix,b),Status='SUPPORTED_BIN' if n>=500 else 'BELOW_ORIGINAL_MIN500',Cohort='identical eventIDs acrossoriginal/16/32/56'))
   if H in [16,32,56]:
    estimates={};boot={}
    for group in [2,3]:
     mask=common&(g==group);pt,rep=paired_mean(L,mask,b,W);med=np.median(L[mask]);mb=quantile_boot(L[mask],b[mask],W,np.array([.5]))[:,0];estimates[group]=(pt,med);boot[group]=(rep,mb)
     for statistic,value,reps in [('Mean',pt,rep),('Median',med,mb)]:cr.append(dict(Hazard=h,Severity_Group=GROUPS[group],Window_Days=H,Row_Type='COMMON_COHORT_GROUP_L',Metric='Fixed_Window_L',Statistic=statistic,Estimate=value,Comparison_Cohort='COMMON_16_32_56',Unit='Acomp_anomaly_days',**support(mask,pix,b),**ci(reps)))
    for z,statistic in enumerate(['Mean','Median']):cr.append(dict(Hazard=h,Severity_Group='Extreme_minus_High',Window_Days=H,Row_Type='COMMON_COHORT_SPATIAL_CONTRAST',Metric='Fixed_Window_L',Statistic=statistic,Estimate=estimates[3][z]-estimates[2][z],Comparison_Cohort='COMMON_16_32_56',N_High=int((common&(g==2)).sum()),N_Extreme=int((common&(g==3)).sum()),Unit='Acomp_anomaly_days',**support(common&((g==2)|(g==3)),pix,b),**ci(boot[3][z]-boot[2][z])))
   if H in [32,56]:
    # Same RNG reset for32 and56 ensures the same bounded sample IDs.
    rng=np.random.default_rng(SEED+hi*1000+777);data=[]
    for k in range(len(edge)-1):
     ix=np.flatnonzero(common&(bi==k));n=len(ix)
     if n<500:continue
     pool=np.empty((0,3))
     for chunk in np.unique(sourcechunk[ix]):
      ids=ix[sourcechunk[ix]==chunk];pool=bounded_matrix(pool,np.column_stack([severity[ids],old[ids],L[ids]]),rng)
     med=np.median(pool,axis=0);data.append((med[0],med[1],med[2],n))
    data=np.array(data)
    for name,col in [('Historical_L_on_same_window_subset',1),('Fixed_Window_L',2)]:
     fit=fit_segmented(data[:,0],data[:,col],data[:,3],r.Search_Min,r.Search_Max) if len(data) else dict(Status='NOT ESTIMABLE',Reason='no supportedbins')
     if 'S_star' in fit:fit['Q_Full']=100*np.searchsorted(full,fit['S_star'],side='right')/len(full)
     fr.append(dict(Hazard=h,Window_Days=H,Response=name,Comparison_Cohort='COMMON_16_32_56',Historical_S_star=r.S_star,Historical_Delta_AIC=r.Delta_AIC,N_Paired_Events=int(common.sum()),N_Paired_Pixels=np.unique(pix[common]).size,N_Paired_Blocks=np.unique(b[common]).size,Seed=SEED+hi*1000+777,Bin_Estimator='bounded20k original1Mchunk order; pairedIDs; same drawfor32/56',**fit))
  print('COMMON FIXED WINDOWS',h,int(common.sum()),flush=True);del e,c,old,b,W,g,pix,severity,full;gc.collect()
 pd.concat([outbins,pd.DataFrame(br)],ignore_index=True).to_csv(O/'outputs/C3_SeverityLegacy_Sensitivity.csv',index=False,encoding='utf-8-sig');pd.concat([outfits,pd.DataFrame(fr)],ignore_index=True).to_csv(O/'outputs/C3_Paired_Breakpoint_Fits.csv',index=False,encoding='utf-8-sig');pd.concat([fixed,pd.DataFrame(cr)],ignore_index=True).to_csv(O/'outputs/C2_FixedWindow_Legacy.csv',index=False,encoding='utf-8-sig');pd.DataFrame(den).to_csv(O/'qc/Common_Window_Denominators.csv',index=False)
 (O/'qc/COMMON_WINDOW_READY.json').write_text(json.dumps(dict(Completed=True,Common16_32_56=True,Same_Pair_Sample_For_32_56=True)),encoding='utf8')
if __name__=='__main__':main()
