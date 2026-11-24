# -*- coding: utf-8 -*-
"""Resolve audit-only administrative/record-end ties; never change any L value."""
import json,gc
import numpy as np
import pandas as pd
from core import *
def main():
 if (O/'qc/Historical_Admin_RecordEnd_Ties.csv').exists():return
 dates=np.load(M5/'cache/dates.npy').astype('datetime64[ns]');c1=pd.read_csv(O/'outputs/C1_Legacy_StopReasons_Duration.csv');rows=[]
 for h in HAZ:
  e=pd.read_parquet(MIN/'paired'/f'{h}_Fixed_Clean_Events.parquet',columns=['onset_idx'],use_threads=False);c=np.load(O/'qc/cache'/f'{h}_Legacy_Audit.npy',mmap_mode='r+');b,W=blockdata(h);g=groups(h);tie=(dates[e.onset_idx.to_numpy(int)]+np.timedelta64(120,'D')==dates[-1])&(c[:,4]==3);n=int(tie.sum());c[tie,4]=2;c.flush();rows.append(dict(Hazard=h,N_Admin_RecordEnd_Ties=n,Convention='120-day endpointcovered=>administrative; simultaneous record-endflag retained; noL values changed'))
  if n:
   for p in sorted((O/'outputs/event_details').glob(h+'_Legacy_*.parquet')):
    x=pd.read_parquet(p,columns=['onset_idx','Historical_Stop_Reason_Code'],use_threads=False);mask=(dates[x.onset_idx.to_numpy(int)]+np.timedelta64(120,'D')==dates[-1])&x.Historical_Stop_Reason_Code.eq(3).to_numpy()
    if not mask.any():continue
    d=pd.read_parquet(p,use_threads=False);d.loc[mask,'Historical_Stop_Reason_Code']=2
    if 'Historical_Admin_RecordEnd_Tie' not in d:d['Historical_Admin_RecordEnd_Tie']=False
    d.loc[mask,'Historical_Admin_RecordEnd_Tie']=True;d.to_parquet(p,index=False,compression='zstd')
  for k,label in enumerate(GROUPS):
   pop=(g==k) if k<4 else np.ones(len(e),bool)
   for code,name in [(2,'ONSET_120_BOUND'),(3,'RECORD_END')]:
    pt,rep=ratio(pop&(c[:,4]==code),pop,b,W);where=c1.Hazard.eq(h)&c1.Severity_Group.eq(label)&c1.Version.eq('HISTORICAL_L53_ONSET120_ACTIVE_INTERRUPT')&c1.Row_Type.eq('STOP_REASON')&c1.Metric.eq(name);c1.loc[where,'N_Numerator']=int((pop&(c[:,4]==code)).sum());c1.loc[where,'Estimate']=pt
    for field,value in ci(rep).items():c1.loc[where,field]=value
  del e,c,b,W,g;gc.collect()
 c1.to_csv(O/'outputs/C1_Legacy_StopReasons_Duration.csv',index=False,encoding='utf-8-sig');pd.DataFrame(rows).to_csv(O/'qc/Historical_Admin_RecordEnd_Ties.csv',index=False)
 print('AUDIT STOP-TIE RESOLUTION',rows,flush=True)
if __name__=='__main__':main()
