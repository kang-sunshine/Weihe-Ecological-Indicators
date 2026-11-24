# -*- coding: utf-8 -*-
"""Independent pre40d medians for baseline-valid/peak-empty rows only.
No stored source baseline is copied into recomputed outputs. Everything else was
already verified exactly against M5. Retain failed operational audit evidence.
"""
from pathlib import Path
import json,shutil
import numpy as np
import pandas as pd
from minor2_core import O,M1
def main():
 old=pd.read_csv(O/'audit/Acomp_Gate_Verification.csv');assert old[old.Check!='Baseline'].Pass.all()
 if not (O/'audit/Gate_Initial_Baseline_Missingness.csv').exists():shutil.copy2(O/'audit/Acomp_Gate_Verification.csv',O/'audit/Gate_Initial_Baseline_Missingness.csv')
 td=np.load(O/'cache/time_days.npy');A=np.load(M1/'cache/Acomp_by_row.npy',mmap_mode='r');records=[]
 for h in ['CDH','HEAT','HR']:
  e=pd.read_parquet(O/'paired'/f'{h}_Fixed_Clean_Events.parquet',columns=['row','col','onset_idx','Acomp_pre']);m=np.load(O/'cache'/h/'Acomp.npy',mmap_mode='r+');ix=np.flatnonzero(np.isfinite(e.Acomp_pre.to_numpy())&~np.isfinite(m[:,0]));assert np.all(m[ix,4]>=3)&np.all(m[ix,5]==0)
  for i in ix:
   s=int(e.onset_idx.iloc[i]);r=int(e.row.iloc[i]);c=int(e.col.iloc[i]);v=np.asarray(A[r,np.searchsorted(td,td[s]-40):s,c]);v=v[np.isfinite(v)];assert len(v)>=3;m[i,0]=np.median(v)
  m.flush();a=e.Acomp_pre.to_numpy();b=m[:,0];good=np.isfinite(a)&np.isfinite(b);mm=int((np.isfinite(a)!=np.isfinite(b)).sum());err=float(np.max(abs(a[good]-b[good])));assert mm==0 and err<=1e-6;z=(old.Event==h)&(old.Check=='Baseline');old.loc[z,['N_Finite','Missing_Mismatch','MaxAbs','Pass']]=[int(good.sum()),mm,err,True];records.append(dict(Event=h,N_Baseline_Only_Recomputed=len(ix),Cause='initialearlyreturnwhenpeakempty,baselineitselfvalid',Source_Metric_Rule_Changed=False,Other_Fields_Changed=False,Independent_Cached_Trajectory_Median=True,MaxAbs=err))
 assert old.Pass.all();old.to_csv(O/'audit/Acomp_Gate_Verification.csv',index=False);pd.DataFrame(records).to_csv(O/'audit/Gate_Baseline_Only_Repair.csv',index=False);(O/'cache/ACOMP_GATE.json').write_text(json.dumps(dict(Pass=True,N_Checks=len(old),N_Indicators_Compared=0,Baseline_EmptyPeak_Rule_Fixed=True,Recovery_And_All_Other_Fields_Unchanged=True),indent=2),encoding='utf8');print(old.to_string(index=False));print('ACOMP GATE PASS',flush=True)
if __name__=='__main__':main()
