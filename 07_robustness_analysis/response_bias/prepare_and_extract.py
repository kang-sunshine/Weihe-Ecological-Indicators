# -*- coding: utf-8 -*-
import json,hashlib,time,gc
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from core import *

def snapshot():
 result={}
 roots=[R/'Weihe database V3',P]+[p for p in R.glob('EI_Reviewer1*') if p.is_dir()]
 for r in roots:
  for p in r.rglob('*'):
   if p.is_file() and not p.is_relative_to(O):s=p.stat();result[str(p)]=[s.st_size,s.st_mtime_ns]
 return result

def main():
 dirs()
 if (O/'qc/EXTRACTION_READY.json').exists():print('Extraction already complete; not repeated.');return
 if (O/'qc/Protected_Inputs_Before.json').exists():raise RuntimeError('Partial extraction exists; inspect before restart.')
 before=snapshot();(O/'qc/Protected_Inputs_Before.json').write_text(json.dumps(before),encoding='utf8')
 paths=[P/'analysis_code'/s for s in ['Step_L4_1_build_EcoFunction_Robust_V3.py','Step_L4_2_build_Phenology_Zscore_Robust_V1.py','Step_L5_3_Build_Event_Severity_Resilience_Metrics_Robust_V1.py','Step_L5_3_Build_Event_Severity_Resilience_Metrics_Robust_V2_FAST.py']]+[M5/'code/extract_m5.py',M3/'code/m3_extract.py',M3/'code/m3_strict_sync.py',MIN/'code/minor2_core.py',MIN/'cache/ACOMP_GATE.json',M2/'code/catalog_overlap_m2.py',M1/'audit/Step_L5_4A_Severity_Response_Thresholds_FAST.py']
 refs=[]
 for p in paths:
  data=p.read_bytes();refs.append(dict(Path=str(p),Bytes=len(data),SHA256=hashlib.sha256(data).hexdigest()))
 pd.DataFrame(refs).to_csv(O/'qc/Source_Code_Manifest.csv',index=False,encoding='utf-8-sig')
 assert json.loads((MIN/'cache/ACOMP_GATE.json').read_text())['Pass']
 cfg=dict(Date='2026-11-17 Asia/Shanghai',Seed=SEED,Bootstrap_Seed=20261159,Bootstrap_N=250,Blocks_km=50,Cohort='original clean-pre finite severity events used in M1; full catalogue totals separately reported',Legacy_Versions=['HISTORICAL_L53_ONSET120_ACTIVE_INTERRUPT','MINOR2_M5RULE_TERM120_CATALOG_INTERRUPT'],Fixed_Windows=WINDOWS,Fixed_Eligibility='full scheduled support termday0..H, record reachesH, no new catalogueonset<=H; no missing interpolation',Fixed_Legacy='retain source term+1 integration start; actual cells clipped toH, final M5-confirmed firstrecoverydate canstop retrospectively; no observed interval=>NA',Controls='samepixel exactcalendarphase differentyear; no merged catalogue occupancy in pre40..post56; same validpre andpeakcounts; maximum3; hashseed20261117; no outcome selection',No_Original_Writes=True,No_Model_Training=True,No_ZIP=True)
 (O/'qc/configuration.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf8')
 td=np.load(M5/'cache/time_days.npy');dates=np.load(M5/'cache/dates.npy').astype('datetime64[D]');di=pd.DatetimeIndex(dates);phase=np.minimum(46,(di.dayofyear.to_numpy()-1)//8+1);year=di.year.to_numpy()
 A=np.load(M1/'cache/Acomp_by_row.npy',mmap_mode='r');active=np.load(M1/'cache/AnyMask_by_row.npy',mmap_mode='r');spans=[np.load(M2/'cache'/f'span_{h}.npy',mmap_mode='r') for h in HAZ]
 E={h:events(h) for h in HAZ};M={h:existing(h) for h in HAZ};out={};B={};gate=[];start=time.time()
 for h in HAZ:
  assert len(E[h])==len(M[h]);assert E[h].event_num.is_unique
  out[h]=np.lib.format.open_memmap(O/'qc/cache'/f'{h}_Legacy_Audit.npy',mode='w+',dtype='float32',shape=(len(E[h]),24))
  B[h]=np.lib.format.open_memmap(O/'qc/cache'/f'{h}_Controls.npy',mode='w+',dtype='float32',shape=(len(E[h]),10))
 for y0 in range(0,414,6):
  y1=min(414,y0+6);a=np.transpose(A[y0:y1],(1,0,2)).reshape(1142,-1);act=np.transpose(active[y0:y1],(1,0,2)).reshape(1142,-1);span=np.logical_or.reduce([np.transpose(z[y0:y1],(1,0,2)).reshape(1142,-1) for z in spans])
  for h,e in E.items():
   ix=np.flatnonzero((e.row.to_numpy()>=y0)&(e.row.to_numpy()<y1));ss=e.onset_idx.to_numpy(np.int32)[ix];tt=e.term_idx.to_numpy(np.int32)[ix];cc=e.col.to_numpy(np.int32)[ix]+(e.row.to_numpy(np.int32)[ix]-y0)*703;m=np.asarray(M[h][ix]);out[h][ix]=legacy_audit(ss,tt,cc,td,a,act,m)
   B[h][ix]=controls(ss,cc,e.event_num.to_numpy(np.int32)[ix],td,phase,year,a,span,m,HAZ.index(h)+1)
  for d in [out,B]:
   for v in d.values():v.flush()
  print('C/B extraction rows',y0,y1,'seconds',round(time.time()-start,1),flush=True);del a,act,span;gc.collect()
 for h,e in E.items():
  c=out[h];m=M[h];good=np.isfinite(e.Acomp_legacy_anomaly_days.to_numpy());same=np.isfinite(c[:,7]);err=float(np.max(abs(c[good,7]-e.Acomp_legacy_anomaly_days.to_numpy()[good]))) if good.any() else 0.
  gate.append(dict(Hazard=h,N_events=len(e),Check='Historical_L_exact_replay',N_Missing_Mismatch=int((good!=same).sum()),MaxAbs=err,Tolerance=2e-4,Pass=bool((good==same).all() and err<=2e-4)))
  gate.append(dict(Hazard=h,N_events=len(e),Check='Historical_weighted_days_vs_existing_Minor2',N_Missing_Mismatch=0,MaxAbs=float(np.nanmax(abs(c[:,6]-m[:,15]))),Tolerance=0,Pass=bool(np.nanmax(abs(c[:,6]-m[:,15]))==0)))
  for field,col in [(0,'Acomp_pre'),(1,'Acomp_damage')]:
   src=e[col].to_numpy();mask=np.isfinite(src)&np.isfinite(m[:,field]);err=float(np.max(abs(src[mask]-m[mask,field]))) if mask.any() else 0;gate.append(dict(Hazard=h,N_events=len(e),Check=col+'_existing_M5rule_input',N_Missing_Mismatch=int((np.isfinite(src)!=np.isfinite(m[:,field])).sum()),MaxAbs=err,Tolerance=1e-6,Pass=bool(err<=1e-6 and (np.isfinite(src)==np.isfinite(m[:,field])).all())))
 pd.DataFrame(gate).to_csv(O/'qc/C_Historical_Reproduction_Gate.csv',index=False)
 assert all(g['Pass'] for g in gate),'STOP: archival version conflict; no automatic repair'
 assert before==snapshot(),'Protected source modified'
 (O/'qc/EXTRACTION_READY.json').write_text(json.dumps(dict(Completed=True,Seconds=time.time()-start,N_Clean={h:len(e) for h,e in E.items()},Historical_Reproduction_Pass=True)),encoding='utf8')
 print('C/B EXTRACTION COMPLETE',time.time()-start,flush=True)
if __name__=='__main__':main()
