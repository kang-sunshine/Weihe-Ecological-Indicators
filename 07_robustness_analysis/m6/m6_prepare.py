# -*- coding: utf-8 -*-
from pathlib import Path
import ast,json,hashlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from netCDF4 import Dataset,num2date
O=Path(r'.\EI_Reviewer1_Major6_Revision');D=Path(r'.\Weihe database V3');P=Path(r'.\analysis_project');M1=Path(r'.\EI_Reviewer1_Major1_Revision');M3=Path(r'.\EI_Reviewer1_Major3_Revision');M4=Path(r'.\EI_Reviewer1_Major4_Revision');M5=Path(r'.\EI_Reviewer1_Major5_Revision')
def main():
 for x in ['code','audit','cache','paired','supplement']:(O/x).mkdir(parents=True,exist_ok=True)
 inventory=[]
 for base in [D,P,M1,M3,M4,M5]:
  for p in base.rglob('*'):
   if p.is_file():z=p.stat();inventory.append(dict(Path=str(p),Bytes=z.st_size,Mtime_ns=z.st_mtime_ns))
 pd.DataFrame(inventory).to_csv(O/'audit/Inputs_Before.csv',index=False)
 sources=[P/'analysis_code'/n for n in ['Step L5.0 - Build Event Masks from L3.py','Step_L4_1_build_EcoFunction_Robust_V3.py','Step_L4_2_build_Phenology_Zscore_Robust_V1.py','Step_L4_3_build_EcoFunction_StateSpace_Robust_V1.py','Step_L5_3_Build_Event_Severity_Resilience_Metrics_Robust_V2_FAST.py','empirical_greening.py','greening_model.py','Step L6.0 FINAL V2.py']]+[M1/'audit/Step_L5_4A_Severity_Response_Thresholds_FAST.py',M5/'code/survival_m5.py',M5/'code/extract_m5.py',M5/'code/analyse_m5.py']
 rows=[];func=[]
 for i,p in enumerate(sources):
  b=p.read_bytes();(O/'audit'/p.name).write_bytes(b);rows.append(dict(Source=str(p),SHA256=hashlib.sha256(b).hexdigest()))
  for n in ast.walk(ast.parse(b.decode('utf-8-sig'))):
   if isinstance(n,ast.FunctionDef):func.append(dict(Source=str(p),Function=n.name,Start=n.lineno,End=n.end_lineno))
 pd.DataFrame(rows).to_csv(O/'audit/Source_Manifest.csv',index=False);pd.DataFrame(func).to_csv(O/'audit/Source_Function_Locations.csv',index=False)
 with Dataset(D/'L4_EcoFunction/Weihe_L4_EcoFunction_RAW_1000m_2000_2024_V3_1.nc') as ds:
  dates=pd.DatetimeIndex([str(q) for q in num2date(ds['time'][:],ds['time'].units,only_use_cftime_datetimes=False)])
 np.save(O/'cache/dates.npy',dates.values);np.save(O/'cache/time_days.npy',(dates-dates[0]).days.to_numpy(np.int32));np.save(O/'cache/phase.npy',np.minimum(46,(dates.dayofyear.to_numpy()-1)//8+1));np.save(O/'cache/year.npy',dates.year.to_numpy())
 ev=pd.read_parquet(M3/'paired/CDH_Original_Clean_Cohort.parquet');assert len(ev)==1755878 and ev.event_num.is_unique;ev['Onset_Year']=dates.year.to_numpy()[ev.onset_idx.to_numpy(int)];ev['Pre_Window_Start_Year']=(dates.values[ev.onset_idx.to_numpy(int)]-np.timedelta64(40,'D')).astype('datetime64[Y]').astype(int)+1970;ev.to_parquet(O/'paired/CDH_Fixed_Original_Clean_Events.parquet',index=False)
 # Complete fixed-event severity populations for explicit period-specific ECDFs.
 f=M1/'cache/CDH';allS=np.concatenate([np.load(p)['severity'] for p in sorted(f.glob('raw_severity_*.npz'))]);assert len(allS)==4796211
 onset=np.full(len(allS),-1,np.int16)
 for p in sorted(Path(r'.\EI_Reviewer1_Major2_Revision\catalogues\CDH').glob('*.parquet')):
  d=pd.read_parquet(p,columns=['Event_Num','onset']);idx=d.Event_Num.to_numpy(int)-1;onset[idx]=d.onset.to_numpy(np.int16)
 assert (onset>=0).all();years=dates.year.to_numpy()[onset]
 for label,mask in [('ALL_2000_2024',np.ones(len(allS),bool)),('REF_2000_2020',years<=2020),('OUT_2021_2024',years>=2021),('EARLY_APPLY_2011_2024',years>=2011)]:np.save(O/'cache'/f'Full_Severity_{label}.npy',np.sort(allS[mask]))
 config=dict(Date='2026-11-11',Primary_Hazard='CDH',Other_Hazards='no full ecological response reconstruction; source audit all3 hazards',Original_Eco_Reference='2000-2020 inclusive, per pixel x calendar phase',LOYO='exclude observation calendar year from reference only for2000-2020 observations;2021-2024 use untouched original distribution',Early='2000-2010 reference; report2011-2024 targets, compute context observations for pre-window consistently',Bin_Response_Sampling='original joint bounded20k perbin, original1M Event_ID chunks, paired sampled IDs shared; fixed seed20260908; probability full counts',Min_Reference_Valid=10,Reference_SD_ddof=1,Min_SD=1e-6,Cap_Z=10.,Formula='sqrt((max(0,-ZSync_cap)^2+max(0,-ZuWUE_cap)^2+max(0,-ZNIRv_cap)^2)/3)',Pre_Window='actual40days before onset,median,>=3',Peak_Window='0..56actualdays inclusive,earliest maximum',Comparison_Support='exact common finite phase support, then same events; original native support also retained to quantify support change',Historical_Legacy='exact original L5.3 onset120/active-next/recovery stopping rule for breakpoint comparison',Recovery='separate M5 termination-origin120 inclusive/postpeak/catalogue-onset/two adjacent confirmation observations',Block_Y=6,Breakpoint_Bins='unchanged original historical severity edges,80quantile design collapsed to75bins;MIN_N500;sqrtN RSS weight',Candidate_Search='search ALL original SAVED_BIN_historical candidate severity values for each response, NOT fixing S*; no new candidates or range expansion',Magnitude_Sample='common finite positive damage intersection for both fits; fixed-original-positive signed sensitivity separately provided',Scopes=['REF_2000_2020','OUT_2021_2024','ALL_2000_2024_secondary','EARLY_APPLY_2011_2024'],Full_ECDF='same full original CDH catalogue restricted to onset years of comparison scope, includes unclean events; right-continuous float64 ties unsplit',Climate_Sampling='5geographically dispersed occupied50km blocks, determined without ecological responses; one representative per reconstructed0.1degree climate source cell',Recovery_Bootstrap_If_Feasible=dict(Scales_km=[20,50],N=250,Seed=20261109,Min_Valid=200),No_Original_Modifications=True)
 (O/'code/configuration.json').write_text(json.dumps(config,indent=2),encoding='utf8')
 pd.DataFrame([dict(Phase=p,Original_Available_Calendar_Years=int(((dates.year<=2020)&(np.minimum(46,(dates.dayofyear-1)//8+1)==p)).sum()),Early_Available_Calendar_Years=int(((dates.year<=2010)&(np.minimum(46,(dates.dayofyear-1)//8+1)==p)).sum()),Year2000_Phase_Present=bool(((dates.year==2000)&(np.minimum(46,(dates.dayofyear-1)//8+1)==p)).any())) for p in range(1,47)]).to_csv(O/'audit/Calendar_Phase_Reference_Years.csv',index=False)
 print('PREPARE READY',len(inventory),'protected files',len(ev),'clean CDH events',flush=True)
if __name__=='__main__':main()
