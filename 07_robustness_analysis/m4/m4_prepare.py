# -*- coding: utf-8 -*-
from pathlib import Path
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='1'
import json,hashlib,ast
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from pyproj import CRS,Transformer,Geod
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
O=Path(r'.\EI_Reviewer1_Major4_Revision');D=Path(r'.\Weihe database V3');P=Path(r'.\analysis_project');M3=Path(r'.\EI_Reviewer1_Major3_Revision');M1=Path(r'.\EI_Reviewer1_Major1_Revision')
FEAT=['NIRv_trend','RootSM_base','DeepSM_base','VPD_base','Tmax_base','Temp_base','Precip_base','RAD_base']
KW=json.loads((M3/'audit/Greening_Reproduction.json').read_text())['Model_KW']
def main():
 for a in ['code','audit','cache','splits','oof','paired','supplement']:(O/a).mkdir(parents=True,exist_ok=True)
 before=[]
 for base in [D,P,M1,M3]:
  for p in base.rglob('*'):
   if p.is_file():z=p.stat();before.append(dict(Path=str(p),Bytes=z.st_size,Mtime_ns=z.st_mtime_ns))
 pd.DataFrame(before).to_csv(O/'audit/Inputs_Before.csv',index=False)
 paths=[P/'analysis_code/greening_model.py',P/'绘图代码/greening_model.py',P/'analysis_code/Step L6.0 FINAL V2.py',P/'analysis_code/Step2_build_common_grid_cube.py',P/'analysis_code/Step_L4_1_build_EcoFunction_Robust_V3.py',M1/'audit/Step_L5_4A_Severity_Response_Thresholds_FAST.py',M1/'R1M1_Reproducible_Code/model_utils.py',M1/'R1M1_Reproducible_Code/independent_breakpoints.py']
 src=[]
 for i,p in enumerate(paths):
  b=p.read_bytes();dst=O/'audit'/f'{i:02d}_{p.name}';dst.write_bytes(b);src.append(dict(Source=str(p),Snapshot=str(dst),SHA256=hashlib.sha256(b).hexdigest()))
 pd.DataFrame(src).to_csv(O/'audit/Source_Manifest.csv',index=False)
 d=pd.read_parquet(M3/'paired/Greening_Common_Pixel_Inputs.parquet').sort_values('Original_Source_Order').reset_index(drop=True);assert len(d)==153932 and not d[FEAT+['Residual_damage','NIRv_Free']].isna().any().any();assert d.block20.nunique()==461
 with Dataset(D/'Weihe_L3_ClimateStress_1000m_2000_2024_V3_1_FINAL.nc') as nc:
  lon=np.asarray(nc['x'][:])[d.col.to_numpy(int)];lat=np.asarray(nc['y'][:])[d.row.to_numpy(int)];attrs={k:str(nc['crs'].getncattr(k)) for k in nc['crs'].ncattrs()};axes=dict(x=np.asarray(nc['x'][:]).tolist(),y=np.asarray(nc['y'][:]).tolist())
 crs=CRS.from_proj4('+proj=aeqd +lat_0=35.5 +lon_0=107.5 +datum=WGS84 +units=m +no_defs');tr=Transformer.from_crs('EPSG:4326',crs,always_xy=True);xx,yy=tr.transform(lon,lat);d['Longitude']=lon;d['Latitude']=lat;d['X_m']=xx;d['Y_m']=yy
 d.to_parquet(O/'paired/Frozen_Model_Pixels.parquet',index=False)
 config=dict(Features=FEAT,Explicit_Parameters=KW,Resolved_Parameters=HistGradientBoostingRegressor(**KW).get_params(),Projection=crs.to_wkt(),Projection_Proj4=crs.to_proj4(),Grid_Origins=['O0:(0,0)m','OHALF:(L/2,L/2)m'],N_Folds=5,Split_Algorithm='GroupKFold; no shuffle or outcome-based selection',Schemes={'V0_LEGACY':dict(Block='original20x20analysis_pixels',Buffer_m=0,Origins=['LEGACY']), 'V0':dict(Block_m=20000,Buffer_m=0),'V1':dict(Block_m=50000,Buffer_m=0),'V2':dict(Block_m=50000,Buffer_m=20000),'V3':dict(Block_m=100000,Buffer_m=0),'V4':dict(Block_m=100000,Buffer_m=20000)},Buffer_Geometry='distance from train pixel centres to union of FULL projected test-block square polygons, not block-centre or test-point-only distance; legacy polygons are full original20x20pixel footprints',Stop_Rules={'Min_Training_Pixels':30000,'Min_Remaining_Training_Blocks':10,'Min_Training_Response_SD':1e-8,'Rule':'stop whole configuration before fitting if any fold violates; no partial pooled comparable score'},Nested_Severity_Schemes=['V0_LEGACY_LEGACY','V0_O0','V2_O0','V4_O0'],NIRv_Free_Schemes=['V0_LEGACY_LEGACY','V0_O0','V2_O0','V4_O0','V0_OHALF','V2_OHALF','V4_OHALF'],Fixed_S_star=.6833955143184973,Origin_Selection='predeclared two origins for every physical block scale; no best-origin selection',M3_Event_Source=str(M3/'paired/CDH_Original_Clean_Cohort.parquet'),N_Event_Original=1501147,Seed=42)
 (O/'code/configuration.json').write_text(json.dumps(config,indent=2),encoding='utf8');(O/'audit/Grid_Metadata.json').write_text(json.dumps(dict(CRS=attrs,Axes=axes),indent=2),encoding='utf8')
 # Gate: reproduce historical exact target/ordering/folds before any new split.
 X=d[FEAT].to_numpy();y=d.Residual_damage.to_numpy();groups=d.block20.to_numpy();pred=np.full(len(d),np.nan)
 for f,(train,test) in enumerate(GroupKFold(5).split(X,y,groups),1):
  assert np.all(d.Formal_Fold.to_numpy()[test]==f)
  pred[test]=HistGradientBoostingRegressor(**KW).fit(X[train],y[train]).predict(X[test]);print('reproduce fold',f,flush=True)
 hist=pd.read_parquet(M3/'paired/Greening_Pixel_Contrasts.parquet');assert np.array_equal(hist.row,d.row) and np.array_equal(hist.col,d.col)
 err=float(np.max(np.abs(pred-hist.Original_Recomputed_OOF.to_numpy())));metrics=dict(N=len(d),N_Blocks=int(d.block20.nunique()),R2=r2_score(y,pred),RMSE=np.sqrt(mean_squared_error(y,pred)),MAE=mean_absolute_error(y,pred),Response_SD=np.std(y),MaxAbs_From_M3=err,Pass=err<1e-10)
 (O/'audit/Historical_Reproduction.json').write_text(json.dumps(metrics,indent=2),encoding='utf8');assert metrics['Pass'],metrics
 np.save(O/'cache/Historical_OOF.npy',pred);print('HISTORICAL GATE PASS',metrics,flush=True)
if __name__=='__main__':main()
