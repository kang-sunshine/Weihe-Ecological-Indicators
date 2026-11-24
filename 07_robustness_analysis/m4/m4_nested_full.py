# -*- coding: utf-8 -*-
from pathlib import Path
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='1'
import json,time
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Polygon,box
from pyproj import Transformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
O=Path(r'.\EI_Reviewer1_Major4_Revision');M3=Path(r'.\EI_Reviewer1_Major3_Revision');D=Path(r'.\Weihe database V3')
def metric(y,p):return dict(R2=r2_score(y,p),RMSE=np.sqrt(mean_squared_error(y,p)),MAE=mean_absolute_error(y,p),Response_SD=np.std(y),NRMSE_SD=np.sqrt(mean_squared_error(y,p))/np.std(y))
def main():
 cfg=json.loads((O/'code/configuration.json').read_text());d=pd.read_parquet(O/'paired/Frozen_Model_Pixels.parquet');e=pd.read_parquet(M3/'paired/CDH_Original_Clean_Cohort.parquet');keep=e.valid_resistance_sample.eq(1)&e.Acomp_damage.gt(0)&np.isfinite(e.climate_cumulative_severity);e=e.loc[keep];assert len(e)==1501147;cell=e.row.to_numpy(int)*703+e.col.to_numpy(int);unique,ix=np.unique(cell,return_inverse=True);N=len(unique);s=e.climate_cumulative_severity.to_numpy();h=np.maximum(0,s-cfg['Fixed_S_star']);damage=e.Acomp_damage.to_numpy();basis=np.column_stack([np.ones(len(e)),s,h]);n=np.bincount(ix);xx=np.zeros((N,3,3));xz=np.zeros((N,3))
 for a in range(3):
  for b in range(3):xx[:,a,b]=np.bincount(ix,weights=basis[:,a]*basis[:,b])
  xz[:,a]=np.bincount(ix,weights=basis[:,a]*damage)
 meanB=np.column_stack([np.ones(N),np.bincount(ix,weights=s)/n,np.bincount(ix,weights=h)/n]);meanD=np.bincount(ix,weights=damage)/n;model_idx=np.searchsorted(unique,d.row.to_numpy(int)*703+d.col.to_numpy(int));assert np.array_equal(unique[model_idx],d.row*703+d.col) and np.array_equal(n[model_idx],d.n_events)
 old=pd.read_csv(D/'L5_EventEcoFunction/L6_SeverityAdjustedResidual_FINAL_V2/L6_0_CDH_EventLevel_FixedBreakpoint_Model_FINAL_V2.csv').iloc[0];globalbeta=np.linalg.solve(xx.sum(axis=0),xz.sum(axis=0));assert np.allclose(globalbeta,[old.intercept,old.pre_break_slope,old.hinge_slope_change],rtol=1e-9,atol=1e-9)
 grid=json.loads((O/'audit/Grid_Metadata.json').read_text())['Axes'];lon=np.array(grid['x']);lat=np.array(grid['y']);row=unique//703;col=unique%703;tr=Transformer.from_crs('EPSG:4326',cfg['Projection'],always_xy=True);xxm,yym=tr.transform(lon[col],lat[row]);points=shapely.points(np.column_stack([xxm,yym]));dx=np.median(np.diff(lon));dy=np.median(np.diff(lat));X=d[cfg['Features']].to_numpy();lo=X.copy();hi=X.copy();lo[:,0],hi[:,0]=np.quantile(d.NIRv_trend,[.2,.8]);summary=[];fd=[];coef=[];started=time.time()
 def oldpoly(g):
  iy=int(g)//1000;jx=int(g)%1000;r0=iy*20;r1=min(r0+20,414);c0=jx*20;c1=min(c0+20,703);x0=lon[c0]-dx/2;x1=lon[c1-1]+dx/2;y0=lat[r1-1]+dy/2;y1=lat[r0]-dy/2;t=np.linspace(0,1,21);ll=np.r_[x0+t*(x1-x0),np.full(21,x1),x1-t*(x1-x0),np.full(21,x0)];la=np.r_[np.full(21,y0),y0+t*(y1-y0),np.full(21,y1),y1-t*(y1-y0)];a,b=tr.transform(ll,la);return Polygon(np.column_stack([a,b]))
 for key in cfg['Nested_Severity_Schemes']:
  sp=np.load(O/'splits'/f'{key}.npz');folds=sp['Fold'];train=sp['Train_Allowed'];ytest=np.full(len(d),np.nan);pred=ytest.copy();benef=ytest.copy();masks=np.zeros((5,N),bool);scheme='V0_LEGACY' if key.startswith('V0_LEGACY') else key.split('_')[0];buffer=cfg['Schemes'][scheme]['Buffer_m'];L=cfg['Schemes'][scheme].get('Block_m',0)
  for fold in range(1,6):
   te=folds==fold;trn=train[fold-1]
   if L:
    pairs=np.unique(np.column_stack([sp['Grid_i'][te],sp['Grid_j'][te]]),axis=0);geo=shapely.union_all([box(a*L,b*L,(a+1)*L,(b+1)*L) for a,b in pairs]);outside=shapely.distance(points,geo)>0
   else:
    testgroups=np.unique(sp['Group'][te]);geo=shapely.union_all([oldpoly(g) for g in testgroups]);fullgroups=(row//20)*1000+(col//20);outside=~np.isin(fullgroups,testgroups)
   dist=shapely.distance(points,geo);allow=outside&(dist>=buffer);masks[fold-1]=allow;assert np.array_equal(allow[model_idx],trn),key
   beta=np.linalg.solve(xx[allow].sum(axis=0),xz[allow].sum(axis=0));y=meanD[model_idx]-meanB[model_idx]@beta;mod=HistGradientBoostingRegressor(**cfg['Explicit_Parameters']).fit(X[trn],y[trn]);ytest[te]=y[te];pred[te]=mod.predict(X[te]);benef[te]=mod.predict(lo[te])-mod.predict(hi[te]);coef.append(dict(Configuration=key,Fold=fold,N_All_Formal_Events=1501147,N_Nuisance_Training_Events=int(n[allow].sum()),N_Nuisance_Training_Cells=int(allow.sum()),N_Ancillary_Training_Events=int(n[allow].sum()-n[model_idx][trn].sum()),N_Test_Events_On_Model_Pixels=int(n[model_idx][te].sum()),N_Excluded_Test_Or_Buffer_Events=int(n[~allow].sum()),Test_Event_Used_In_Nuisance=0,Buffer_Event_Used_In_Nuisance=0,S_star=cfg['Fixed_S_star'],Intercept=beta[0],Slope=beta[1],Hinge=beta[2],Nuisance_Spatial_Rule='all original formal event cells outside held-out FULL test units and20km buffer; ancillary non-model cells retained iff spatially eligible'))
   fd.append(dict(Configuration=key,Scheme=scheme,Origin='LEGACY' if not L else 'O0',Fold=fold,Response='Acomp_original',Layer='TRAIN_ONLY_SEVERITY_COEFFICIENTS',**metric(y[te],pred[te])))
  met=metric(ytest,pred);summary.append(dict(Configuration=key,Scheme=scheme,Origin='LEGACY' if not L else 'O0',Response='Acomp_original',Layer='TRAIN_ONLY_SEVERITY_COEFFICIENTS',Status='COMPLETED',N_Pixels=len(d),OOF_Coverage=1.,OOF_N_Test=len(d),**met,Target_MeanAbs_Delta_From_Historical=float(np.mean(np.abs(ytest-d.Residual_damage))),Target_MaxAbs_Delta_From_Historical=float(np.max(np.abs(ytest-d.Residual_damage))),Greening_OOF_Mean=benef.mean(),Greening_OOF_Mean_over_SD=benef.mean()/met['Response_SD'],Min_Fold_R2=min(q['R2'] for q in fd if q['Configuration']==key),Max_Fold_R2=max(q['R2'] for q in fd if q['Configuration']==key),Limit='coefficients trained on spatially eligible original event population; historical S*, original response-based selection and seasonal reference frozen; not a fully reselected pipeline'))
  pd.DataFrame(dict(Pixel_Index=np.arange(len(d)),Fold=folds,Target=ytest,OOF=pred,Historical_Target=d.Residual_damage,Greening_OOF_Contrast=benef)).to_parquet(O/'oof'/f'{key}_Acomp_original_NESTED.parquet',index=False);np.savez_compressed(O/'splits'/f'{key}_Nuisance_Event_Cell_Masks.npz',Cell_Grid_ID=unique,Train_Allowed=masks);print(key,'full-original-cohort nested','R2',met['R2'],'seconds',round(time.time()-started,1),flush=True)
 oldsum=pd.read_csv(O/'M4_Spatial_CV_Comparison.csv');oldsum=oldsum[oldsum.Layer!='TRAIN_ONLY_SEVERITY_COEFFICIENTS'];pd.concat([oldsum,pd.DataFrame(summary)],ignore_index=True).to_csv(O/'M4_Spatial_CV_Comparison.csv',index=False,encoding='utf-8-sig');oldfd=pd.read_csv(O/'M4_Spatial_CV_Fold_Details.csv');oldfd=oldfd[oldfd.Layer!='TRAIN_ONLY_SEVERITY_COEFFICIENTS'];pd.concat([oldfd,pd.DataFrame(fd)],ignore_index=True).to_csv(O/'M4_Spatial_CV_Fold_Details.csv',index=False,encoding='utf-8-sig');pd.DataFrame(coef).to_csv(O/'paired/Training_Only_Severity_Coefficients.csv',index=False);pd.DataFrame(dict(Cell_Grid_ID=unique,Row=row,Col=col,N_Formal_Events=n,X_m=xxm,Y_m=yym)).to_parquet(O/'paired/All_Formal_Nuisance_Event_Cells.parquet',index=False)
 meta=json.loads((O/'audit/Event_Pixel_Mapping.json').read_text());meta.update(dict(N_All_Formal_Event_Cells=N,Nested_Training_Rule='all1501147 original formal events retain eligibility; exclude test FULL units and buffer per fold; ancillary cells outside frozen model queue are retained only when eligible',Global_OLS_Reproduction_MaxAbs=float(np.max(np.abs(globalbeta-[old.intercept,old.pre_break_slope,old.hinge_slope_change])))));(O/'audit/Event_Pixel_Mapping.json').write_text(json.dumps(meta,indent=2),encoding='utf8');(O/'cache/NESTED_READY.json').write_text(json.dumps(dict(Configurations=len(summary),Rows=len(fd),N_Original_Events=len(e),N_All_Event_Cells=N),indent=2),encoding='utf8')
if __name__=='__main__':main()
