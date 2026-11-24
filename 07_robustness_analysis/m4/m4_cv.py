# -*- coding: utf-8 -*-
from pathlib import Path
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='1'
import json,time,gc
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Polygon,box
from scipy.spatial import cKDTree
from pyproj import Transformer,Geod
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
O=Path(r'.\EI_Reviewer1_Major4_Revision');M3=Path(r'.\EI_Reviewer1_Major3_Revision')
def metrics(y,p):return dict(R2=r2_score(y,p),RMSE=np.sqrt(mean_squared_error(y,p)),MAE=mean_absolute_error(y,p),Response_SD=np.std(y),NRMSE_SD=np.sqrt(mean_squared_error(y,p))/np.std(y))
def main():
 assert json.loads((O/'audit/Historical_Reproduction.json').read_text())['Pass'];cfg=json.loads((O/'code/configuration.json').read_text());d=pd.read_parquet(O/'paired/Frozen_Model_Pixels.parquet');N=len(d);FEAT=cfg['Features'];KW=cfg['Explicit_Parameters'];X=d[FEAT].to_numpy();Y=d.Residual_damage.to_numpy();Yfree=d.NIRv_Free.to_numpy();xy=d[['X_m','Y_m']].to_numpy();points=shapely.points(xy);geod=Geod(ellps='WGS84');fullhist=pd.read_parquet(M3/'paired/Greening_Pixel_Contrasts.parquet');oldB=fullhist.Original_Recomputed_Benefit.to_numpy();oldBfree=fullhist.NIRv_Free_Benefit.to_numpy();G20,G80=np.quantile(d.NIRv_trend,[.2,.8]);Xlo=X.copy();Xhi=X.copy();Xlo[:,0]=G20;Xhi[:,0]=G80
 # Per-pixel event sufficient statistics, original fixed event selection.
 ev=pd.read_parquet(M3/'paired/CDH_Original_Clean_Cohort.parquet');em=np.load(M3/'cache/Event_Metrics.npy',mmap_mode='r');keep=ev.valid_resistance_sample.eq(1).to_numpy()&ev.Acomp_damage.gt(0).to_numpy()&np.isfinite(ev.climate_cumulative_severity.to_numpy());assert keep.sum()==1501147
 pixmap=np.full(414*703,-1,np.int32);pixmap[d.row.to_numpy(int)*703+d.col.to_numpy(int)]=np.arange(N);ix=pixmap[ev.row.to_numpy(int)[keep]*703+ev.col.to_numpy(int)[keep]];model=ix>=0;s=ev.climate_cumulative_severity.to_numpy()[keep][model];damage=ev.Acomp_damage.to_numpy()[keep][model];ix=ix[model];h=np.maximum(0,s-cfg['Fixed_S_star']);basis=np.column_stack([np.ones(len(s)),s,h]);n=np.bincount(ix,minlength=N);assert np.array_equal(n,d.n_events)
 xx=np.zeros((N,3,3));xz=np.zeros((N,3))
 for a in range(3):
  for b in range(3):xx[:,a,b]=np.bincount(ix,weights=basis[:,a]*basis[:,b],minlength=N)
  xz[:,a]=np.bincount(ix,weights=basis[:,a]*damage,minlength=N)
 meanbasis=np.column_stack([np.ones(N),np.bincount(ix,weights=s,minlength=N)/n,np.bincount(ix,weights=h,minlength=N)/n]);meanD=np.bincount(ix,weights=damage,minlength=N)/n
 np.savez_compressed(O/'paired/Event_Severity_Sufficient_Statistics.npz',N_Events=n,XtX=xx,Xty=xz,Mean_Design=meanbasis,Mean_Damage=meanD)
 eventmap=dict(Original_Global_Nuisance_N=1501147,N_Events_On_Frozen_Model_Pixels=int(n.sum()),N_Outside_Frozen_Model_Pixel_Queue=int(1501147-n.sum()),Frozen_N_Pixels=N,Nested_Training_Rule='only original formal events at retained TRAIN model pixels; test and buffer-excluded pixel events absent; same fixed S*; no new event selection')
 (O/'audit/Event_Pixel_Mapping.json').write_text(json.dumps(eventmap,indent=2),encoding='utf8')
 # Define all group labels and test square footprints independently of outcomes.
 grid=json.loads((O/'audit/Grid_Metadata.json').read_text())['Axes'];lonax=np.array(grid['x']);latax=np.array(grid['y']);dx=np.median(np.diff(lonax));dy=np.median(np.diff(latax));trans=Transformer.from_crs('EPSG:4326',cfg['Projection'],always_xy=True)
 def legacy_poly(group):
  iy=int(group)//1000;ix=int(group)%1000;r0=iy*20;r1=min((iy+1)*20,414);c0=ix*20;c1=min((ix+1)*20,703);lo0=lonax[c0]-dx/2;lo1=lonax[c1-1]+dx/2;la0=latax[r1-1]+dy/2;la1=latax[r0]-dy/2
  # Full angular cell footprint, densely transformed edges to curved projection.
  pts=np.vstack([np.column_stack([np.linspace(lo0,lo1,21),np.full(21,la0)]),np.column_stack([np.full(21,lo1),np.linspace(la0,la1,21)]),np.column_stack([np.linspace(lo1,lo0,21),np.full(21,la1)]),np.column_stack([np.full(21,lo0),np.linspace(la1,la0,21)])]);x,y=trans.transform(pts[:,0],pts[:,1]);return Polygon(np.column_stack([x,y]))
 schemes=[('V0_LEGACY','LEGACY',0,0)]+[(v,o,cfg['Schemes'][v]['Block_m'],cfg['Schemes'][v]['Buffer_m']) for o in ['O0','OHALF'] for v in ['V0','V1','V2','V3','V4']]
 summaries=[];details=[];leak=[];nestedrows=[];betas=[];greenrows=[];start=time.time()
 for scheme,origin,L,buffer in schemes:
  key=scheme+'_'+origin
  if scheme=='V0_LEGACY':groups=d.block20.to_numpy(int);u=np.unique(groups);polys={int(g):legacy_poly(g) for g in u};gi=d.row.to_numpy(int)//20;gj=d.col.to_numpy(int)//20
  else:
   offset=0 if origin=='O0' else L/2;gi=np.floor((xy[:,0]-offset)/L).astype(np.int32);gj=np.floor((xy[:,1]-offset)/L).astype(np.int32);pairs=np.column_stack([gi,gj]);upairs,groups=np.unique(pairs,axis=0,return_inverse=True);polys={g:box(a*L+offset,b*L+offset,(a+1)*L+offset,(b+1)*L+offset) for g,(a,b) in enumerate(upairs)}
  folds=np.zeros(N,np.uint8);trainmask=np.zeros((5,N),bool);excluded=np.zeros_like(trainmask);valid=True
  for fold,(initial,test) in enumerate(GroupKFold(5).split(X,Y,groups),1):
   folds[test]=fold;tg=np.unique(groups[test]);area=shapely.union_all([polys[int(k)] for k in tg]);dist=shapely.distance(points[initial],area);remove=dist<buffer if buffer else np.zeros(len(initial),bool);tr=initial[~remove];trainmask[fold-1,tr]=True;excluded[fold-1,initial[remove]]=True
   good=(len(tr)>=cfg['Stop_Rules']['Min_Training_Pixels'] and np.unique(groups[tr]).size>=cfg['Stop_Rules']['Min_Remaining_Training_Blocks'] and np.std(Y[tr])>cfg['Stop_Rules']['Min_Training_Response_SD']);valid&=good
   tree=cKDTree(xy[tr]);near,where=tree.query(xy[test],k=3);j=tr[where];l1=np.repeat(d.Longitude.to_numpy()[test],3);a1=np.repeat(d.Latitude.to_numpy()[test],3);_,_,gd=geod.inv(l1,a1,d.Longitude.to_numpy()[j].ravel(),d.Latitude.to_numpy()[j].ravel());gd=gd.reshape(-1,3)
   minarea=float(shapely.distance(points[tr],area).min());minproj=float(np.min(near));mingeod=float(gd.min());assert not np.intersect1d(groups[tr],tg).size and not trainmask[fold-1,test].any();assert not np.any(trainmask[fold-1]&excluded[fold-1]);assert buffer==0 or minarea>=buffer-1e-7
   row=dict(Configuration=key,Scheme=scheme,Origin=origin,Fold=fold,Block_km=L/1000 if L else np.nan,Buffer_km=buffer/1000,N_Total=N,N_Initial_Train=len(initial),N_Train=len(tr),N_Test=len(test),N_Buffer_Excluded=int(remove.sum()),N_Train_Blocks=np.unique(groups[tr]).size,N_Test_Blocks=len(tg),N_Total_Blocks=len(polys),Train_Xmin_m=xy[tr,0].min(),Train_Xmax_m=xy[tr,0].max(),Train_Ymin_m=xy[tr,1].min(),Train_Ymax_m=xy[tr,1].max(),Test_Xmin_m=xy[test,0].min(),Test_Xmax_m=xy[test,0].max(),Test_Ymin_m=xy[test,1].min(),Test_Ymax_m=xy[test,1].max(),Test_Lon_Min=d.Longitude.iloc[test].min(),Test_Lon_Max=d.Longitude.iloc[test].max(),Test_Lat_Min=d.Latitude.iloc[test].min(),Test_Lat_Max=d.Latitude.iloc[test].max(),Nearest_Train_Test_Point_m=minproj,Nearest_Train_Test_Geodesic_m=mingeod,Nearest_Train_To_Test_Unit_m=minarea,Train_ConvexHull_km2=shapely.convex_hull(shapely.multipoints(xy[tr])).area/1e6,Test_ConvexHull_km2=shapely.convex_hull(shapely.multipoints(xy[test])).area/1e6,Fold_Feasible=good)
   details.append(row);leak.append(dict(Configuration=key,Fold=fold,N_Test_Leaked_To_Train=0,N_Buffer_Reintroduced=0,N_Shared_Train_Test_Blocks=0,Buffer_km=buffer/1000,Min_Train_To_Test_Unit_km=minarea/1000,Min_Train_Test_Point_km=minproj/1000,Min_Train_Test_Geodesic_km=mingeod/1000,Buffer_Geometry=cfg['Buffer_Geometry'],Pass=buffer==0 or minarea>=buffer-1e-7))
  np.savez_compressed(O/'splits'/f'{key}.npz',Group=groups,Fold=folds,Train_Allowed=trainmask,Buffer_Excluded=excluded,Grid_i=gi,Grid_j=gj)
  if not valid:
   summaries.append(dict(Configuration=key,Scheme=scheme,Origin=origin,Response='Acomp_original',Layer='FIXED_HISTORICAL_RESPONSE',Status='STOPPED_INSUFFICIENT_TRAIN_COVERAGE',N_Pixels=N,N_Total_Blocks=len(polys),OOF_Coverage=0,OOF_N_Test=0,Missing_Reason='at least one fold violates predeclared train coverage; all scores withheld'));print('STOP',key,flush=True);continue
  responses=[('Acomp_original',Y,oldB)]
  if key in cfg['NIRv_Free_Schemes']:responses.append(('NIRv_free',Yfree,oldBfree))
  for response,y,fullB in responses:
   pred=np.full(N,np.nan);benef=np.full(N,np.nan)
   if scheme=='V0_LEGACY':
    pred=fullhist['Original_Recomputed_OOF' if response=='Acomp_original' else 'NIRv_Free_OOF'].to_numpy().copy();benef=fullhist['Original_Recomputed_OOF_Benefit' if response=='Acomp_original' else 'NIRv_Free_OOF_Benefit'].to_numpy().copy()
   else:
    for fold in range(1,6):
     tr=trainmask[fold-1];te=folds==fold;mod=HistGradientBoostingRegressor(**KW).fit(X[tr],y[tr]);pred[te]=mod.predict(X[te]);benef[te]=mod.predict(Xlo[te])-mod.predict(Xhi[te])
   for fold in range(1,6):
    te=folds==fold;met=metrics(y[te],pred[te]);details.append(dict(Configuration=key,Scheme=scheme,Origin=origin,Fold=fold,Response=response,Layer='FIXED_HISTORICAL_RESPONSE',**met))
   assert np.isfinite(pred).all();met=metrics(y,pred);summaries.append(dict(Configuration=key,Scheme=scheme,Origin=origin,Block_km=L/1000 if L else np.nan,Buffer_km=buffer/1000,Response=response,Layer='FIXED_HISTORICAL_RESPONSE',Status='COMPLETED',N_Pixels=N,N_Total_Blocks=len(polys),OOF_Coverage=1.,OOF_N_Test=N,**met,Greening_OOF_Mean=benef.mean(),Greening_OOF_Mean_over_SD=benef.mean()/met['Response_SD'],Greening_OOF_Correlation_With_Published_Full_Model=np.corrcoef(benef,fullB)[0,1],Min_Fold_R2=min(metrics(y[folds==f],pred[folds==f])['R2'] for f in range(1,6)),Max_Fold_R2=max(metrics(y[folds==f],pred[folds==f])['R2'] for f in range(1,6))))
   pd.DataFrame(dict(Pixel_Index=np.arange(N),Fold=folds,Target=y,OOF=pred,Greening_OOF_Contrast=benef)).to_parquet(O/'oof'/f'{key}_{response}_FIXED.parquet',index=False)
   moisture=pd.qcut(d.RootSM_base,10,labels=False,duplicates='drop').to_numpy()+1
   for k in range(1,11):
    ok=moisture==k;greenrows.append(dict(Configuration=key,Response=response,Layer='FIXED_HISTORICAL_RESPONSE',RootSM_Decile=k,N_Pixels=int(ok.sum()),RootSM_Mean=d.RootSM_base.to_numpy()[ok].mean(),Mean_OOF_Contrast=benef[ok].mean(),Mean_OOF_Contrast_over_SD=benef[ok].mean()/met['Response_SD']))
   print(key,response,'R2',round(met['R2'],5),'nRMSE',round(met['NRMSE_SD'],5),'seconds',round(time.time()-start,1),flush=True)
  # Final nested layer is built by m4_nested_full.py from ALL original formal
  # event cells, including eligible ancillary cells outside the model queue.
  if False and key in cfg['Nested_Severity_Schemes']:
   actual=np.full(N,np.nan);pred=np.full(N,np.nan);nb=np.full(N,np.nan)
   for fold in range(1,6):
    tr=trainmask[fold-1];te=folds==fold;beta=np.linalg.solve(xx[tr].sum(axis=0),xz[tr].sum(axis=0));yfold=meanD-meanbasis@beta;model=HistGradientBoostingRegressor(**KW).fit(X[tr],yfold[tr]);actual[te]=yfold[te];pred[te]=model.predict(X[te]);nb[te]=model.predict(Xlo[te])-model.predict(Xhi[te]);met=metrics(yfold[te],pred[te]);betas.append(dict(Configuration=key,Fold=fold,N_Training_Pixels=int(tr.sum()),N_Nuisance_Training_Events=int(n[tr].sum()),N_Test_Events=int(n[te].sum()),N_Buffer_Pixel_Events=int(n[excluded[fold-1]].sum()),S_star=cfg['Fixed_S_star'],Intercept=beta[0],Slope=beta[1],Hinge=beta[2],Test_Event_Used_In_Nuisance=0,Buffer_Event_Used_In_Nuisance=0));details.append(dict(Configuration=key,Scheme=scheme,Origin=origin,Fold=fold,Response='Acomp_original',Layer='TRAIN_ONLY_SEVERITY_COEFFICIENTS',**met))
   met=metrics(actual,pred);nestedrows.append(dict(Configuration=key,Scheme=scheme,Origin=origin,Response='Acomp_original',Layer='TRAIN_ONLY_SEVERITY_COEFFICIENTS',Status='COMPLETED',N_Pixels=N,OOF_Coverage=1.,OOF_N_Test=N,**met,Target_MeanAbs_Delta_From_Historical=np.mean(np.abs(actual-Y)),Target_MaxAbs_Delta_From_Historical=np.max(np.abs(actual-Y)),Greening_OOF_Mean=nb.mean(),Greening_OOF_Mean_over_SD=nb.mean()/met['Response_SD'],Limit='fold-specific targets use train-only coefficients at historically fixed S*; event selection and seasonal/trend preprocessing remain fixed; not entirely reselected full pipeline'))
   pd.DataFrame(dict(Pixel_Index=np.arange(N),Fold=folds,Target=actual,OOF=pred,Historical_Target=Y,Greening_OOF_Contrast=nb)).to_parquet(O/'oof'/f'{key}_Acomp_original_NESTED.parquet',index=False);print(key,'NESTED','R2',round(met['R2'],5),flush=True)
  pd.DataFrame(summaries+nestedrows).to_csv(O/'M4_Spatial_CV_Comparison.csv',index=False,encoding='utf-8-sig');pd.DataFrame(details).to_csv(O/'M4_Spatial_CV_Fold_Details.csv',index=False,encoding='utf-8-sig');pd.DataFrame(leak).to_csv(O/'M4_Buffer_Leakage_Audit.csv',index=False,encoding='utf-8-sig');pd.DataFrame(betas).to_csv(O/'paired/Training_Only_Severity_Coefficients.csv',index=False);pd.DataFrame(greenrows).to_csv(O/'supplement/OOF_Greening_RootSM_Profiles.csv',index=False)
 pd.DataFrame(summaries)[lambda a:a.Response.eq('NIRv_free')].to_csv(O/'M4_NIRvFree_Spatial_CV_Comparison.csv',index=False,encoding='utf-8-sig');(O/'cache/CV_READY.json').write_text(json.dumps(dict(Configurations=len(schemes),Rows=len(summaries)+len(nestedrows),Seconds=time.time()-start),indent=2),encoding='utf8');print('M4 CV COMPLETE',flush=True)
if __name__=='__main__':main()
