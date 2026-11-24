# -*- coding: utf-8 -*-
from pathlib import Path
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='1'
import json,time,hashlib,shutil
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
R=Path(r'.');O=R/'EI_Reviewer1_Major8_Revision';M4=R/'EI_Reviewer1_Major4_Revision'
def metrics(y,p):
 sd=np.std(y);return dict(R2=r2_score(y,p),RMSE=np.sqrt(mean_squared_error(y,p)),MAE=mean_absolute_error(y,p),Response_SD=sd,NRMSE_SD=np.sqrt(mean_squared_error(y,p))/sd,N_Test=len(y))
def summary(b):return dict(Contrast_Mean=np.mean(b),Contrast_Median=np.median(b),Contrast_Q05=np.quantile(b,.05),Contrast_Q95=np.quantile(b,.95),Fraction_Positive=np.mean(b>1e-12),Fraction_Negative=np.mean(b< -1e-12),Fraction_Numerical_Zero=np.mean(np.abs(b)<=1e-12))
def main():
 start=time.time();cfg=json.loads((O/'code/configuration.json').read_text());d=pd.read_parquet(O/'paired/Frozen_Model_Pixels_With_Terrain.parquet');N=len(d);assert N==153932;y=d.Residual_damage.to_numpy();FEAT=cfg['Features'];kw=cfg['Explicit_Parameters'];cols={'T0':FEAT,'T1':FEAT+['Elevation_Archived','Slope_Archived'],'T2':FEAT+['Elevation_Archived','Slope_Archived','Aspect_Sin_Archived','Aspect_Cos_Archived']};g20,g80=np.quantile(d.NIRv_trend,[.2,.8]);psd=np.std(y);rows=[];foldrows=[];green=[];unc=[];verify=[]
 (O/'models').mkdir(exist_ok=True);(O/'splits').mkdir(exist_ok=True)
 # Same50km blocks, paired spatial test-set resampling; fitted models are fixed.
 spl=np.load(M4/'splits/V2_O0.npz');block=spl['Group'];ub,bi=np.unique(block,return_inverse=True);B=len(ub);rng=np.random.default_rng(cfg['Spatial_Bootstrap_Seed']);W=np.array([np.bincount(rng.integers(0,B,B),minlength=B) for _ in range(cfg['Spatial_Bootstrap_N'])]);base={k:np.bincount(bi,weights=v,minlength=B) for k,v in [('n',np.ones(N)),('y',y),('yy',y*y)]};bn=W@base['n'];sst=W@base['yy']-(W@base['y'])**2/bn
 rootdec=d.RootSM_decile.to_numpy();moist=d.moisture_regime.astype(str).to_numpy()
 for key in cfg['Configurations']:
  src=M4/'splits'/f'{key}.npz';shutil.copyfile(src,O/'splits'/src.name);sp=np.load(src);fold=sp['Fold'];allowed=sp['Train_Allowed'];excluded=sp['Buffer_Excluded'];assert not np.any(allowed&excluded)
  old=pd.read_parquet(M4/'oof'/f'{key}_Acomp_original_FIXED.parquet');assert np.array_equal(old.Pixel_Index,np.arange(N)) and np.array_equal(old.Target,y) and np.array_equal(old.Fold,fold)
  preds={};benefits={};out=pd.DataFrame(dict(Pixel_Index=np.arange(N),row=d.row,col=d.col,Fold=fold,Target=y))
  for model,features in cols.items():
   X=d[features].to_numpy();pred=np.full(N,np.nan);bene=np.full(N,np.nan)
   for f in range(1,6):
    tr=allowed[f-1];te=fold==f;assert not np.any(tr&te);assert tr.sum()>=30000
    md=HistGradientBoostingRegressor(**kw).fit(X[tr],y[tr]);pred[te]=md.predict(X[te]);lo=X[te].copy();hi=lo.copy();lo[:,0]=g20;hi[:,0]=g80;bene[te]=md.predict(lo)-md.predict(hi)
    foldrows.append(dict(Configuration=key,Model=model,Fold=f,N_Train=int(tr.sum()),N_Buffer_Excluded=int(excluded[f-1].sum()),N_Train_Blocks=len(np.unique(sp['Group'][tr])),N_Test_Blocks=len(np.unique(sp['Group'][te])),OOF_Coverage=1,**metrics(y[te],pred[te]),**summary(bene[te])))
    joblib.dump(md,O/'models'/f'{key}_{model}_fold{f}.joblib',compress=3)
   assert np.isfinite(pred).all() and np.isfinite(bene).all()
   if model=='T0':
    pe=np.max(np.abs(pred-old.OOF.to_numpy()));be=np.max(np.abs(bene-old.Greening_OOF_Contrast.to_numpy()));assert pe<1e-11 and be<1e-11;verify.append(dict(Configuration=key,Prediction_MaxAbs=pe,Contrast_MaxAbs=be,N_Pixels=N,Split_SHA256=hashlib.sha256(src.read_bytes()).hexdigest()))
   preds[model]=pred;benefits[model]=bene;out[f'{model}_OOF']=pred;out[f'{model}_Greening_Contrast']=bene
   met=metrics(y,pred);r=dict(Configuration=key,Model=model,Response='M4_FIXED_HISTORICAL_Residual_damage',N_Pixels=N,OOF_Coverage=1,**met,**summary(bene),Greening_P20=g20,Greening_P80=g80,Greening_Unit='NIRv_per_decade',Features=';'.join(features),TNI_Included=False,Missing_Terrain_Pixels=int(d[features].isna().any(axis=1).sum()),Status='COMPUTED_SOURCE_LIMITED_TERRAIN' if model!='T0' else 'REPRODUCED',Limit='archived terrain product/unit metadata incomplete; aspect uses explicitdegree convention corroborated numerically;TNI omitted; fixed historical target retains prior M4 preprocessing caveat; not causal')
   r['Delta_R2_From_T0']=met['R2']-metrics(y,preds['T0'])['R2'];r['Delta_RMSE_From_T0']=met['RMSE']-metrics(y,preds['T0'])['RMSE'];r['Mean_Contrast_Change_From_T0']=np.mean(bene-benefits['T0']);r['MeanAbs_Contrast_Change_Over_ResponseSD']=np.mean(np.abs(bene-benefits['T0']))/psd;r['Contrast_Sign_Category_Changed_Fraction']=np.mean(np.sign(np.where(np.abs(bene)<=1e-12,0,bene))!=np.sign(np.where(np.abs(benefits['T0'])<=1e-12,0,benefits['T0'])));rows.append(r)
   err=(y-pred)**2;err0=(y-preds['T0'])**2;sse=W@np.bincount(bi,weights=err,minlength=B);sse0=W@np.bincount(bi,weights=err0,minlength=B);dr=(sse0-sse)/sst;db=(W@np.bincount(bi,weights=bene-benefits['T0'],minlength=B))/bn
   for metric,z,point in [('Delta_R2_From_T0',dr,r['Delta_R2_From_T0']),('Mean_Greening_Contrast_Change_From_T0',db,r['Mean_Contrast_Change_From_T0'])]:unc.append(dict(Analysis='PAIRED_FIXED_OOF_SPATIAL_TESTSET_RESAMPLING',Configuration=key,Model=model,Metric=metric,Estimate=point,CI_Lower=np.quantile(z,.025),CI_Upper=np.quantile(z,.975),N_Pixels=N,N_Spatial_Blocks=B,Block_km=50,Bootstrap_N=len(W),Seed=cfg['Spatial_Bootstrap_Seed'],Status='COMPUTED',Limit='whole50km test blocks resampled with same draws for models; does not refit model or nuisance preprocessing; source-limited terrain, not causal'))
   for dec in range(1,11):
    m=rootdec==dec;green.append(dict(Analysis='BASIN_OOF_CONDITIONAL_NOT_VEGETATION_STRATIFIED',Configuration=key,Model=model,Vegetation='ALL_UNCLASSIFIED_2001',RootSM_Decile=dec,N_Pixels=int(m.sum()),N_50km_Blocks=len(np.unique(block[m])),RootSM_Min=d.RootSM_base.to_numpy()[m].min(),RootSM_Max=d.RootSM_base.to_numpy()[m].max(),Greening_Min=d.NIRv_trend.to_numpy()[m].min(),Greening_Max=d.NIRv_trend.to_numpy()[m].max(),Greening_P20=g20,Greening_P80=g80,**summary(bene[m]),Status='COMPUTED_BASIN_ONLY',Limit='no2001 native landcover; cannot answer type-specific direction; same basin q20-q80 as M4'))
   print(key,model,'R2',round(met['R2'],6),'delta',round(r['Delta_R2_From_T0'],6),'sec',round(time.time()-start,1),flush=True)
  out.to_parquet(O/'oof'/f'{key}_Paired_Terrain_OOF.parquet',index=False)
  pd.DataFrame(rows).to_csv(O/'M8_Terrain_Ablation.csv',index=False,encoding='utf-8-sig');pd.DataFrame(foldrows).to_csv(O/'supplement/Terrain_Fold_Details.csv',index=False);pd.DataFrame(green).to_csv(O/'supplement/Basin_OOF_Greening_RootSM.csv',index=False);pd.DataFrame(unc).to_csv(O/'M8_Spatial_Support_And_Uncertainty.csv',index=False,encoding='utf-8-sig');pd.DataFrame(verify).to_csv(O/'audit/T0_M4_Reproduction.csv',index=False)
 # Full-data internal associations/ALE distinct from geographical OOF validity.
 ale=[];fullrows=[];full=pd.DataFrame(dict(Pixel_Index=np.arange(N),row=d.row,col=d.col));historical=pd.read_parquet(R/'EI_Reviewer1_Major3_Revision/paired/Greening_Pixel_Contrasts.parquet')
 for model,features in cols.items():
  X=d[features].to_numpy();md=HistGradientBoostingRegressor(**kw).fit(X,y);joblib.dump(md,O/'models'/f'FULL_{model}.joblib',compress=3);lo=X.copy();hi=X.copy();lo[:,0]=g20;hi[:,0]=g80;b=md.predict(lo)-md.predict(hi);full[f'{model}_Greening_Contrast']=b
  if model=='T0':assert np.max(np.abs(b-historical.Original_Recomputed_Benefit.to_numpy()))<1e-10
  fullrows.append(dict(Analysis='BASIN_FULL_MODEL_CONDITIONAL',Model=model,N_Pixels=N,Greening_P20=g20,Greening_P80=g80,**summary(b),Status='COMPUTED_BASIN_ONLY',Limit='same original model and target; main2001 vegetation unresolved; fullmodel is internal association, not cross-region validation'))
  for j,name in enumerate(FEAT):
   edges=np.unique(np.quantile(X[:,j],np.linspace(0,1,11)));bn=np.clip(np.searchsorted(edges,X[:,j],side='right')-1,0,len(edges)-2);local=[];ns=[]
   for k in range(len(edges)-1):
    m=bn==k;l=X[m].copy();h=l.copy();l[:,j]=edges[k];h[:,j]=edges[k+1];local.append(np.mean(md.predict(h)-md.predict(l)));ns.append(int(m.sum()))
   accumulated=np.cumsum(local);accumulated-=np.average(accumulated,weights=ns)
   for k,v in enumerate(accumulated):ale.append(dict(Model=model,Feature=name,Bin=k+1,X_Lower=edges[k],X_Upper=edges[k+1],X_Centre=(edges[k]+edges[k+1])/2,N_Pixels=ns[k],Local_Mean_Prediction_Change=local[k],ALE=v,ALE_Over_ResponseSD=v/psd,Estimator='10quantile-bin first-order accumulated local effects; weightedmean-centred',Limit='FULL_MODEL_INTERNAL, no causal or robust spatial-transfer claim; terrain sources incompletely documented'))
 full.to_parquet(O/'paired/Full_Model_Greening_Contrasts.parquet',index=False);pd.DataFrame(fullrows).to_csv(O/'supplement/Basin_Full_Model_Contrasts.csv',index=False);pd.DataFrame(ale).to_csv(O/'supplement/Terrain_Model_ALE.csv',index=False)
 corr=d[cols['T2']].corr(method='spearman');corr.to_csv(O/'supplement/Terrain_Hydroclimate_Spearman.csv',index_label='Feature')
 (O/'cache/MODELS_READY.json').write_text(json.dumps(dict(N_Configurations=5,N_Models=3,N_CV_Fits=75,N_Full_Fits=3,N_Pixels=N,T0_All_5_Reproduced=True,TNI_Excluded=True,Missing_Class_Group_Errors='NOT COMPLETED:2001 fineclassification missing',Seconds=time.time()-start),indent=2),encoding='utf8');print('MODELS READY',round(time.time()-start,1),flush=True)
if __name__=='__main__':main()
