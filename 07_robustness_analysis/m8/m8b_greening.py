# -*- coding: utf-8 -*-
from pathlib import Path
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='1'
import json,time,gc
import numpy as np
import pandas as pd
import joblib
from scipy.stats import spearmanr
O=Path(r'.\EI_Reviewer1_Major8_Revision');R=O.parent
def main():
 start=time.time();cfg=json.loads((O/'code/M8B_configuration.json').read_text());oldcfg=json.loads((O/'code/configuration.json').read_text());d=pd.read_parquet(O/'paired/Frozen_Model_Pixels_With_Terrain.parquet');N=len(d);lc=np.load(O/'cache/M8B/Landcover_Cohorts.npz');sp=np.load(O/'cache/M8B/Spatial_Grid.npz');W=np.load(O/'cache/M8B/Shared_50km_Block_Multiplicities.npy');B=W.shape[1];r=d.row.to_numpy(int);c=d.col.to_numpy(int);g=lc['Main_Group'][r,c];mode=lc['Main_Mode'][r,c];exact=lc['Exact_ThreePeriod'][r,c];pure=lc['Main_Purity'][r,c]>=.75;block=sp['Block50_Index'][r,c];y=d.Residual_damage.to_numpy();green=d.NIRv_trend.to_numpy();sm=d.RootSM_base.to_numpy();area=np.load(O/'cache/Pixel_Area_km2.npy')[r,c];baseSD=np.std(y);g20,g80=np.quantile(green,[.2,.8]);qtype={i:np.quantile(green[g==i],[.2,.8]) for i in [1,2,3]};feat=oldcfg['Features'];features={'T0':feat,'T1':feat+['Elevation_Archived','Slope_Archived'],'T2':feat+['Elevation_Archived','Slope_Archived','Aspect_Sin_Archived','Aspect_Cos_Archived']};allmain=np.isin(g,[1,2,3]);results=[];endpoints=[];errors=[];correls=[];curves=[];pairdiff=[];trainchecks=[]
 def count(m):return np.bincount(block[m],minlength=B)
 def sums(m,z):return np.bincount(block[m],weights=z[m],minlength=B)
 def bootmean(m,z):
  nn=W@count(m);v=W@sums(m,z);return np.divide(v,nn,out=np.full(len(W),np.nan),where=nn>0)
 def limits(v,support=True):
  good=np.isfinite(v);return (float(np.quantile(v[good],.025)),float(np.quantile(v[good],.975)),int(good.sum())) if support and good.sum()>=cfg['Minimum_Valid_Bootstrap'] else (np.nan,np.nan,int(good.sum()))
 def spatial(m):
  nn=count(m);n=int(nn.sum());nb=int(np.count_nonzero(nn));eff=1/np.sum((nn/n)**2) if n else np.nan;return n,nb,eff
 def supported(m):
  n,nb,_=spatial(m);return n>=cfg['Minimum_Green_Group_Pixels'] and nb>=cfg['Minimum_Green_Group_50km_Blocks']
 def endpoint(m,lo,hi):
  n,nb,eff=spatial(m);low=m&(green<=lo);high=m&(green>=hi);nl,bl,_=spatial(low);nh,bh,_=spatial(high);mi=float(green[m].min()) if n else np.nan;ma=float(green[m].max()) if n else np.nan;obs=n>0 and mi<=lo<=ma and mi<=hi<=ma and nl>=cfg['Minimum_Endpoint_Arm_Pixels'] and nh>=cfg['Minimum_Endpoint_Arm_Pixels'] and bl>=cfg['Minimum_Endpoint_Arm_Blocks'] and bh>=cfg['Minimum_Endpoint_Arm_Blocks']
  return dict(N_Pixels=n,N_50km_Blocks=nb,Effective_50km_InvHHI=eff,N_Events=int(d.n_events.to_numpy()[m].sum()),Greening_Min=mi,Greening_Max=ma,Greening_P20=lo,Greening_P80=hi,Greening_Span=hi-lo,Endpoint_P20_Percentile_Within_Subset=100*np.mean(green[m]<=lo) if n else np.nan,Endpoint_P80_Percentile_Within_Subset=100*np.mean(green[m]<=hi) if n else np.nan,N_Low_Green=nl,N_High_Green=nh,N_Low_50km_Blocks=bl,N_High_50km_Blocks=bh,Endpoint_Within_Observed_Range=bool(n and mi<=lo<=ma and mi<=hi<=ma),Endpoint_Observed_Support=bool(obs),Response_SD_Type=np.std(y[m]) if n else np.nan,Response_SD_Basin=baseSD,RootSM_Min=sm[m].min() if n else np.nan,RootSM_Max=sm[m].max() if n else np.nan),low,high
 defs=[('MAIN_2001',i,n,g==i) for i,n in [(1,'Forest'),(2,'Grassland_Shrubland'),(3,'Cropland')]]+ [('THREE_PERIOD_EXACT_CLASS',i,n,(g==i)&exact) for i,n in [(1,'Forest'),(2,'Grassland_Shrubland'),(3,'Cropland')]]+[('PURITY_GE0p75',i,n,(g==i)&pure) for i,n in [(1,'Forest'),(2,'Grassland_Shrubland'),(3,'Cropland')]]+[('CROP14_INCLUSION_SENSITIVITY',3,'Cropland12plus14',np.isin(mode,[12,14]))]
 strata=[('ALL',np.ones(N,bool))]+[(f'RootSM_decile_{i}',d.RootSM_decile.to_numpy()==i) for i in range(1,11)]+[(f'Moisture_{k}',d.moisture_regime.astype(str).to_numpy()==k) for k in sorted(d.moisture_regime.astype(str).unique())]
 # Empirical residual contrasts and distribution curves: no new target or model.
 for cohort,gid,name,gm in defs:
  for ep,qq in [('BASIN_P20_P80',(g20,g80)),('TYPE_P20_P80',qtype[gid])]:
   if cohort=='CROP14_INCLUSION_SENSITIVITY' and ep=='TYPE_P20_P80':continue
   for st,stmask in strata:
    m=gm&stmask;e,low,high=endpoint(m,*qq);ok=supported(m) and e['Endpoint_Observed_Support'];bm=bootmean(low,y)-bootmean(high,y);lo,hi,nboot=limits(bm,ok);desc=float(np.mean(y[low])-np.mean(y[high])) if low.any() and high.any() else np.nan;status='ESTIMABLE_OBSERVATIONAL' if ok and nboot>=280 else 'NOT ESTIMABLE';shared=dict(Cohort=cohort,Vegetation=name,RootSM_Stratum=st,Endpoint_Set=ep,**e,Unit='Acomp_residual_damage',Landcover_Year=2001)
    results.append(dict(**shared,Analysis='EMPIRICAL_TAIL_CONTRAST',Model='NONE',Configuration='OBSERVED_FIXED_TARGET',Estimate=desc if status.startswith('ESTIMABLE') else np.nan,Descriptive_Estimate=desc,CI_Lower=lo,CI_Upper=hi,Bootstrap_Valid=nboot,N_Bootstrap=len(W),Low_Residual_Mean=np.mean(y[low]) if low.any() else np.nan,High_Residual_Mean=np.mean(y[high]) if high.any() else np.nan,Low_Residual_Median=np.median(y[low]) if low.any() else np.nan,High_Residual_Median=np.median(y[high]) if high.any() else np.nan,Residual_Q25=np.quantile(y[m],.25) if m.any() else np.nan,Residual_Median=np.median(y[m]) if m.any() else np.nan,Residual_Q75=np.quantile(y[m],.75) if m.any() else np.nan,Status=status,Missing_Reason='' if ok else 'N<100 or spatial blocks<10 or endpoint arm<50pixels/<5blocks or endpoint outside observed range',Limit='observed low-high mean residual, original storedRootSM/greening strata on fixed153932queue; does not reproduce a differentFigure8sample or imply causality'))
    if st=='ALL':endpoints.append(dict(Cohort=cohort,Vegetation=name,Endpoint_Set=ep,**e,Status='SUPPORTED' if ok else 'NOT ESTIMABLE'))
  if cohort=='MAIN_2001':
   for moist in sorted(d.moisture_regime.astype(str).unique()):
    for dec in range(1,11):
     m=gm&(d.moisture_regime.astype(str).to_numpy()==moist)&(d.greening_decile.to_numpy()==dec);n,nb,eff=spatial(m);bm=bootmean(m,y);lo,hi,nboot=limits(bm,supported(m));curves.append(dict(Vegetation=name,Moisture_Regime=moist,Greening_Decile_Original=dec,N_Pixels=n,N_50km_Blocks=nb,NIRv_Trend_Mean=green[m].mean() if n else np.nan,NIRv_Trend_Min=green[m].min() if n else np.nan,NIRv_Trend_Max=green[m].max() if n else np.nan,Residual_Mean=y[m].mean() if n else np.nan,Residual_Median=np.median(y[m]) if n else np.nan,Residual_Q25=np.quantile(y[m],.25) if n else np.nan,Residual_Q75=np.quantile(y[m],.75) if n else np.nan,Mean_CI_Lower=lo,Mean_CI_Upper=hi,Status='ESTIMABLE_DESCRIPTIVE' if supported(m) and nboot>=280 else 'NOT ESTIMABLE',Limit='stored basin stratification retained, not within-type rebinning; observational'))
 print('EMPIRICAL READY',round(time.time()-start,1),flush=True)
 contrasts=[];full=pd.read_parquet(O/'paired/Full_Model_Greening_Contrasts.parquet');assert np.array_equal(full.row,d.row) and np.array_equal(full.col,d.col)
 for md,fs in features.items():
  mod=joblib.load(O/'models'/f'FULL_{md}.joblib');X=d[fs].to_numpy();lo=X[allmain].copy();hi=lo.copy();gs=g[allmain];lo[:,0]=np.array([qtype[int(i)][0] for i in gs]);hi[:,0]=np.array([qtype[int(i)][1] for i in gs]);bb=np.full(N,np.nan);bb[allmain]=mod.predict(lo)-mod.predict(hi);contrasts.append(('FULL',md,'BASIN_P20_P80',full[f'{md}_Greening_Contrast'].to_numpy()));contrasts.append(('FULL',md,'TYPE_P20_P80',bb));full[f'{md}_TypeP20P80_Contrast']=bb
 full.to_parquet(O/'paired/M8B/Full_Model_Type_Contrasts.parquet',index=False)
 for key in oldcfg['Configurations']:
  oo=pd.read_parquet(O/'oof'/f'{key}_Paired_Terrain_OOF.parquet');split=np.load(O/'splits'/f'{key}.npz');assert np.array_equal(oo.Target,y) and np.array_equal(oo.Fold,split['Fold']);new=oo[['Pixel_Index','row','col','Fold','Target']].copy()
  for md,fs in features.items():
   X=d[fs].to_numpy();bb=np.full(N,np.nan)
   for f in range(1,6):
    te=(split['Fold']==f)&allmain;mod=joblib.load(O/'models'/f'{key}_{md}_fold{f}.joblib');lo=X[te].copy();hi=lo.copy();gs=g[te];lo[:,0]=np.array([qtype[int(i)][0] for i in gs]);hi[:,0]=np.array([qtype[int(i)][1] for i in gs]);bb[te]=mod.predict(lo)-mod.predict(hi)
    for gid,name in [(1,'Forest'),(2,'Grassland_Shrubland'),(3,'Cropland')]:
     m=(g==gid)&split['Train_Allowed'][f-1];trlo=green[m].min() if m.any() else np.nan;trhi=green[m].max() if m.any() else np.nan
     if md=='T0':
      for ep,qq in [('BASIN_P20_P80',(g20,g80)),('TYPE_P20_P80',qtype[gid])]:trainchecks.append(dict(Configuration=key,Fold=f,Vegetation=name,Endpoint_Set=ep,N_Training_Type_Pixels=int(m.sum()),Training_Type_Min=trlo,Training_Type_Max=trhi,P20=qq[0],P80=qq[1],Within_Training_Type_Marginal_Range=bool(m.any() and trlo<=qq[0]<=trhi and trlo<=qq[1]<=trhi),Limit='range diagnostic only, not joint positivity or complete pipeline independence'))
   contrasts.append((key,md,'BASIN_P20_P80',oo[f'{md}_Greening_Contrast'].to_numpy()));contrasts.append((key,md,'TYPE_P20_P80',bb));new[f'{md}_TypeP20P80_Contrast']=bb
   pred=oo[f'{md}_OOF'].to_numpy()
   for cohort,gid,name,gm in defs:
    for f in [0,1,2,3,4,5]:
     m=gm if f==0 else gm&(oo.Fold.to_numpy()==f);n,nb,eff=spatial(m);v=y[m];p=pred[m];sd=np.std(v) if n else np.nan;sse=np.sum((v-p)**2);sst=np.sum((v-v.mean())**2) if n else 0;r2=1-sse/sst if sst>0 else np.nan;ok=supported(m) and np.isfinite(r2);rmse=np.sqrt(np.mean((v-p)**2)) if n else np.nan
     errors.append(dict(Configuration=key,Model=md,Cohort=cohort,Vegetation=name,Fold='POOLED' if f==0 else str(f),N_Test=n,N_50km_Blocks=nb,Effective_50km_InvHHI=eff,OOF_Coverage=1 if n else np.nan,R2=r2 if ok else np.nan,Descriptive_R2=r2,RMSE=rmse if ok else np.nan,MAE=np.mean(abs(v-p)) if ok else np.nan,Response_SD=sd,RMSE_Over_TypeSD=rmse/sd if ok and sd>0 else np.nan,RMSE_Over_BasinSD=rmse/baseSD if ok else np.nan,Status='ESTIMABLE_FIXED_TARGET_OOF' if ok else 'NOT ESTIMABLE',Limit='same frozen target/parameters/pixels/splits, reused predictions, no training;M4 global nuisance preprocessing caveat retained'))
  new.to_parquet(O/'paired/M8B'/f'{key}_WithinType_Contrasts.parquet',index=False);print('PREDICT_ONLY',key,round(time.time()-start,1),flush=True)
 # Conditional summaries and RootSM relationship; same block draws across types.
 for config,md,ep,z in contrasts:
  for cohort,gid,name,gm in defs:
   if cohort=='CROP14_INCLUSION_SENSITIVITY' and ep=='TYPE_P20_P80':continue
   qq=(g20,g80) if ep=='BASIN_P20_P80' else qtype[gid]
   for st,stm in strata:
    m=gm&stm&np.isfinite(z);e,_,_=endpoint(m,*qq);ok=supported(m) and e['Endpoint_Observed_Support'];bm=bootmean(m,z);lo,hi,nboot=limits(bm,ok);desc=np.mean(z[m]) if m.any() else np.nan;row=dict(Cohort=cohort,Vegetation=name,RootSM_Stratum=st,Endpoint_Set=ep,**e,Analysis='FULL_MODEL_CONDITIONAL' if config=='FULL' else 'SPATIAL_OOF_CONDITIONAL',Model=md,Configuration=config,Estimate=desc if ok and nboot>=280 else np.nan,Descriptive_Estimate=desc,CI_Lower=lo,CI_Upper=hi,Bootstrap_Valid=nboot,N_Bootstrap=len(W),Unit='Acomp_residual_damage',Contrast_Median=np.median(z[m]) if m.any() else np.nan,Contrast_Q05=np.quantile(z[m],.05) if m.any() else np.nan,Contrast_Q95=np.quantile(z[m],.95) if m.any() else np.nan,Fraction_Positive=np.mean(z[m]>1e-12) if m.any() else np.nan,Fraction_Negative=np.mean(z[m]< -1e-12) if m.any() else np.nan,Fraction_Numerical_Zero=np.mean(abs(z[m])<=1e-12) if m.any() else np.nan,Estimate_Over_Type_ResponseSD=desc/e['Response_SD_Type'] if e['Response_SD_Type']>0 else np.nan,Estimate_Over_Basin_ResponseSD=desc/baseSD,Contrast_Per0p01_NIRvDecade=desc/(qq[1]-qq[0])*.01,Status='ESTIMABLE_MODEL_CONDITIONAL' if ok and nboot>=280 else 'NOT ESTIMABLE',Missing_Reason='' if ok and nboot>=280 else 'spatial/sample/endpoint support insufficient or fewer than280/350 finite blockreplicates; descriptive values not validated ecological effects',Limit='fixed fitted model, block resampling pixels, not refitted model uncertainty, no causal inference; numeric sign tolerance1e-12; endpoints whole frozen type cohort, no new feature/model tuning');results.append(row)
   if cohort=='MAIN_2001':
    m=gm&np.isfinite(z);n,nb,eff=spatial(m);sumx=sums(m,sm);sumz=sums(m,z);sumxx=sums(m,sm*sm);sumzz=sums(m,z*z);sumxz=sums(m,sm*z);nn=W@count(m);vx=W@sumxx-(W@sumx)**2/np.maximum(nn,1);vz=W@sumzz-(W@sumz)**2/np.maximum(nn,1);cov=W@sumxz-(W@sumx)*(W@sumz)/np.maximum(nn,1);rho=np.divide(cov,np.sqrt(np.maximum(vx*vz,0)),out=np.full(len(W),np.nan),where=(vx>1e-15)&(vz>1e-15));validpoint=n>0 and np.std(z[m])>1e-15 and np.std(sm[m])>1e-15;point=np.corrcoef(sm[m],z[m])[0,1] if validpoint else np.nan;lo,hi,nboot=limits(rho,supported(m) and validpoint);correls.append(dict(Configuration=config,Model=md,Endpoint_Set=ep,Vegetation=name,N_Pixels=n,N_50km_Blocks=nb,Pearson_Rho=point if supported(m) and nboot>=280 else np.nan,Descriptive_Pearson=point,Pearson_CI_Lower=lo,Pearson_CI_Upper=hi,Spearman_Rho_Descriptive=spearmanr(sm[m],z[m]).statistic if validpoint else np.nan,Bootstrap_Valid=nboot,Status='ESTIMABLE_ASSOCIATION' if supported(m) and validpoint and nboot>=280 else 'NOT ESTIMABLE',Limit='correlation ofRootSMwithfixedmodelcontrast;50kmblockPearsonCI,Spearmanpointdescriptive,not independentmechanism orcausal'))
  if ep=='BASIN_P20_P80':
   for a,b in [(1,2),(1,3),(2,3)]:
    ma=g==a;mb=g==b;dist=bootmean(ma,z)-bootmean(mb,z);lo,hi,nboot=limits(dist,supported(ma)&supported(mb));pairdiff.append(dict(Configuration=config,Model=md,Endpoint_Set=ep,Vegetation_First=cfg['Groups'][str(a)],Vegetation_Second=cfg['Groups'][str(b)],Difference=np.mean(z[ma])-np.mean(z[mb]),CI_Lower=lo,CI_Upper=hi,Bootstrap_Valid=nboot,CI_Excludes_0=bool(lo>0 or hi<0) if np.isfinite(lo) else np.nan,Limit='same basin endpoints and same block draws, fixed fitted model, exploratory unadjusted95%CI; not causal landcovereffect'))
 # Contribution decomposition retains OTHER classes; sums to existing basin mean.
 contributions=[]
 for config,md,ep,z in contrasts:
  if ep!='BASIN_P20_P80':continue
  for gid,name in [(1,'Forest'),(2,'Grassland_Shrubland'),(3,'Cropland'),(4,'All_Other_IGBP')]:
   m=g==gid;contributions.append(dict(Configuration=config,Model=md,Vegetation=name,N_Pixels=int(m.sum()),Pixel_Fraction=m.mean(),Area_Fraction=area[m].sum()/area.sum(),Mean_Contrast=np.mean(z[m]),Area_Weighted_Mean_Contrast=np.average(z[m],weights=area[m]),Contribution_To_Basin_PixelMean=np.sum(z[m])/N,Fraction_Of_Basin_SignedMean=np.sum(z[m])/np.sum(z),Full_Basin_Mean=np.mean(z),Limit='composition decomposition, not causal attribution; signed shares can exceed100%ornegative'))
 pd.DataFrame(results).to_csv(O/'M8_Stratified_Greening_Associations.csv',index=False,encoding='utf-8-sig');pd.DataFrame(endpoints).to_csv(O/'supplement/M8B/Greening_Endpoint_Support.csv',index=False);pd.DataFrame(errors).to_csv(O/'supplement/M8B/Stratified_OOF_Performance.csv',index=False);pd.DataFrame(correls).to_csv(O/'supplement/M8B/RootSM_Contrast_Relationship.csv',index=False);pd.DataFrame(curves).to_csv(O/'supplement/M8B/Empirical_Residual_Distributions.csv',index=False);pd.DataFrame(pairdiff).to_csv(O/'supplement/M8B/Paired_Type_Conditional_Contrasts.csv',index=False);pd.DataFrame(trainchecks).to_csv(O/'supplement/M8B/Training_Type_Endpoint_Ranges.csv',index=False);pd.DataFrame(contributions).to_csv(O/'supplement/M8B/Type_Composition_Decomposition.csv',index=False)
 pix=d[['row','col','Residual_damage','n_events','NIRv_trend','RootSM_base','RootSM_decile','greening_decile','moisture_regime']].copy();pix['IGBP_2001']=mode;pix['Vegetation_Group']=g;pix['ThreePeriod_Exact_Class']=exact;pix['Purity_GE0p75']=pure;pix.to_parquet(O/'paired/M8B/Frozen_Pixel_Type_Assignment.parquet',index=False)
 (O/'cache/M8B/GREENING_READY.json').write_text(json.dumps(dict(N_Records=len(results),N_OOF_Performance_Rows=len(errors),N_Type_Assigned_Model_Pixels=N,N_New_Model_Fits=0,Seconds=time.time()-start),indent=2),encoding='utf8');print('GREENING READY',len(results),round(time.time()-start,1),flush=True)
if __name__=='__main__':main()
