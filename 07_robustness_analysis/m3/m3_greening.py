from pathlib import Path
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import json,time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score,mean_squared_error
from scipy.stats import spearmanr
O=Path(r'.\EI_Reviewer1_Major3_Revision');D=Path(r'.\Weihe database V3');L=D/'L5_EventEcoFunction';FEAT=['NIRv_trend','RootSM_base','DeepSM_base','VPD_base','Tmax_base','Temp_base','Precip_base','RAD_base']
KW=dict(loss='squared_error',learning_rate=.05,max_iter=280,max_leaf_nodes=15,min_samples_leaf=100,l2_regularization=.15,random_state=42)

def main():
 start=time.time();assert (O/'cache/EXTRACTION_READY.json').exists()
 ev=pd.read_parquet(O/'paired/CDH_Original_Clean_Cohort.parquet');m=np.load(O/'cache/Event_Metrics.npy',mmap_mode='r');oldmodel=pd.read_csv(L/'L6_SeverityAdjustedResidual_FINAL_V2/L6_0_CDH_EventLevel_FixedBreakpoint_Model_FINAL_V2.csv').iloc[0];star=oldmodel.breakpoint_severity
 # Preserve the original formal event IDs (clean, valid, finite, original damage>0).
 # The new signed increment may be <=0: do not drop the very events whose
 # apparent damage was carried by removed NIRv, or change the pixel universe.
 keep=ev.valid_resistance_sample.eq(1).to_numpy()&np.isfinite(ev.Acomp_damage)&ev.Acomp_damage.gt(0)&np.isfinite(ev.climate_cumulative_severity)
 keep=np.asarray(keep);assert keep.sum()==int(oldmodel.n_events)
 assert np.isfinite(m[keep,1,1]).all()
 err=np.abs(m[keep,0,1]-ev.Acomp_damage.to_numpy()[keep]);assert err.max()<1e-6,err.max()
 e=ev.loc[keep].copy();S=e.climate_cumulative_severity.to_numpy();Xn=np.column_stack([np.ones(len(e)),S,np.maximum(0,S-star)])
 betas=[];pixels=[];suff=[]
 for k,var in enumerate(['Acomp_original','Acomp_NIRv_free']):
  yy=e.Acomp_damage.to_numpy() if k==0 else m[keep,1,1];beta=np.linalg.solve(Xn.T@Xn,Xn.T@yy);res=yy-Xn@beta
  betas.append(beta);suff.append(dict(Variant=var,N_Events=len(e),S_star=star,Intercept=beta[0],Slope=beta[1],Hinge=beta[2],Event_Response_SD=float(np.std(yy)),Event_Residual_SD=float(np.std(res)),Nonpositive_Damage_N=int((yy<=0).sum()),Selection='FIXED_ORIGINAL_FORMAL_POSITIVE_Acomp_IDS; NIRv-free signed increments retained'))
  temp=pd.DataFrame(dict(row=e.row.to_numpy(),col=e.col.to_numpy(),res=res));pix=temp.groupby(['row','col'],sort=True).agg(Residual_damage=('res','mean'),n_events=('res','size')).reset_index();pix=pix[pix.n_events>=3].copy();pixels.append(pix)
 assert np.allclose(betas[0],[oldmodel.intercept,oldmodel.pre_break_slope,oldmodel.hinge_slope_change],rtol=1e-9,atol=1e-9)
 pd.DataFrame(suff).to_csv(O/'paired/Fixed_Break_Severity_Models.csv',index=False)
 source=pd.read_csv(L/'L6_SeverityAdjustedResidual_FINAL_V2/L6_0_CDH_Pixel_ResidualDamage_GreeningHydroclimate_FINAL_V2.csv.gz');old=pd.read_csv(L/'L6_Mechanisms/L6_2_SpatialML/L6_2_PixelLevel_ResilienceDriverDataset.csv.gz',usecols=['row','col']+FEAT[2:]);source=source.merge(old,on=['row','col'],validate='one_to_one');source=source.dropna(subset=['Residual_damage','block20']+FEAT).copy()
 # Preserve source ordering: GroupKFold allocates the same published blocks.
 source['Original_Source_Order']=np.arange(len(source));source['Formal_Fold']=0
 for fold,(tr,te) in enumerate(GroupKFold(5).split(source[FEAT],source.Residual_damage,source.block20),1):source.iloc[te,source.columns.get_loc('Formal_Fold')]=fold
 srcN=len(source);d=source.merge(pixels[0].rename(columns={'Residual_damage':'Original_Recomputed','n_events':'N_Common_Events'}),on=['row','col'],validate='one_to_one').merge(pixels[1][['row','col','Residual_damage']].rename(columns={'Residual_damage':'NIRv_Free'}),on=['row','col'],validate='one_to_one').sort_values('Original_Source_Order').reset_index(drop=True)
 assert len(d)==srcN==153932 and np.max(np.abs(d.Residual_damage-d.Original_Recomputed))<1e-8
 pixel_reconstruction_error=float(np.max(np.abs(d.Residual_damage-d.Original_Recomputed)))
 # Preserve the exact published original target vector after validating its
 # reconstruction. Sub-picounit OLS accumulation differences can alter tree ties.
 d['Original_Recomputed']=d.Residual_damage.to_numpy()
 d.to_parquet(O/'paired/Greening_Common_Pixel_Inputs.parquet',index=False)
 XX=d[FEAT].to_numpy();folds=d.Formal_Fold.to_numpy();G20,G80=np.quantile(d.NIRv_trend,[.2,.8]);Xlo=XX.copy();Xhi=XX.copy();Xlo[:,0]=G20;Xhi[:,0]=G80
 results=[];perms=[];foldrows=[];preds=d[['row','col','block20','Formal_Fold','N_Common_Events','RootSM_base','NIRv_trend']].copy();benefits=[]
 for v,target in enumerate(['Original_Recomputed','NIRv_Free']):
  y=d[target].to_numpy();sd=np.std(y,ddof=0);oof=np.full(len(y),np.nan);oofB=oof.copy()
  for fold in range(1,6):
   tr=folds!=fold;te=~tr;model=HistGradientBoostingRegressor(**KW).fit(XX[tr],y[tr]);pred=model.predict(XX[te]);oof[te]=pred;oofB[te]=model.predict(Xlo[te])-model.predict(Xhi[te]);base=np.mean((y[te]-pred)**2);sst=np.mean((y[te]-y[te].mean())**2)
   foldrows.append(dict(Variant=target,Fold=fold,N_Train=int(tr.sum()),N_Test=int(te.sum()),OOF_R2=r2_score(y[te],pred),RMSE=np.sqrt(base),Response_SD=sd,RMSE_over_Response_SD=np.sqrt(base)/sd))
   rng=np.random.default_rng(3000+fold)
   for f in range(len(FEAT)):
    for repeat in range(3):
     xp=XX[te].copy();xp[:,f]=xp[rng.permutation(len(xp)),f];err=np.mean((y[te]-model.predict(xp))**2)
     perms.append(dict(Variant=target,Fold=fold,Feature=FEAT[f],Repeat=repeat,N_Test=len(xp),Delta_MSE=err-base,Delta_RMSE_over_SD=(np.sqrt(err)-np.sqrt(base))/sd,R2_Drop=(err-base)/sst))
   print(target,'fold',fold,'R2',round(foldrows[-1]['OOF_R2'],4),'sec',round(time.time()-start,1),flush=True)
  model=HistGradientBoostingRegressor(**KW).fit(XX,y);b=model.predict(Xlo)-model.predict(Xhi);benefits.append(b);rmse=np.sqrt(mean_squared_error(y,oof));sg=np.where(b>1e-12,1,np.where(b< -1e-12,-1,0));preds[target+'_Target']=y;preds[target+'_OOF']=oof;preds[target+'_Benefit']=b;preds[target+'_OOF_Benefit']=oofB;preds[target+'_Class']=sg
  results.append(dict(Row_Type='MODEL_SUMMARY',Variant=target,Status='COMPLETED',N_Pixels=len(d),N_Events=len(e),OOF_R2=r2_score(y,oof),RMSE=rmse,Response_SD=sd,RMSE_over_Response_SD=rmse/sd,Greening_Q20=G20,Greening_Q80=G80,Benefit_Mean=b.mean(),Benefit_Median=np.median(b),Benefit_Q05=np.quantile(b,.05),Benefit_Q95=np.quantile(b,.95),Benefit_Mean_over_SD=b.mean()/sd,Positive_Fraction=np.mean(sg==1),Negative_Fraction=np.mean(sg== -1),Numerical_Zero_Fraction=np.mean(sg==0),OOF_Benefit_Mean=oofB.mean(),OOF_Benefit_Mean_over_SD=oofB.mean()/sd,Source='Figure9 V5 framework; paired fixed original formal cohort; separate fixed-S* nuisance OLS per response',Limit='algebraic NIRv sharing removed from response; no complete statistical independence; numerical zero is not significance'))
 pd.DataFrame(foldrows).to_csv(O/'paired/Spatial_OOF_Folds.csv',index=False)
 pi=pd.DataFrame(perms);pi.to_csv(O/'paired/OOF_Permutation_Details.csv',index=False)
 for var in ['Original_Recomputed','NIRv_Free']:
  imp=pi[pi.Variant.eq(var)].groupby('Feature').agg(Mean_R2_Drop=('R2_Drop','mean'),Delta_RMSE_over_SD=('Delta_RMSE_over_SD','mean')).reset_index();pos=np.maximum(0,imp.Mean_R2_Drop.to_numpy());imp['Relative_Positive_R2_Drop']=pos/pos.sum()
  for rr in imp.to_dict('records'):results.append(dict(Row_Type='OOF_PERMUTATION_IMPORTANCE',Variant=var,Status='COMPLETED',N_Pixels=len(d),**rr,Limit='held-out spatial-fold feature permutation; correlated predictors; relative positive drops normalized within model;3 repeats/fold'))
 # Fixed root-zone moisture deciles, same pixels and cuts across responses.
 d['Moisture_Decile']=pd.qcut(d.RootSM_base,10,labels=False,duplicates='drop')+1
 for var,b in zip(['Original_Recomputed','NIRv_Free'],benefits):
  sd=next(r['Response_SD'] for r in results if r['Row_Type']=='MODEL_SUMMARY' and r['Variant']==var);d['B']=b
  for dec,g in d.groupby('Moisture_Decile'):
   bm=g.groupby('block20').B.mean();se=bm.std(ddof=1)/np.sqrt(len(bm));results.append(dict(Row_Type='ROOTSM_MODERATION',Variant=var,Status='COMPLETED',Moisture_Decile=dec,N_Pixels=len(g),N_Blocks=len(bm),RootSM_Mean=g.RootSM_base.mean(),Benefit_Mean=g.B.mean(),Benefit_Mean_over_SD=g.B.mean()/sd,BlockMean_SE=se,Positive_Fraction=np.mean(g.B>1e-12),Negative_Fraction=np.mean(g.B< -1e-12),Numerical_Zero_Fraction=np.mean(np.abs(g.B)<=1e-12),Limit='block-mean SE is descriptive; no new bootstrap or causal moderation estimate'))
  rho=spearmanr(d.RootSM_base,b).statistic;results.append(dict(Row_Type='ROOTSM_RANK_TREND',Variant=var,Status='COMPLETED',N_Pixels=len(d),Spearman_Rho=rho,Limit='descriptive rank trend; no independent-pixel p value'))
 s0=preds.Original_Recomputed_Class.to_numpy();s1=preds.NIRv_Free_Class.to_numpy()
 for a in [-1,0,1]:
  for b in [-1,0,1]:results.append(dict(Row_Type='PIXEL_SIGN_TRANSITION',Variant='PAIRED',Status='COMPLETED',Original_Class=a,NIRv_Free_Class=b,N_Pixels=int(((s0==a)&(s1==b)).sum()),Fraction_All_Pixels=np.mean((s0==a)&(s1==b))))
 results.append(dict(Row_Type='SPATIAL_SIGN_AGREEMENT',Variant='PAIRED',Status='COMPLETED',N_Pixels=len(d),Fraction_All_Pixels=np.mean(s0==s1),Pearson_Benefit_Correlation=np.corrcoef(*benefits)[0,1],Limit='full-data fitted conditional contrasts; independent spatial OOF predictions also supplied'))
 hist=pd.read_csv(D/'Figure9_FINAL_V5_V2Residual/Figure9_ConditionalGreeningBenefit_FINAL_V5_PixelResults.csv.gz');j=preds.merge(hist[['row','col','GreeningBenefit','OOF_pred']],on=['row','col'],validate='one_to_one');check=dict(N=len(j),Damage_Reproduction_MaxAbs=float(np.max(np.abs(m[keep,0,1]-ev.Acomp_damage.to_numpy()[keep]))),Original_Pixel_Residual_MaxAbs=pixel_reconstruction_error,Original_Model_Target='exact saved formal pixel residual after numerical reconstruction check',Historical_Benefit_MaxAbs=float(np.max(np.abs(j.GreeningBenefit-j.Original_Recomputed_Benefit))),Historical_OOF_MaxAbs=float(np.max(np.abs(j.OOF_pred-j.Original_Recomputed_OOF))),Historical_Source_R2=.393028,Original_New_Nonpositive_N=int((m[keep,1,1]<=0).sum()),Model_KW=KW,Formal_Cohort_Selection='original clean & valid & original damage>0; same IDs and pixels retained in both responses')
 pd.DataFrame(results).to_csv(O/'M3_NIRv_Free_Greening_Comparison.csv',index=False,encoding='utf-8-sig');preds.to_parquet(O/'paired/Greening_Pixel_Contrasts.parquet',index=False);(O/'audit/Greening_Reproduction.json').write_text(json.dumps(check,indent=2),encoding='utf8')
 print('M3 GREENING READY',len(d),'seconds',round(time.time()-start,1),flush=True)
if __name__=='__main__':main()
