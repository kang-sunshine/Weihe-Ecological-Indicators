# -*- coding: utf-8 -*-
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import json,ast,time,gc,warnings,sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from numba import njit
from minor2_core import R,O,M1,M5,F
warnings.filterwarnings('ignore',category=RuntimeWarning)
IND=['Acomp','NIRv','GPP'];GROUPS=['Low','Moderate','High','Extreme','All']
@njit(cache=True)
def ranks_from_order(x,order):
 out=np.empty(len(x),np.float32);i=0
 while i<len(x):
  j=i+1
  while j<len(x) and x[order[j]]==x[order[i]]:j+=1
  rank=.5*(i+1+j)
  for k in range(i,j):out[order[k]]=rank
  i=j
 return out
@njit(cache=True)
def centred_rank_correlation(a,b):
 mu=(len(a)+1)/2.;xy=0.;xx=0.;yy=0.;cxy=0.;cxx=0.;cyy=0.
 for i in range(len(a)):
  da=float(a[i])-mu;db=float(b[i])-mu
  p=da*db-cxy;t=xy+p;cxy=(t-xy)-p;xy=t
  p=da*da-cxx;t=xx+p;cxx=(t-xx)-p;xx=t
  p=db*db-cyy;t=yy+p;cyy=(t-yy)-p;yy=t
 return xy/np.sqrt(xx*yy) if xx>0 and yy>0 else np.nan
def exact_spearman_low_memory(a,b):
 assert len(a)==len(b) and len(a)<2**23
 rx=ranks_from_order(a,np.argsort(a));ry=ranks_from_order(b,np.argsort(b));return float(centred_rank_correlation(rx,ry))
def statistics(x):
 x=np.asarray(x);x=x[np.isfinite(x)]
 if not len(x):return dict(N_Valid=0,Mean=np.nan,SD=np.nan,Q05=np.nan,Q25=np.nan,Median=np.nan,Q75=np.nan,Q95=np.nan,Numeric_Zero_Fraction=np.nan)
 q=np.quantile(x,[.05,.25,.5,.75,.95]);return dict(N_Valid=len(x),Mean=float(np.mean(x,dtype=np.float64)),SD=float(np.std(x,dtype=np.float64)),Q05=q[0],Q25=q[1],Median=q[2],Q75=q[3],Q95=q[4],Numeric_Zero_Fraction=float(np.mean(abs(x)<=1e-12)))
def main():
 assert json.loads((O/'cache/CONTROLS_READY.json').read_text())['N_Hazards']==3;assert json.loads((O/'cache/ACOMP_GATE.json').read_text())['Pass'];start=time.time();cfg=json.loads((O/'code/configuration.json').read_text());fn=next(f for f in ast.parse((O/'audit/survival_m5.py').read_text()).body if isinstance(f,ast.FunctionDef) and f.name=='estimates');env={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<unchangedM5estimates>','exec'),env);estimate=env['estimates'];support=[];descriptions=[];paired=[];recovery=[];spatial=[];discordance=[];curves=[];replicas=[]
 test_a=np.array([0,0,1,3,4,4,8,-2],np.float32);test_b=np.array([3,3,2,4,-1,0,8,2],np.float32);err=abs(exact_spearman_low_memory(test_a,test_b)-spearmanr(test_a,test_b).statistic);assert err<1e-12
 (O/'audit/Low_Memory_Rank_Verification.json').write_text(json.dumps(dict(Definition='exactaverage-tieSpearman,n<2**23 sohalfintegerranksexactinfloat32;float64centredPearsonaccumulation',Vs_Scipy_Abs_Error=err,No_Subset_Sampling=True),indent=2),encoding='utf8')
 resume='--resume-heat' in sys.argv
 if resume:
  for path,dest in [('Minor2_Event_Support.csv',support),('Minor2_Matched_Recovery.csv',recovery),('Minor2_Spatial_Bootstrap.csv',spatial),('supplement/Paired_Event_Discordance.csv',discordance),('supplement/Daily_KM_AJ.csv',curves),('supplement/Paired_Recovery_Bootstrap_Replicates.csv',replicas)]:
   old=pd.read_csv(O/path);assert set(old.Event)=={'CDH'};dest.extend(old.to_dict('records'))
  old=pd.read_csv(O/'Minor2_Matched_Damage_Legacy.csv');descriptions=old[old.Cohort=='NATIVE_RESPONSE_SPECIFIC'].to_dict('records');paired=old[old.Cohort!='NATIVE_RESPONSE_SPECIFIC'].to_dict('records')
 for h in ['HEAT','HR'] if resume else ['CDH','HEAT','HR']:
  folder=O/'cache'/h;e=pd.read_parquet(O/'paired'/f'{h}_Fixed_Clean_Events.parquet',columns=['row','col','event_num']).astype({'row':'int16','col':'int16','event_num':'int32'});N=len(e);pix=e.row.to_numpy(np.int32)*703+e.col.to_numpy(np.int32);q=np.load(folder/'Full_ECDF_Q.npy',mmap_mode='r');group=np.searchsorted([50,75,90],q,side='right');blocks=np.load(folder/'Block50.npy',mmap_mode='r');B=int(np.load(M5/'cache'/f'{h}_pixel_blocks_50.npy',mmap_mode='r').max()+1);W=np.random.default_rng(cfg['Seed']+50).multinomial(B,np.ones(B)/B,size=cfg['Bootstrap_N']);np.save(folder/'Bootstrap_50km_Weights.npy',W);m=[np.load(folder/(n+'.npy'),mmap_mode='r') for n in IND];strict=[np.load(folder/(n+'_CommonPhase.npy'),mmap_mode='r') for n in IND];valid=[v[:,9]==1 for v in m];common=np.logical_and.reduce(valid);commonbase=np.logical_and.reduce([np.isfinite(v[:,1]) for v in m]);positive=common&np.logical_and.reduce([v[:,1]>0 for v in m]);scommon=common&np.logical_and.reduce([v[:,9]==1 for v in strict]);np.savez_compressed(folder/'Matched_Cohort_Masks.npz',Common_Base_Peak=commonbase,Matched_Native=common,Matched_All_Positive=positive,Matched_CommonPhase=scommon)
  unpolluted=common&np.logical_and.reduce([v[:,17]==0 for v in m]);np.save(folder/'Matched_Uncontaminated_Peaks.npy',unpolluted)
  cohorts=[('NATIVE_OWN_VALID',m,None),('MATCHED_NATIVE',m,common),('MATCHED_ALL_POSITIVE',m,positive),('NATIVE_AT_COMMON_PHASE_COHORT',m,scommon),('MATCHED_COMMON_PHASE',strict,scommon),('MATCHED_ALL_PEAKS_BEFORE_NEXT',m,unpolluted)]
  def blockcounts(mask,weights=None):return np.bincount(blocks[mask],weights=None if weights is None else weights[mask],minlength=B)
  def ratio(num,den):
   nn=blockcounts(num);dd=blockcounts(den);pt=nn.sum()/dd.sum() if dd.sum() else np.nan;bden=W@dd;vv=np.divide(W@nn,bden,out=np.full(len(W),np.nan),where=bden>0);return float(pt),vv
  def add_interval(row,values):
   ok=np.isfinite(values);n=int(ok.sum());pt=row['Estimate'];ci=np.quantile(values[ok],[.025,.975]) if n>=cfg['Minimum_Valid_Bootstrap'] and np.isfinite(pt) else [np.nan,np.nan];row.update(CI_Lower=ci[0],CI_Upper=ci[1],Bootstrap_Valid=n,Bootstrap_N=len(W),Bootstrap_Block_km=50,CI_Status='ESTIMABLE_CONDITIONAL' if np.isfinite(ci).all() else 'NOT ESTIMABLE');spatial.append(row.copy());return row
  for k,ind in enumerate(IND):
   v=m[k]
   for gi,label in list(enumerate(GROUPS[:4]))+[(4,'All')]:
    pop=(group==gi) if gi<4 else np.ones(N,bool);base=pop&(v[:,4]>=3);peak=base&np.isfinite(v[:,1]);follow=pop&valid[k];known=follow&((v[:,8]==0)|((v[:,8]==2)&(v[:,7]>=120)));leg=pop&np.isfinite(v[:,12]);support.append(dict(Event=h,Scope='NATIVE_OWN',Indicator=ind,Severity_Group=label,N_Source_Clean_Events=int(pop.sum()),N_Pre40_AtLeast3=int(base.sum()),N_Baseline_And_Peak=int(peak.sum()),N_Valid_Followup=int(follow.sum()),N_Positive_Damage=int((peak&(v[:,1]>0)).sum()),N_Finite_Observed_Legacy=int(leg.sum()),N_Legacy_Undefined_No_Observed_Interval=int((peak&~np.isfinite(v[:,12])&(v[:,13]==0)).sum()),N_Known_120_Status=int(known.sum()),N_risk_120=int((follow&(v[:,7]>=120)).sum()),N_Damage_Pixels=len(np.unique(pix[peak])),N_Followup_Pixels=len(np.unique(pix[follow])),N_50km_Blocks=int(np.count_nonzero(blockcounts(follow))),Pre_N_Median=float(np.median(v[peak,4])) if peak.any() else np.nan,Peak_N_Median=float(np.median(v[peak,5])) if peak.any() else np.nan,N_Peak_AtOrAfter_Next=int((peak&(v[:,17]==1)).sum()),Definition='nativeQA,ownbaseline/peak/support; recoveryM5, no new positiveDamage gate; counts not independent1km replicates'))
   for field,label in [(1,'Damage'),(2,'Peak_Lag_Days'),(12,'Legacy_Observed'),(14,'Legacy_Historical_Secondary')]:
    mask=np.isfinite(v[:,field]);descriptions.append(dict(Event=h,Cohort='NATIVE_RESPONSE_SPECIFIC',Indicator=ind,Metric=label,N_Source_Events=N,N_Response_Pixels=len(np.unique(pix[mask])),N_50km_Blocks=int(np.count_nonzero(blockcounts(mask))),**statistics(v[mask,field]),Limit='own valid population; cannot use differences between indicators as matched effect; SD is dispersion, not spatialSE'))
  for cohort,mm,cohortmask in cohorts:
   masks=[v[:,9]==1 for v in mm] if cohortmask is None else [cohortmask]*3
   if cohortmask is not None:
    for gi,label in list(enumerate(GROUPS[:4]))+[(4,'All')]:
     pop=cohortmask&((group==gi) if gi<4 else True);support.append(dict(Event=h,Scope=cohort,Indicator='ALL3_IDENTICAL_IDS',Severity_Group=label,N_Source_Clean_Events=N,N_Pre40_AtLeast3=int(pop.sum()),N_Baseline_And_Peak=int(pop.sum()),N_Valid_Followup=int(pop.sum()),N_Followup_Pixels=len(np.unique(pix[pop])),N_50km_Blocks=int(np.count_nonzero(blockcounts(pop))),N_Finite_Observed_Legacy=int((pop&np.logical_and.reduce([np.isfinite(v[:,12]) for v in mm])).sum()),Definition='sameeventIDsfor3indicators; '+('nativephases' if cohort!='MATCHED_COMMON_PHASE' else 'samefinitephasesbeforebaseline/peak; originalZparametersunchanged')))
    for field,label in [(0,'Baseline'),(1,'Damage'),(2,'Peak_Lag_Days'),(12,'Legacy_Observed'),(14,'Legacy_Historical_Secondary')]:
     match=cohortmask&np.logical_and.reduce([np.isfinite(v[:,field]) for v in mm]);values=[np.asarray(v[match,field]) for v in mm]
     for k,ind in enumerate(IND):paired.append(dict(Event=h,Cohort=cohort,Row_Type='MATCHED_DISTRIBUTION',Indicator=ind,Metric=label,N_Common_Events=int(match.sum()),N_Common_Pixels=len(np.unique(pix[match])),N_50km_Blocks=int(np.count_nonzero(blockcounts(match))),**statistics(values[k]),Unit='days' if field==2 else 'indicator_anomaly_days' if field in [12,14] else 'indicator_anomaly',Limit='sameIDs; eachownbaseline; no cross-scale amplitude superiority; legacyversion separated'))
     if field in [1,2,12,14]:
      for k in [1,2]:
       a=values[0];b=values[k];rho=exact_spearman_low_memory(a,b) if len(a)>1 else np.nan;row=dict(Event=h,Cohort=cohort,Row_Type='MATCHED_PAIR_RANK',Indicator='Acomp_vs_'+IND[k],Metric=label,N_Common_Events=len(a),Spearman_Rho=rho,Limit='exactaverage-tieeventranks, no sample or naive independent-event pvalue; amplitude differences not ecological gain')
       if field==2:row.update(Paired_Lag_Difference_Mean_Acomp_Minus_Single=float(np.mean(a-b)),Paired_Lag_Difference_Median_Acomp_Minus_Single=float(np.median(a-b)),Fraction_Abs_Peak_Time_Difference_GE8=float(np.mean(abs(a-b)>=8)))
       paired.append(row)
   # Each indicator has same matched mask; native own descriptive masks differ.
   C=np.zeros((B,3,4,5,121),np.int64)
   for k,v in enumerate(mm):
    mask=masks[k];flat=(((blocks[mask]*4+group[mask])*5+v[mask,8].astype(int))*121+v[mask,7].astype(int));C[:,k]=np.bincount(flat,minlength=B*4*5*121).reshape(B,4,5,121)
   C=np.concatenate([C,C.sum(axis=2,keepdims=True)],axis=2);pt=estimate(C.sum(axis=0));fb=estimate((W@C.reshape(B,-1)).reshape(len(W),3,5,5,121))
   for k,ind in enumerate(IND):
    for gi,label in enumerate(GROUPS):
     records=C[:,k,gi].sum(axis=0);n=int(records.sum());events=records.sum(axis=1);nr=int(pt['N_risk_120'][k,gi]);censor=1-events[0]/n if n else np.nan
     for metric in ['P120','RMPT120','AJ_Recovery120','AJ_Next120']:
      row=dict(Event=h,Cohort=cohort,Row_Type='GROUP_ESTIMATE',Indicator=ind,Group_Or_Contrast=label,Metric=metric,Estimate=float(pt[metric][k,gi]),N_Events=n,N_Recovered=int(events[0]),N_Next_Interrupted=int(events[1]),N_Admin120=int(events[2]),N_RecordEnd=int(events[3]),N_Missing_Censored=int(events[4]),Censoring_Rate=censor,N_risk_120=nr,Unit='days' if metric=='RMPT120' else 'probability',Estimand='KM net underinterruptioncensoring' if metric in ['P120','RMPT120'] else 'AJ competingrisk CIF,not KM',Limit='M5PRIMARY;'+('ownnativepopulation,notmatched' if cohort=='NATIVE_OWN_VALID' else 'same matchedIDs; fixed rules; no causal or independent validation'))
      recovery.append(add_interval(row,fb[metric][:,k,gi]));
     if cohort in ['NATIVE_OWN_VALID','MATCHED_NATIVE','MATCHED_COMMON_PHASE']:
      for day in range(121):curves.append(dict(Event=h,Cohort=cohort,Indicator=ind,Group=label,Day=day,KM_S=float(pt['KM_S'][k,gi,day]),AJ_Recovery_CIF=float(pt['AJ_Recovery_Curve'][k,gi,day]),AJ_Next_CIF=float(pt['AJ_Next_Curve'][k,gi,day]),N_Risk=float(pt['Risk'][k,gi,day])))
    for metric in ['P120','RMPT120','AJ_Recovery120','AJ_Next120']:
     factor=1 if metric=='RMPT120' else 100;value=float((pt[metric][k,3]-pt[metric][k,2])*factor);v=(fb[metric][:,k,3]-fb[metric][:,k,2])*factor;row=dict(Event=h,Cohort=cohort,Row_Type='HIGH_EXTREME_CONTRAST',Indicator=ind,Group_Or_Contrast='Extreme_minus_High',Metric=metric,Estimate=value,N_Events=int(C[:,k,2:4].sum()),Unit='days' if metric=='RMPT120' else 'percentage_points',Limit='same bootstrapgroupdifferences; no CIendpointsubtraction; fixed fullhazardECDF')
     recovery.append(add_interval(row,v));
   if cohortmask is not None:
    for k in [1,2]:
     for metric in ['P120','RMPT120','AJ_Recovery120']:
      factor=1 if metric=='RMPT120' else 100
      for gi,label in enumerate(GROUPS):
       values=(fb[metric][:,0,gi]-fb[metric][:,k,gi])*factor;row=dict(Event=h,Cohort=cohort,Row_Type='INDICATOR_PAIR_DIFFERENCE',Indicator='Acomp_minus_'+IND[k],Group_Or_Contrast=label,Metric=metric,Estimate=float((pt[metric][0,gi]-pt[metric][k,gi])*factor),N_Events=int(C[:,0,gi].sum()),Unit='days' if metric=='RMPT120' else 'percentage_points',Limit='sameIDs,sameblockdraws,directdifference; fixedcomparison not externalvalidation');recovery.append(add_interval(row,values))
       if cohort in ['MATCHED_NATIVE','MATCHED_COMMON_PHASE'] and gi in [2,3,4]:
        for b,value in enumerate(values):replicas.append(dict(Event=h,Cohort=cohort,Indicator='Acomp_minus_'+IND[k],Metric=metric,Severity_Group=label,Replicate=b,Value=value))
      values=((fb[metric][:,0,3]-fb[metric][:,0,2])-(fb[metric][:,k,3]-fb[metric][:,k,2]))*factor;row=dict(Event=h,Cohort=cohort,Row_Type='DIFFERENCE_OF_HIGH_EXTREME_CONTRASTS',Indicator='Acomp_minus_'+IND[k],Group_Or_Contrast='Delta_Extreme_minus_High',Metric=metric,Estimate=float(((pt[metric][0,3]-pt[metric][0,2])-(pt[metric][k,3]-pt[metric][k,2]))*factor),N_Events=int(C[:,0,2:4].sum()),Unit='days' if metric=='RMPT120' else 'percentage_points',Limit='directpaireddifferenceofcontrasts fromsamereplicate');recovery.append(add_interval(row,values))
    # Discordance ratios: censor-unknown is never counted as failure at120.
    finiteleg=cohortmask&np.logical_and.reduce([np.isfinite(v[:,12]) for v in mm]);known=cohortmask&np.logical_and.reduce([(v[:,8]==0)|((v[:,8]==2)&(v[:,7]>=120)) for v in mm])
    for k in [1,2]:
     a=mm[0];b=mm[k];a_pos=np.isfinite(a[:,12])&(a[:,12]>1e-12);b_pos=np.isfinite(b[:,12])&(b[:,12]>1e-12);rec_a=a[:,8]==0;rec_b=b[:,8]==0;order_b=cohortmask&rec_b&((rec_a&(b[:,10]<a[:,10]))|(~rec_a&(a[:,7]>b[:,10])));order_a=cohortmask&rec_a&((rec_b&(a[:,10]<b[:,10]))|(~rec_b&(b[:,7]>a[:,10])))
     definitions=[('Single_Recovered_Acomp_Not_By120',known&rec_b&~rec_a,known,'3indicatorknown120subset; recovered oradmin120; excludesunknowncensoring'),('Acomp_Recovered_Single_Not_By120',known&rec_a&~rec_b,known,'3indicatorknown120subset; selectionconditional,not populationP120'),('Single_Recovery_Observed_Earlier',order_b,cohortmask,'strictlyearlierformalrecovery; otherstillobservedbeyondthatday; not120failure'),('Acomp_Recovery_Observed_Earlier',order_a,cohortmask,'reverseobservedorder, notassumingunknowncensorisfailure'),('Acomp_Legacy_Positive_Single_Zero',finiteleg&a_pos&(b[:,12]<=1e-12),finiteleg,'all3finitelegacy; numericzeronotstatisticalabsence'),('Single_Legacy_Positive_Acomp_Zero',finiteleg&b_pos&(a[:,12]<=1e-12),finiteleg,'oppositediscordance, no scalecomparison'),('Single_Zero_Given_Acomp_Positive_Legacy',finiteleg&a_pos&(b[:,12]<=1e-12),finiteleg&a_pos,'conditionalonpositiveAcompandall3observedlegacy'),('Acomp_Positive_Single_Legacy_NoWindow_Recovered',cohortmask&a_pos&~np.isfinite(b[:,12])&rec_b,cohortmask,'singlelegacyundefinedbecauseearlyobservedrecovery/nointegrationinterval,notdataomission'),('Acomp_Positive_Single_Legacy_Undefined_Other',cohortmask&a_pos&~np.isfinite(b[:,12])&~rec_b,cohortmask,'insufficient/censoredinterval; cannotclaimpersistentanomalyomission')]
     # Rank-based weak signal diagnostic avoids cross-indicator absolute units.
     if finiteleg.any() and (finiteleg&a_pos).any() and (finiteleg&b_pos).any():
      aq=np.quantile(a[finiteleg&a_pos,12],.75);bq=np.quantile(b[finiteleg&b_pos,12],.25);definitions.append(('Acomp_OwnPositive_Q75_Single_BelowOwnPositive_Q25',finiteleg&(a[:,12]>=aq)&(b[:,12]<=bq),finiteleg,'ownmatchedpositivequartiles,rankdiagnostic not physicalincrement'))
     for name,num,den,limit in definitions:
      for gi,label in [(4,'All'),(2,'High'),(3,'Extreme')]:
       pop=(group==gi) if gi<4 else np.ones(N,bool);dd=den&pop;nn=num&pop;value,values=ratio(nn,dd);row=dict(Event=h,Cohort=cohort,Row_Type='PAIRED_EVENT_PROPORTION',Indicator='Acomp_vs_'+IND[k],Group_Or_Contrast=label,Metric=name,Estimate=value,N_Numerator=int(nn.sum()),N_Denominator=int(dd.sum()),N_50km_Blocks=int(np.count_nonzero(blockcounts(dd))),Unit='fraction',Limit=limit);discordance.append(add_interval(row,values))
   print('STATS',h,cohort,'seconds',round(time.time()-start,1),flush=True);del C,pt,fb;gc.collect()
  # Progress saved perhazard; only READY signifies complete3hazards.
  pd.DataFrame(support).to_csv(O/'Minor2_Event_Support.csv',index=False,encoding='utf-8-sig');pd.DataFrame(descriptions+paired).to_csv(O/'Minor2_Matched_Damage_Legacy.csv',index=False,encoding='utf-8-sig');pd.DataFrame(recovery).to_csv(O/'Minor2_Matched_Recovery.csv',index=False,encoding='utf-8-sig');pd.DataFrame(spatial).to_csv(O/'Minor2_Spatial_Bootstrap.csv',index=False,encoding='utf-8-sig');pd.DataFrame(discordance).to_csv(O/'supplement/Paired_Event_Discordance.csv',index=False);pd.DataFrame(curves).to_csv(O/'supplement/Daily_KM_AJ.csv',index=False);pd.DataFrame(replicas).to_csv(O/'supplement/Paired_Recovery_Bootstrap_Replicates.csv',index=False)
  del e,m,strict,valid,common,commonbase,positive,scommon,cohorts,group,blocks,pix,q;gc.collect()
 (O/'cache/STATISTICS_READY.json').write_text(json.dumps(dict(N_Hazards=3,N_Recovery_Rows=len(recovery),N_Paired_Proportions=len(discordance),Bootstrap_N=250,Seconds=time.time()-start),indent=2),encoding='utf8');print('STATISTICS READY',round(time.time()-start,1),flush=True)
if __name__=='__main__':main()
