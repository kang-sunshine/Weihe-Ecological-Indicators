# -*- coding: utf-8 -*-
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,time,hashlib
import numpy as np
import pandas as pd
from minor2_core import R,O,M1
IND=['Acomp','NIRv','GPP']
def fit(x,y,n,c=None):
 X=np.column_stack([np.ones(len(x)),x]+([] if c is None else [np.maximum(0,x-c)]));w=np.sqrt(n);b=np.linalg.lstsq(X*np.sqrt(w)[:,None],y*np.sqrt(w),rcond=None)[0];rss=np.sum(w*(y-X@b)**2);a=len(x)*np.log(max(rss,1e-300)/len(x))+2*(2 if c is None else 4);return b,a
def sample(q,eid,rng):
 z=np.empty(0,np.int64);chunks=(eid[q]-1)//1000000
 for c in np.unique(chunks):
  cur=q[chunks==c]
  if len(cur)>20000:cur=cur[rng.choice(len(cur),20000,replace=False)]
  z=np.concatenate([z,cur])
  if len(z)>20000:z=z[rng.choice(len(z),20000,replace=False)]
 return z
def main():
 assert (O/'cache/CONTROLS_READY.json').exists();start=time.time();out=[];bins=[];curves=[];hist=pd.read_csv(R/'Weihe database V3/L5_EventEcoFunction/L5_4_ThresholdAnalysis/L5_4A_Severity_Response_Thresholds.csv');allcand=pd.read_csv(M1/'R1M1_Breakpoint_Candidate_AIC.csv');allcand=allcand[(allcand.Analysis=='SAVED_BIN_historical')&(allcand.Response=='Acomp_legacy')]
 for hi,h in enumerate(['CDH','HEAT','HR']):
  folder=O/'cache'/h;e=pd.read_parquet(O/'paired'/f'{h}_Fixed_Clean_Events.parquet',columns=['event_num','row','col','climate_cumulative_severity']);s=e.climate_cumulative_severity.to_numpy();eid=e.event_num.to_numpy(int);pix=e.row.to_numpy(int)*703+e.col.to_numpy(int);blocks=np.load(folder/'Block50.npy',mmap_mode='r');full=np.load(M1/'cache'/h/'all_severity_sorted.npy',mmap_mode='r');m=[np.load(folder/(n+'.npy'),mmap_mode='r') for n in IND];strict=[np.load(folder/(n+'_CommonPhase.npy'),mmap_mode='r') for n in IND];mask=np.load(folder/'Matched_Cohort_Masks.npz');edgesfile=R/f'EI_Reviewer1_Major8_Revision/cache/M8B/{h}/Original_Historical_Bin_Edges.npy';edges=np.load(edgesfile);bn=np.clip(np.searchsorted(edges,s,side='right')-1,0,len(edges)-2);cand=np.sort(allcand[allcand.Event==h].Candidate_S.unique());old=hist[(hist.event_type==h)&(hist.response=='Acomp_legacy')].iloc[0]
  cases=[('NATIVE_OWN',m,None,12,'M3_M5_RULE_OBSERVED_POSTTERM_LEGACY'),('MATCHED_NATIVE',m,mask['Matched_Native'],12,'M3_M5_RULE_OBSERVED_POSTTERM_LEGACY'),('MATCHED_COMMON_PHASE',strict,mask['Matched_CommonPhase'],12,'M3_M5_RULE_OBSERVED_POSTTERM_LEGACY'),('MATCHED_HISTORICAL_LEGACY_SECONDARY',m,mask['Matched_Native'],14,'ORIGINAL_L53_ONSET120_ACTIVE_NEXT_LEGACY')]
  for cohort,mm,common,f,version in cases:
   joint=(common&np.logical_and.reduce([np.isfinite(v[:,f]) for v in mm])) if common is not None else None
   for k,ind in enumerate(IND):
    pop=np.isfinite(mm[k][:,f]) if joint is None else joint;x=[];y=[];ns=[];rng=np.random.default_rng(20260908+hi*100);fid=h+'_'+cohort+'_'+ind
    for j in range(len(edges)-1):
     q=np.flatnonzero(pop&(bn==j))
     if len(q)<500:continue
     take=sample(q,eid,rng);x.append(float(np.median(s[take])));y.append(float(np.median(mm[k][take,f])));ns.append(len(q));bins.append(dict(Fit_ID=fid,Event=h,Cohort=cohort,Indicator=ind,Bin=j,N_Events=len(q),N_Sampled=len(take),Severity_Median=x[-1],Legacy_Median=y[-1],N_50km_Blocks=len(np.unique(blocks[q])),Sample_Seed=20260908+hi*100,Sample_ID_SHA256=hashlib.sha256(eid[take].tobytes()).hexdigest(),Edges_Source=str(edgesfile),Response_Version=version))
    x=np.asarray(x);y=np.asarray(y);ns=np.asarray(ns);eligible=np.array([c for c in cand if (x<c).sum()>=4 and (x>c).sum()>=4]);commonrow=dict(Event=h,Cohort=cohort,Indicator=ind,Response_Version=version,Original_Fixed_Reference_S=old.breakpoint_severity,Original_Source_Delta_AIC=old.delta_AIC_linear_minus_segmented,N_Events=int(pop.sum()),N_Pixels=len(np.unique(pix[pop])),N_50km_Blocks=len(np.unique(blocks[pop])),N_Bins=len(x),Candidate_Min_Original=cand.min(),Candidate_Max_Original=cand.max(),N_Eligible_Candidates=len(eligible),Q_Reference='unchangedfulloriginalhazardcataloguefloat64rightcontinuousECDF;tiedseveritynotjittered',N_Full_ECDF=len(full),Location_Uncertainty='NOT COMPUTED:pointfitonly; no newthresholdclaim',Limit='legacy scales different, slopes not physicalincrement; newM5-rule response differsfromhistoricalL53; originalS/AICkeptasinputonly; frozenoriginalhazardbinsfromauditedM8B/M1')
    for method,cc in [('FIXED_ORIGINAL_BASIN_REFERENCE',np.array([old.breakpoint_severity])),('FREE_ORIGINAL_CANDIDATE_GRID',eligible)]:
     cc=np.array([c for c in cc if (x<c).sum()>=4 and (x>c).sum()>=4]);r=dict(**commonrow,Analysis=method)
     if len(x)<12 or len(np.unique(blocks[pop]))<10 or not len(cc) or np.ptp(y)<=1e-12:r.update(Status='NOT ESTIMABLE',Missing_Reason='insufficient bins/spatial support/tailbins or constantbinned response');out.append(r);continue
     b0,a0=fit(x,y,ns);ff=[fit(x,y,ns,c) for c in cc];aa=np.array([t[1] for t in ff]);j=int(np.argmin(aa));st=cc[j];b,a=ff[j];delta=a0-a;edge=bool(np.isclose(st,cand.min(),rtol=0,atol=1e-12) or np.isclose(st,cand.max(),rtol=0,atol=1e-12));subedge=bool(method.startswith('FREE') and (st==eligible.min() or st==eligible.max()));r.update(S_star=st,Q_Full_Hazard=100*np.searchsorted(full,st,side='right')/len(full),Q_Matched_Response_Auxiliary=100*np.mean(s[pop]<=st),Linear_AIC=a0,Segmented_AIC=a,Delta_AIC=delta,Pre_Slope=b[1],Post_Slope=b[1]+b[2],Slope_Change=b[2],Original_Edge_Hit=edge,Supported_Subrange_Edge_Hit=subedge,Support_Level='STRONG' if delta>=10 else 'MODERATE' if delta>=4 else 'UNSUPPORTED',Status='NO_SUPPORTED_BREAKPOINT' if delta<4 else 'EDGE_CONSTRAINED' if edge or subedge else 'CONDITIONAL_POINT_CANDIDATE')
     out.append(r)
     if method.startswith('FREE'):
      for st0,a1 in zip(cc,aa):curves.append(dict(Fit_ID=fid,Candidate_S=st0,AIC=a1,Delta_AIC_Linear=a0-a1,Delta_AIC_From_Best=a1-a))
   print('SEVERITY',h,cohort,'seconds',round(time.time()-start,1),flush=True)
 pd.DataFrame(out).to_csv(O/'Minor2_Severity_Comparison.csv',index=False,encoding='utf-8-sig');pd.DataFrame(bins).to_csv(O/'supplement/Severity_Legacy_Bins.csv',index=False);pd.DataFrame(curves).to_csv(O/'supplement/Severity_Candidate_AIC.csv',index=False);(O/'cache/SEVERITY_READY.json').write_text(json.dumps(dict(N_Fits=len(out),Seconds=time.time()-start,No_Original_Boundary_Changed=True,No_Expanded_Search=True),indent=2),encoding='utf8');print('SEVERITY READY',flush=True)
if __name__=='__main__':main()
