# -*- coding: utf-8 -*-
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,time,gc,sys
import numpy as np
import pandas as pd
from numba import njit
O=Path(r'.\EI_Reviewer1_Major8_Revision');R=O.parent;M1=R/'EI_Reviewer1_Major1_Revision'
@njit(cache=True)
def bootmedian(y,b,W):
 out=np.full(W.shape[0],np.nan)
 for j in range(W.shape[0]):
  total=0
  for k in range(len(y)):total+=W[j,b[k]]
  if total==0:continue
  lo=(total+1)//2;hi=total//2+1;cs=0;a=0.;got=False
  for k in range(len(y)):
   cs+=W[j,b[k]]
   if not got and cs>=lo:a=y[k];got=True
   if cs>=hi:out[j]=.5*(a+y[k]);break
 return out
def fit(x,y,n,c=None):
 X=np.column_stack([np.ones(len(x)),x]+([] if c is None else [np.maximum(0,x-c)]));w=np.sqrt(n);b=np.linalg.lstsq(X*np.sqrt(w)[:,None],y*np.sqrt(w),rcond=None)[0];rss=np.sum(w*(y-X@b)**2);return b,float(len(x)*np.log(max(rss,1e-300)/len(x))+2*(2 if c is None else 4))
def admissible(x,c):return (np.sum(x<c)>=4)&(np.sum(x>c)>=4)
def sampleids(q,eventnum,rng):
 selected=np.empty(0,np.int64);chunk=(eventnum[q]-1)//1000000
 for k in np.unique(chunk):
  u=q[chunk==k]
  if len(u)>20000:u=u[rng.choice(len(u),20000,replace=False)]
  selected=np.concatenate([selected,u])
  if len(selected)>20000:selected=selected[rng.choice(len(selected),20000,replace=False)]
 return selected
def qrank(a,c):return 100*np.searchsorted(a,c,side='right')/len(a) if len(a) else np.nan
def main():
 assert (O/'cache/M8B/EVENTS_READY.json').exists();start=time.time();extra='--extra-exact3-legacy' in sys.argv;cfg=json.loads((O/'code/M8B_configuration.json').read_text());lc=np.load(O/'cache/M8B/Landcover_Cohorts.npz');sp=np.load(O/'cache/M8B/Spatial_Grid.npz');B=len(sp['Block50_Labels']);NBOOT=cfg['Bootstrap_N'];W=np.random.default_rng(cfg['Seed']).multinomial(B,np.ones(B)/B,size=NBOOT).astype(np.int32);np.save(O/'cache/M8B/Shared_50km_Block_Multiplicities.npy',W);hist=pd.read_csv(R/'Weihe database V3/L5_EventEcoFunction/L5_4_ThresholdAnalysis/L5_4A_Severity_Response_Thresholds.csv');cand0=pd.read_csv(M1/'R1M1_Breakpoint_Candidate_AIC.csv');cand0=cand0[cand0.Analysis=='SAVED_BIN_historical'];result=[];binsout=[];curves=[];bootrows=[]
 if extra:
  old=pd.read_csv(O/'M8_Stratified_Breakpoints.csv');result=old[~((old.Cohort=='THREE_PERIOD_EXACT_CLASS')&(old.Response=='Acomp_legacy'))].to_dict('records')
  for path,dest in [('supplement/M8B/Stratified_Response_Bins.csv',binsout),('supplement/M8B/Stratified_Candidate_AIC.csv',curves)]:
   old=pd.read_csv(O/path);dest.extend(old[~(old.Fit_ID.str.contains('THREE_PERIOD_EXACT_CLASS')&old.Fit_ID.str.endswith('Acomp_legacy'))].to_dict('records'))
  old=pd.read_csv(O/'supplement/M8B/Stratified_Breakpoint_Bootstrap.csv');old['Cohort']=old['Cohort'] if 'Cohort' in old else 'MAIN_2001';bootrows=old[old.Cohort!='THREE_PERIOD_EXACT_CLASS'].to_dict('records')
 for hi,h in enumerate(['CDH'] if extra else ['CDH','HEAT','HR']):
  folder=O/'cache/M8B'/h;cache=M1/'cache'/h;r=np.load(cache/'row.npy',mmap_mode='r').astype(np.int32);c=np.load(cache/'col.npy',mmap_mode='r').astype(np.int32);pix=r*703+c;g=np.load(folder/'Clean_Group.npy');eid=np.load(folder/'Clean_Event_Num.npy');s=np.load(cache/'climate_cumulative_severity.npy',mmap_mode='r');damage=np.load(cache/'Acomp_damage.npy',mmap_mode='r');legacy=np.load(cache/'Acomp_legacy_anomaly_days.npy',mmap_mode='r');vr=np.load(cache/'valid_resistance_sample.npy',mmap_mode='r');edges=np.load(folder/'Original_Historical_Bin_Edges.npy');bn=np.clip(np.searchsorted(edges,s,side='right')-1,0,len(edges)-2);block=sp['Block50_Index'][r,c];assert block.min()>=0;clim=sp['Climate'][r,c];b100=sp['Block100'][r,c];fullbasin=np.load(cache/'all_severity_sorted.npy',mmap_mode='r')
  cohorts=[('MAIN_2001',i,n,g==i,f'Full_Severity_Group{i}.npy') for i,n in [(1,'Forest'),(2,'Grassland_Shrubland'),(3,'Cropland')]]
  if h=='CDH':
   cohorts += [('THREE_PERIOD_EXACT_CLASS',i,n,(g==i)&lc['Exact_ThreePeriod'][r,c],f'Full_Severity_Exact3_Group{i}.npy') for i,n in [(1,'Forest'),(2,'Grassland_Shrubland'),(3,'Cropland')]]
   cohorts += [('CROP14_INCLUSION_SENSITIVITY',3,'Cropland12plus14',np.isin(lc['Main_Mode'][r,c],[12,14]),'Full_Severity_Crop12plus14.npy')]
  for cohort,gid,name,gm,fullfile in cohorts:
   if extra and cohort!='THREE_PERIOD_EXACT_CLASS':continue
   full=np.load(folder/fullfile,mmap_mode='r');cleanS=np.sort(s[gm]);do_boot=cohort=='MAIN_2001' or (cohort=='THREE_PERIOD_EXACT_CLASS' and extra)
   for ri,(response,val,valid) in enumerate([('Damage_probability',damage,np.isfinite(damage)),('Acomp_damage',damage,np.isfinite(damage)&(damage>0)&(vr==1)),('Acomp_legacy',legacy,np.isfinite(legacy))]):
    if extra and response!='Acomp_legacy':continue
    m=gm&valid;id0=f'{h}_{cohort}_{name}_{response}';orig=hist[(hist.event_type==h)&(hist.response==response)].iloc[0];cand=np.sort(cand0[(cand0.Event==h)&(cand0.Response==response)].Candidate_S.unique());N=int(m.sum());nblo=len(np.unique(block[m]));bw=np.bincount(block[m],minlength=B);eff=1/np.sum((bw/N)**2) if N else np.nan;xb=[];yb=[];ns=[];by=[];bnboot=[];rng=np.random.default_rng(20260908+hi*100+gid*10+ri)
    for k in range(len(edges)-1):
     q=np.flatnonzero(m&(bn==k));n=len(q)
     if n<cfg['Minimum_Bin_N']:continue
     take=q if response=='Damage_probability' else sampleids(q,eid,rng);x=float(np.mean(s[take]) if response=='Damage_probability' else np.median(s[take]));y=float(np.mean(val[take]>0) if response=='Damage_probability' else np.median(val[take]));xb.append(x);yb.append(y);ns.append(n);nb=np.bincount(block[q],minlength=B);binsout.append(dict(Fit_ID=id0,Event=h,Cohort=cohort,Vegetation=name,Response=response,Bin=k,N_Events=n,N_Sampled=len(take),Severity_X=x,Response_Y=y,RSS_Weight=np.sqrt(n),N_Pixels=len(np.unique(pix[q])),N_50km_Blocks=int(np.count_nonzero(nb)),N_100km_Blocks=len(np.unique(b100[q])),N_Climate_Support=len(np.unique(clim[q])),Max_50km_Event_Share=float(nb.max()/n),Effective_50km_InvHHI=float(1/np.sum((nb/n)**2)),Sample_Seed=20260908+hi*100+gid*10+ri))
     if do_boot:
      nn=W@nb;bnboot.append(nn)
      if response=='Damage_probability':z=W@np.bincount(block[q],weights=(val[q]>0),minlength=B);by.append(np.divide(z,nn,out=np.full(NBOOT,np.nan),where=nn>0))
      else:order=np.argsort(val[take]);by.append(bootmedian(np.asarray(val[take][order]),block[take][order],W))
    xb=np.array(xb);yb=np.array(yb);ns=np.array(ns);eligible=np.array([v for v in cand if admissible(xb,v)]);common=dict(Fit_ID=id0,Event=h,Cohort=cohort,Vegetation=name,Response=response,Landcover_Year=2001,N_Events=N,N_Events_In_Used_Bins=int(ns.sum()),N_Pixels=len(np.unique(pix[m])),N_50km_Blocks=nblo,N_100km_Blocks=len(np.unique(b100[m])),N_Distributed_Climate_Support_Cells=len(np.unique(clim[m])),Effective_50km_InvHHI=eff,Max_50km_Event_Share=bw.max()/N if N else np.nan,N_Bins=len(xb),N_Candidate_Original=len(cand),N_Candidate_Admissible=len(eligible),Search_Min_Original=cand.min(),Search_Max_Original=cand.max(),Search_Min_Admissible=eligible.min() if len(eligible) else np.nan,Search_Max_Admissible=eligible.max() if len(eligible) else np.nan,Original_Basin_S_star=orig.breakpoint_severity,Original_Basin_Delta_AIC=orig.delta_AIC_linear_minus_segmented,Full_Type_ECDF_N=len(full),ECDF_Reference='ALL finite original '+h+' events at2001type pixels; '+cohort+' support; includes unclean;float64,right-continuous,ties unsplit',Clean_Type_ECDF_N=len(cleanS),Support_Rule='>=12 original bins each>=500events;>=10occupied50km blocks;>=4binpositions on each side of evaluated knot',Sampling='original bounded20k joint severity/response sample perbin using original1MEventNumchunks; probability fullcounts',Bootstrap_N=NBOOT if do_boot else 0,Block_km=50 if do_boot else np.nan,Limit='original2000-2020 normalization; historicalL5.3legacy, no new ecological critical-point claim; spatial bootstrap conditional on frozen binX and bounded sample')
    primaryok=len(xb)>=cfg['Minimum_Fit_Bins'] and nblo>=cfg['Minimum_Total_50km_Blocks'];a0=fit(xb,yb,ns)[1] if len(xb)>=2 else np.nan
    for method in ['A_FIXED_ORIGINAL_BASIN_CANDIDATE','B_FREE_ORIGINAL_CANDIDATE_GRID']:
     fixed=method.startswith('A_');cs=np.array([orig.breakpoint_severity]) if fixed else eligible
     cs=np.array([v for v in cs if admissible(xb,v)])
     if not primaryok or not len(cs):result.append(dict(**common,Analysis=method,Status='NOT ESTIMABLE',Missing_Reason='insufficient bins/50km support or fewer than4 supported binpositions on each side within original candidate bounds'));continue
     fits=[fit(xb,yb,ns,v) for v in cs];aics=np.array([z[1] for z in fits]);j=int(np.argmin(aics));star=cs[j];beta,aa=fits[j];delta=a0-aa;upper=bool(np.isclose(star,cand.max(),rtol=0,atol=1e-12));lower=bool(np.isclose(star,cand.min(),rtol=0,atol=1e-12));edge=upper or lower;subedge=not fixed and (star==eligible.min() or star==eligible.max());status='NO_SUPPORTED_BREAKPOINT' if delta<4 else 'EDGE_CONSTRAINED_NONLINEARITY_ONLY' if edge or subedge else 'SUPPORTED_CONDITIONAL_CANDIDATE';z=dict(**common,Analysis=method,S_star=float(star),Q_Full_Type=qrank(full,star),Q_Clean_Type=qrank(cleanS,star),Q_Full_Basin_Auxiliary=qrank(fullbasin,star),Pre_Slope=beta[1],Post_Slope=beta[1]+beta[2],Slope_Change=beta[2],Linear_AIC=a0,Segmented_AIC=aa,Delta_AIC=delta,Fixed_Knot_K3_Delta_AIC_Sensitivity=delta+2 if fixed else np.nan,Support_Level='STRONG' if delta>=10 else 'MODERATE' if delta>=4 else 'UNSUPPORTED',Segmented_Better_Than_Linear=delta>0,Upper_Edge_Hit=upper,Lower_Edge_Hit=lower,Any_Original_Edge_Hit=edge,Supported_Subrange_Edge_Hit=subedge,Status=status,S_star_Role='fixed input reference, not location identification' if fixed else 'selected diagnostic candidate, not ecologicalcriticalpoint')
     result.append(z)
     if not fixed:
      for st,ac in zip(cs,aics):curves.append(dict(Fit_ID=id0,Candidate_S=st,Q_Full_Type=qrank(full,st),AIC=ac,Delta_AIC_Linear=a0-ac,Delta_AIC_From_Best=ac-aa))
    if do_boot and primaryok:
     Y=np.asarray(by).T;NN=np.asarray(bnboot).T
     for b in range(NBOOT):
      keep=np.isfinite(Y[b])&(NN[b]>=500)
      if keep.sum()<12:continue
      xx=xb[keep];yy=Y[b,keep];nn=NN[b,keep];a00=fit(xx,yy,nn)[1]
      for method,cc in [('A_FIXED_ORIGINAL_BASIN_CANDIDATE',np.array([orig.breakpoint_severity])),('B_FREE_ORIGINAL_CANDIDATE_GRID',cand)]:
       cc=np.array([v for v in cc if admissible(xx,v)])
       if not len(cc):continue
       ff=[fit(xx,yy,nn,v) for v in cc];j=int(np.argmin([v[1] for v in ff]));be,ab=ff[j];st=cc[j];bootrows.append(dict(Fit_ID=id0,Event=h,Cohort=cohort,Vegetation=name,Response=response,Analysis=method,Replicate=b,S_star=st,Q_Full_Type=qrank(full,st),Delta_AIC=a00-ab,Pre_Slope=be[1],Post_Slope=be[1]+be[2],Slope_Change=be[2],Original_Edge_Hit=bool(st==cand.min() or st==cand.max()),Supported_Subrange_Edge=bool(st==cc.min() or st==cc.max()) if method.startswith('B_') else False))
    print(id0,'N',N,'bins',len(xb),'sec',round(time.time()-start,1),flush=True)
  del r,c,pix,g,eid,s,damage,legacy,vr,bn,block,clim,b100;gc.collect()
 out=pd.DataFrame(result);boots=pd.DataFrame(bootrows);boots.to_csv(O/'supplement/M8B/Stratified_Breakpoint_Bootstrap.csv',index=False)
 for col in ['S_star_CI_Lower','S_star_CI_Upper','Q_Full_Type_CI_Lower','Q_Full_Type_CI_Upper']:
  if col in out:out.loc[out.Analysis.str.startswith('A_'),col]=np.nan
 for (fid,method),z in boots.groupby(['Fit_ID','Analysis']):
  m=(out.Fit_ID==fid)&(out.Analysis==method);out.loc[m,'Bootstrap_Valid']=len(z);out.loc[m,'Bootstrap_Original_Edge_Fraction']=z.Original_Edge_Hit.mean();out.loc[m,'Bootstrap_Supported_Subrange_Edge_Fraction']=z.Supported_Subrange_Edge.mean();out.loc[m,'Bootstrap_Delta_AIC_GE4_Fraction']=(z.Delta_AIC>=4).mean()
  if len(z)>=cfg['Minimum_Valid_Bootstrap']:
   for field in (['Pre_Slope','Post_Slope','Slope_Change'] if method.startswith('A_') else ['S_star','Q_Full_Type','Pre_Slope','Post_Slope','Slope_Change']):out.loc[m,field+'_CI_Lower']=z[field].quantile(.025);out.loc[m,field+'_CI_Upper']=z[field].quantile(.975)
   out.loc[m,'CI_Status']='ESTIMABLE_CONDITIONAL_50km_FIXED_BIN_DESIGN'
  else:out.loc[m,'CI_Status']='NOT ESTIMABLE:valid replicas below280/350'
 hetero=[]
 for (h,response,cohort),z in boots[boots.Analysis.str.startswith('A_')].groupby(['Event','Response','Cohort']):
  for g1,g2 in [('Forest','Grassland_Shrubland'),('Forest','Cropland'),('Grassland_Shrubland','Cropland')]:
   a=z[z.Vegetation==g1].set_index('Replicate');b=z[z.Vegetation==g2].set_index('Replicate');common=a.index.intersection(b.index)
   for field in ['Pre_Slope','Post_Slope','Slope_Change']:
    values=a.loc[common,field]-b.loc[common,field];n=len(values);old1=out[(out.Event==h)&(out.Cohort==cohort)&(out.Vegetation==g1)&(out.Response==response)&out.Analysis.str.startswith('A_')];old2=out[(out.Event==h)&(out.Cohort==cohort)&(out.Vegetation==g2)&(out.Response==response)&out.Analysis.str.startswith('A_')];lo=values.quantile(.025) if n>=280 else np.nan;up=values.quantile(.975) if n>=280 else np.nan;hetero.append(dict(Event=h,Cohort=cohort,Response=response,Vegetation_First=g1,Vegetation_Second=g2,Metric=field,Difference=old1[field].iloc[0]-old2[field].iloc[0] if field in old1 and field in old2 else np.nan,CI_Lower=lo,CI_Upper=up,Bootstrap_Valid=n,CI_Excludes_0=bool(lo>0 or up<0) if np.isfinite(lo) else np.nan,Status='ESTIMABLE_CONDITIONAL' if n>=280 else 'NOT ESTIMABLE',Limit='paired same50km blockweights, fixed basin knot; conditional on bins/sample; exploratory unadjusted95%interval, no familywise multiplicity correction'))
 out.to_csv(O/'M8_Stratified_Breakpoints.csv',index=False,encoding='utf-8-sig');pd.DataFrame(binsout).to_csv(O/'supplement/M8B/Stratified_Response_Bins.csv',index=False);pd.DataFrame(curves).to_csv(O/'supplement/M8B/Stratified_Candidate_AIC.csv',index=False);pd.DataFrame(hetero).to_csv(O/'supplement/M8B/Paired_Type_Slope_Differences.csv',index=False);(O/'cache/M8B/BREAKPOINTS_READY.json').write_text(json.dumps(dict(N_Fits=len(out),N_Bootstrap_Rows=len(boots),N_New_Estimator_Model_Training=0,Seconds=time.time()-start),indent=2),encoding='utf8');print('BREAKPOINTS READY',len(out),flush=True)
if __name__=='__main__':main()
