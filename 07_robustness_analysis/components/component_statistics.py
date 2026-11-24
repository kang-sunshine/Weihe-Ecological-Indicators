# -*- coding: utf-8 -*-
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,time,gc
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from audit_core import O,WORK,VARS,PAIRS,sufficient,corr_from_sufficient,reranked_spearman_bootstrap,mixture_quantile_codes,prefix_midrank_lookup
from numba import set_num_threads
set_num_threads(4)

def float_buckets(v):
 bits=v.view(np.uint32);return (np.where((bits&np.uint32(0x80000000))!=0,~bits,bits^np.uint32(0x80000000))>>16).astype(np.int32)

PROBS=np.array([.05,.25,.5,.75,.95])
CORR=[];CONTRIB=[];LOO=[];QUANTCHECK=[]
def interval(v):
 good=np.isfinite(v);return (np.quantile(v[good],[.025,.975]) if good.sum()>=200 else [np.nan,np.nan]),int(good.sum())
def quantiles_sorted(x):
 t=(len(x)-1)*PROBS;i=np.floor(t).astype(np.int64);j=np.ceil(t).astype(np.int64);return np.asarray(x[i],np.float64)*(j-t)+np.asarray(x[j],np.float64)*(t-i)+np.where(i==j,np.asarray(x[i],np.float64),0)
def case_corr(name,level,s,W,Npixels,method,exact_boot=None):
 point=corr_from_sufficient(s.sum(axis=0));bootstrap=corr_from_sufficient(W@s) if exact_boot is None else exact_boot
 for i,(a,b) in enumerate(PAIRS):
  ci,nv=interval(bootstrap[:,i]);r=dict(Cohort=name,Level=level,Variable_1=VARS[a],Variable_2=VARS[b],Method=method,Estimate=point[i],CI_lower=ci[0],CI_upper=ci[1],N_Observations=int(s[:,0].sum()),N_Pixels=Npixels,N_50km_Blocks=int((s[:,0]>0).sum()),Bootstrap_N=250,Bootstrap_valid_n=nv,Seed=20261159,Unit='correlation',CI_Estimator='exact wholeblock,reranked within replicate' if exact_boot is not None else 'wholeblock sufficient-statistic Pearson' if method=='Pearson' else 'wholeblock fixed-pooled-rank conditional correlation; ranks not re-estimated',Status='ESTIMABLE' if nv>=200 else 'NOT ESTIMABLE')
  (CORR if a<3 else LOO).append(r)
def distribution(name,level,X,S,W,Npixels,qvals,absdiff):
 pt=S.sum(axis=0);bb=W@S;mean=pt[1:8]/pt[0];bootmean=bb[:,1:8]/bb[:,0,None];sd=np.sqrt(np.maximum(0,pt[8:15]/pt[0]-mean**2))
 for j in range(3,7):
  ci,nv=interval(bootmean[:,j]);LOO.append(dict(Cohort=name,Level=level,Variant=VARS[j],Statistic='Mean_Value',Estimate=mean[j],CI_lower=ci[0],CI_upper=ci[1],N_Observations=int(pt[0]),N_Pixels=Npixels,N_50km_Blocks=int((S[:,0]>0).sum()),Bootstrap_N=250,Bootstrap_valid_n=nv,Unit='dimensionless negative anomaly magnitude'))
  for q,v in zip(PROBS,qvals[j]):LOO.append(dict(Cohort=name,Level=level,Variant=VARS[j],Statistic='Q'+str(int(q*100)).zfill(2),Estimate=v,N_Observations=int(pt[0]),N_Pixels=Npixels,Status='DESCRIPTIVE_DISTRIBUTION; no quantileCI requested as formaldifference',Unit='dimensionless negative anomaly magnitude'))
 for j in range(3):
  dif=bootmean[:,4+j]-bootmean[:,3];ci,nv=interval(dif);value=mean[4+j]-mean[3];LOO.append(dict(Cohort=name,Level=level,Variant=VARS[4+j],Statistic='Paired_Mean_Difference_Minus_Acomp',Estimate=value,CI_lower=ci[0],CI_upper=ci[1],Estimate_Over_Acomp_SD=value/sd[3] if sd[3]>0 else np.nan,CI_lower_Over_Acomp_SD=ci[0]/sd[3] if sd[3]>0 else np.nan,CI_upper_Over_Acomp_SD=ci[1]/sd[3] if sd[3]>0 else np.nan,N_Observations=int(pt[0]),N_Pixels=Npixels,N_50km_Blocks=int((S[:,0]>0).sum()),Bootstrap_N=250,Bootstrap_valid_n=nv,Unit='dimensionless anomaly; standardized columns use fixedcohort AcompSD'))
  val=np.sum(absdiff[:,j])/pt[0];v=(W@absdiff[:,j])/bb[:,0];ci,nv=interval(v);LOO.append(dict(Cohort=name,Level=level,Variant=VARS[4+j],Statistic='Paired_Mean_Absolute_Difference',Estimate=val,CI_lower=ci[0],CI_upper=ci[1],N_Observations=int(pt[0]),N_Pixels=Npixels,N_50km_Blocks=int((S[:,0]>0).sum()),Bootstrap_N=250,Bootstrap_valid_n=nv))
def test_quantile_engine():
 blocks=[np.array([0,.25,1],np.float32),np.array([0,.1,.4,.5],np.float32),np.array([.8,1],np.float32)];pool=np.concatenate(blocks).view(np.uint32);start=np.array([0,3,7],np.int64);end=np.array([3,7,9],np.int64);W=np.array([[1,1,1],[0,2,1],[3,0,0]],np.int32);codes,fr=mixture_quantile_codes(pool,start,end,W,PROBS);v=codes.view(np.float32).astype(float);out=v[:,:,0]*(1-fr)+v[:,:,1]*fr;err=0.
 for i,w in enumerate(W):ref=np.quantile(np.concatenate([np.repeat(vv,w[j]) for j,vv in enumerate(blocks)]),PROBS);err=max(err,float(np.max(abs(ref-out[i]))))
 assert err<1e-7
 (O/'qc/Exact_Quantile_Engine_Verification.json').write_text(json.dumps(dict(MaxAbs_vs_explicit_np_quantile=err,No_Histogram_Approximation=True,Definition='exact linearquantiles ofinteger wholeblock-replicatedfloat32 empiricalmixture, rankorderbisection'),indent=2),encoding='utf8')
def contribution(name,z,W,Npixels):
 Ncommon=Npixels;Npixels=int((z['pixels'][:,4]>0).sum())
 folder=WORK/name;counts=z['Cstats'][:,0].astype(np.int64);start=np.r_[0,np.cumsum(counts)[:-1]];end=np.cumsum(counts);n=int(counts.sum());den=W@counts
 for j,label in enumerate(['CSync','CuWUE','CNIRv']):
  total=z['Cstats'][:,1+j];pt=total.sum()/n;vv=np.divide(W@total,den,out=np.full(250,np.nan),where=den>0);ci,nv=interval(vv);CONTRIB.append(dict(Cohort=name,Component=label,Statistic='Mean',Estimate=pt,CI_lower=ci[0],CI_upper=ci[1],N_Observations=n,N_Pixels=Npixels,N_50km_Blocks=int((counts>0).sum()),Bootstrap_N=250,Bootstrap_valid_n=nv,Unit='share of squarednegativeanomalies',Precision='float64calculation thenfloat32share; exactquantiles ofstoredshares'))
  poolpath=folder/f'quantile_pool_{j}.bin'
  with open(poolpath,'wb') as out:
   for b,nb in enumerate(counts):
    if nb==0:continue
    raw=np.memmap(folder/f'contrib_block_{b:02d}.bin',mode='r',dtype=np.float32,shape=(int(nb),3));v=np.sort(np.asarray(raw[:,j]).copy());v.view(np.uint32).tofile(out);del raw,v
  pool=np.memmap(poolpath,mode='r',dtype=np.uint32);codes,fr=mixture_quantile_codes(pool,start,end,np.vstack([np.ones(76,np.int32),W]),PROBS);a=codes.view(np.float32).astype(float);values=a[:,:,0]*(1-fr)+a[:,:,1]*fr
  for k,q in enumerate(PROBS):
   ci,nv=interval(values[1:,k]);CONTRIB.append(dict(Cohort=name,Component=label,Statistic='Q'+str(int(q*100)).zfill(2),Estimate=values[0,k],CI_lower=ci[0],CI_upper=ci[1],N_Observations=n,N_Pixels=Npixels,N_50km_Blocks=int((counts>0).sum()),Bootstrap_N=250,Bootstrap_valid_n=nv,Unit='share of squarednegativeanomalies',Quantile_Estimator='exactfloat32empirical mixturelinearquantile, not histogram/interpolation overmissingdata'))
  del pool;gc.collect();poolpath.unlink()
 alln=z['stats'][:,0];zeros=z['zero'];v=(W@zeros)/(W@alln);ci,nv=interval(v);CONTRIB.append(dict(Cohort=name,Component='Acomp_ZERO',Statistic='Fraction_All_Common_Records',Estimate=zeros.sum()/alln.sum(),CI_lower=ci[0],CI_upper=ci[1],N_Observations=int(alln.sum()),N_Zero=int(zeros.sum()),N_Positive_For_Contribution=n,N_Pixels=Ncommon,N_50km_Blocks=int((alln>0).sum()),Bootstrap_N=250,Bootstrap_valid_n=nv,Interpretation='contributions undefined atcompositezero; not set0'))
def saveprogress():
 for data,name in [(CORR,'A1_Component_Correlation.csv'),(CONTRIB,'A2_Component_Contribution.csv'),(LOO,'A3_LeaveOneComponentOut.csv')]:pd.DataFrame(data).to_csv(O/'outputs'/name,index=False,encoding='utf-8-sig')
def main():
 assert (O/'qc/EXTRACTION_READY.json').exists();test_quantile_engine();W=np.load(O/'qc/Spatial_Bootstrap_250_Weights.npy');cfg=json.loads((O/'qc/configuration.json').read_text())
 from audit_core import R
 grid=np.load(R/'EI_Reviewer1_Major5_Revision/cache/CDH_pixel_blocks_50.npy').ravel();cases=[h+'_'+typ for h in ['CDH','HEAT','HR'] for typ in ['PEAK','EPISODE']]+['GLOBAL_COMMON'];overall=time.time()
 completed={name for name in cases if (WORK/name/'STATS_DONE.json').exists()}
 if completed:
  for target,filename in [(CORR,'A1_Component_Correlation.csv'),(CONTRIB,'A2_Component_Contribution.csv'),(LOO,'A3_LeaveOneComponentOut.csv')]:
   old=pd.read_csv(O/'outputs'/filename);target.extend(old[old.Cohort.isin(completed)].to_dict('records'))
 for name in cases:
  if (WORK/name/'STATS_DONE.json').exists():
   print('SKIP completed',name,flush=True);continue
  folder=WORK/name;z=np.load(folder/'aggregates.npz');N=int(z['N']);X=np.memmap(folder/'values.bin',mode='r',dtype=np.float32,shape=(N,7));bl=np.memmap(folder/'blocks.bin',mode='r',dtype=np.uint8,shape=(N,));pix=z['pixels'];pvalid=pix[:,0]>0;npix=int(pvalid.sum());pX=pix[pvalid,1:]/pix[pvalid,:1];pblocks=grid[pvalid];S=z['stats'];qvals=[];rankpaths=[]
  case_corr(name,'OBSERVATION_RECORD',S,W,npix,'Pearson')
  # Auxiliarywithin-pixel demeaning, exactPearson fromper-pixel sums.
  demean=S.copy();demean[:,1:8]=0
  for b in np.unique(pblocks):
   m=pblocks==b;nn=pix[pvalid,0][m];v=pX[m];demean[b,8:15]-=np.sum(nn[:,None]*v*v,axis=0)
   for k,(a,c) in enumerate(PAIRS):demean[b,15+k]-=np.sum(nn*v[:,a]*v[:,c])
  case_corr(name,'OBSERVATION_PIXEL_DEMEANED',demean,W,npix,'Pearson')
  # Pixel means giveone observation perunique pixel, not event repetition.
  ps=sufficient(pX,pblocks,76);case_corr(name,'PIXEL_MEAN',ps,W,npix,'Pearson')
  orders=np.column_stack([np.argsort(pX[:,j],kind='stable') for j in range(7)]);pb=reranked_spearman_bootstrap(pX,orders,pblocks,np.vstack([np.ones(76,np.int32),W]));pxr=np.column_stack([pd.Series(pX[:,j]).rank(method='average').to_numpy()/len(pX) for j in range(7)]);prs=sufficient(pxr,pblocks,76);case_corr(name,'PIXEL_MEAN',prs,W,npix,'Spearman',pb[1:])
  assert np.max(abs(pb[0]-corr_from_sufficient(prs.sum(axis=0))))<1e-10
  absdiff=np.stack([np.bincount(pblocks,weights=abs(pX[:,4+j]-pX[:,3]),minlength=76) for j in range(3)],axis=1);pquant=[np.quantile(pX[:,j],PROBS) for j in range(7)];distribution(name,'PIXEL_MEAN',pX,ps,W,npix,pquant,absdiff)
  del orders,pb,pxr;gc.collect()
  # Exact pooled Spearman ranks through single-variable, on-disk sorting.
  for j,vname in enumerate(VARS):
   sortedpath=folder/f'sorted_{j}.bin';rankpath=folder/f'rank_{j}.bin';rankpaths.append(rankpath)
   if sortedpath.exists() and sortedpath.stat().st_size==N*4:
    # A completed sorted column may be resumed after interruption of its rank lookup.
    print('REUSE completed sorted column',name,j,flush=True)
   else:
    with open(sortedpath,'wb') as f:
     for start in range(0,N,1_000_000):np.asarray(X[start:min(N,start+1_000_000),j]).copy().tofile(f)
    ss=np.memmap(sortedpath,mode='r+',dtype=np.float32,shape=(N,));ss.sort(kind='quicksort');ss.flush();del ss
   ss=np.memmap(sortedpath,mode='r',dtype=np.float32,shape=(N,));qvals.append(quantiles_sorted(ss))
   # Keep sorted keys in RAM and stream the interleaved source sequentially.
   # Random binary searches against an on-disk memory map cause HDD paging storms.
   sorted_values=np.array(ss,copy=True);del ss;gc.collect();counts=np.zeros(65536,np.int64)
   for st in range(0,N,1_000_000):
    vv=sorted_values[st:min(N,st+1_000_000)];vv[vv==0]=0 # normalize signed zero; equal values must have equal ranks
    assert np.all(vv[:-1]<=vv[1:]) and (st==0 or sorted_values[st-1]<=vv[0]),'Incomplete/unsorted temporarycolumn: do not resume invalidrankinput'
    counts+=np.bincount(float_buckets(vv),minlength=65536)
   prefix=np.r_[0,np.cumsum(counts)];print('PREFIX ready',name,vname,flush=True)
   with open(rankpath,'wb') as f:
    for start in range(0,N,500_000):
     nv=min(N,start+500_000)-start
     chunk=np.fromfile(folder/'values.bin',dtype=np.float32,count=nv*7,offset=start*7*4).reshape(nv,7)
     vals=chunk[:,j].copy();vals[vals==0]=0
     rr=prefix_midrank_lookup(sorted_values,vals,float_buckets(vals),prefix);rr.tofile(f)
   del sorted_values,chunk,vals,rr;gc.collect();sortedpath.unlink();print('EXACT RANKS',name,vname,'N',N,'elapsed',round(time.time()-overall,1),flush=True)
  rs=np.zeros((76,21));ranks=[np.memmap(f,mode='r',dtype=np.float64,shape=(N,)) for f in rankpaths]
  for start in range(0,N,500_000):
   end=min(N,start+500_000);v=np.column_stack([r[start:end] for r in ranks]);rs+=sufficient(v,bl[start:end],76)
  exact=None
  if name.endswith('_PEAK'):
   x=np.asarray(X).copy();orders=np.column_stack([np.argsort(x[:,j],kind='stable') for j in range(7)]);exact=reranked_spearman_bootstrap(x,orders,np.asarray(bl),W);del x,orders;gc.collect()
  case_corr(name,'OBSERVATION_RECORD',rs,W,npix,'Spearman',exact)
  distribution(name,'OBSERVATION_RECORD',None,S,W,npix,qvals,z['absdiff']);contribution(name,z,W,npix)
  # Release large temporaryrankfiles after exactpoint andintervals havebeen saved.
  del ranks;gc.collect()
  for f in rankpaths:f.unlink()
  saveprogress();(folder/'STATS_DONE.json').write_text(json.dumps(dict(Cohort=name,N=N,Exact_Pooled_Midrank=True,Record_Spearman_Bootstrap='reranked' if name.endswith('_PEAK') else 'fixedpooledranksconditional',No_Record_Sampling=True,Elapsed=time.time()-overall),indent=2),encoding='utf8')
  print('A CASE COMPLETE',name,'elapsed',round(time.time()-overall,1),flush=True);del X,bl,z,pix,pX,ps,prs;gc.collect()
 (O/'qc/A_READY.json').write_text(json.dumps(dict(N_Cases=len(cases),Exact_Pooled_Midrank=True,Full_Global_Cohort_Not_Sampled=True,RecoveryUncertainty='B exactpairedwholeblock',Record_Spearman_Limit='GLOBAL/EPISODE CI conditional onfixedpooledranks, notfullyreranked nonparametricCI; exactreranking forpixelmean andoriginaleventpeak'),indent=2),encoding='utf8')
if __name__=='__main__':main()
