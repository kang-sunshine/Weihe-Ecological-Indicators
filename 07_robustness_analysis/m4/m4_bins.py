# -*- coding: utf-8 -*-
from pathlib import Path
import ast,json
import numpy as np
import pandas as pd
from pyproj import Transformer
O=Path(r'.\EI_Reviewer1_Major4_Revision');M1=Path(r'.\EI_Reviewer1_Major1_Revision');D=Path(r'.\Weihe database V3');F=M1/'cache/CDH'
def fit(x,y,w,c=None):
 X=np.column_stack([np.ones(len(x)),x]+([] if c is None else [np.maximum(0,x-c)]));b=np.linalg.lstsq(X*np.sqrt(w)[:,None],y*np.sqrt(w),rcond=None)[0];rss=np.sum(w*(y-X@b)**2);return b,len(x)*np.log(max(rss,1e-300)/len(x))+2*(2 if c is None else 4)
def main():
 src=(M1/'audit/Step_L5_4A_Severity_Response_Thresholds_FAST.py').read_text(encoding='utf-8-sig');tree=ast.parse(src);node=next(q for q in tree.body if isinstance(q,ast.FunctionDef) and q.name=='bounded_sample');env={'np':np,'RNG':np.random.default_rng(20260908)};exec(compile(ast.Module(body=[node],type_ignores=[]),'<historical bounded_sample only>','exec'),env)
 # Reuse original cached order and original 1M-row chunk boundaries; no model rerun.
 files=sorted(F.glob('raw_severity_*.npz'));sample=np.empty(0)
 for j in range(0,len(files),10):
  zs=[np.load(p) for p in files[j:j+10]];s=np.concatenate([z['severity'] for z in zs]);clean=np.concatenate([z['clean'] for z in zs]);sample=env['bounded_sample'](sample,s[clean&np.isfinite(s)],500000)
 edges=np.unique(np.quantile(sample,np.linspace(0,1,81)));edges[0]=-np.inf;edges[-1]=np.inf;np.save(O/'paired/CDH_Original_Historical_Bin_Edges.npy',edges)
 s=np.load(F/'climate_cumulative_severity.npy',mmap_mode='r');r=np.load(F/'row.npy',mmap_mode='r').astype(int);c=np.load(F/'col.npy',mmap_mode='r').astype(int);lon=np.load(F/'x.npy',mmap_mode='r');lat=np.load(F/'y.npy',mmap_mode='r');legacy=np.load(F/'Acomp_legacy_anomaly_days.npy',mmap_mode='r');damage=np.load(F/'Acomp_damage.npy',mmap_mode='r');vr=np.load(F/'valid_resistance_sample.npy',mmap_mode='r');assert len(s)==1755878
 cfg=json.loads((O/'code/configuration.json').read_text());x,y=Transformer.from_crs('EPSG:4326',cfg['Projection'],always_xy=True).transform(lon,lat);b50=np.floor(x/50000).astype(np.int64)*10000+np.floor(y/50000).astype(np.int64);b100=np.floor(x/100000).astype(np.int64)*10000+np.floor(y/100000).astype(np.int64);parent=np.load(O/'cache/Reconstructed_ERA5_0p1_Cell_ID.npy',mmap_mode='r')[r,c];bins=np.clip(np.searchsorted(edges,s,side='right')-1,0,len(edges)-2);saved=pd.read_csv(D/'L5_EventEcoFunction/L5_4ThresholdAnalysis/L5_4A_Severity_Response_Binned.csv') if False else pd.read_csv(D/'L5_EventEcoFunction/L5_4_ThresholdAnalysis/L5_4A_Severity_Response_Binned.csv');rows=[];fits=[];curves=[]
 for response,val,valid in [('Acomp_legacy',legacy,np.isfinite(legacy)),('Acomp_damage',damage,np.isfinite(damage)&(damage>0)&(vr==1))]:
  hist=saved[(saved.event_type=='CDH')&(saved.response==response)].sort_values('severity_median').copy();countmap={}
  for bn in sorted(np.unique(bins[valid])):
   ix=np.flatnonzero(valid&(bins==bn));n=len(ix)
   if n<500:continue
   cells,cellcounts=np.unique(parent[ix],return_counts=True);bb,nn=np.unique(b50[ix],return_counts=True);n100=len(np.unique(b100[ix]));hhi=float(np.sum((nn/n)**2));countmap[int(bn)]=n
   rows.append(dict(Event='CDH',Response=response,Historical_Bin_ID=int(bn),Severity_Lower=float(edges[bn]),Severity_Upper=float(edges[bn+1]),N_Events=n,N_Ecological_Pixels=len(np.unique(r[ix]*703+c[ix])),N_Reconstructed_0p1_Climate_Cells=len(cells),N_Native_Independent_Climate_Grid=np.nan,N_50km_Blocks=len(bb),N_100km_Blocks=n100,Events_Per_Climate_Cell=n/len(cells),Events_Per_50km_Block=n/len(bb),Max_50km_Event_Share=float(nn.max()/n),HHI_50km_Event_Concentration=hhi,Inverse_HHI_50km_Effective_Support=1/hhi,Original_RSS_Weight=np.sqrt(n),Design_Matrix_Row_Multiplier=n**.25,Original_Source=str(D/'L5_EventEcoFunction/L5_4_ThresholdAnalysis/L5_4A_Severity_Response_Binned.csv'),Climate_Count_Definition='empirically reconstructed regular0.1deg support; not independent Gaussian model nodes',Missing_Reason='native Gaussian identities/statistical independence unavailable; tail bounds +/-inf are defined open endpoints, not observed values'))
  assert all(countmap[int(q.bin_id)]==q.n for q in hist.itertuples()),response
  xb=hist.severity_median.to_numpy();yb=hist.response_median.to_numpy();origw=np.sqrt(hist.n.to_numpy());n=len(hist);lo=max(int(np.floor(.1*n)),4);hi=min(int(np.ceil(.9*n)),n-4);cand=np.unique(xb[lo:hi]);sub=pd.DataFrame(rows);sub=sub[sub.Response==response].set_index('Historical_Bin_ID').loc[hist.bin_id]
  for name,w in [('ORIGINAL_SQRT_EVENT_N',origw),('SQRT_50KM_BLOCK_SUPPORT',np.sqrt(sub.N_50km_Blocks.to_numpy())),('SQRT_100KM_BLOCK_SUPPORT',np.sqrt(sub.N_100km_Blocks.to_numpy()))]:
   if name!='ORIGINAL_SQRT_EVENT_N':w=w*origw.mean()/w.mean()
   b0,a0=fit(xb,yb,w);ff=[fit(xb,yb,w,v) for v in cand];aics=np.array([v[1] for v in ff]);best=int(np.argmin(aics));star=cand[best];full=np.load(F/'all_severity_sorted.npy',mmap_mode='r');q=100*np.searchsorted(full,star,side='right')/len(full)
   fits.append(dict(Event='CDH',Response=response,Weight_Scheme=name,N_Bins=n,S_star=star,Q_Full=q,Search_Min=cand.min(),Search_Max=cand.max(),Upper_Edge_Hit=bool(star==cand.max()),Any_Edge_Hit=bool(star==cand.max() or star==cand.min()),Delta_AIC_Linear_Minus_Segmented=a0-aics[best],Linear_AIC=a0,Segmented_AIC=aics[best],Pre_Slope=ff[best][0][1],Post_Slope=ff[best][0][1]+ff[best][0][2],Spatial_CI_Status='NO_NEW_BOOTSTRAP; existing M1 original-scheme full-supported-search intervals reported separately',Limit='saved historical x/y medians fixed; only RSS weights changed; cross-weight raw AIC is not a common likelihood comparison'))
   for st,aa in zip(cand,aics):curves.append(dict(Event='CDH',Response=response,Weight_Scheme=name,Candidate_S=st,AIC=aa,Delta_AIC_From_Best=aa-aics[best],Delta_AIC_Linear=a0-aa))
 pd.DataFrame(rows).to_csv(O/'M4_Binned_Spatial_Dependence_Audit.csv',index=False,encoding='utf-8-sig');pd.DataFrame(fits).to_csv(O/'supplement/Reweighted_Breakpoint_Fits.csv',index=False,encoding='utf-8-sig');pd.DataFrame(curves).to_csv(O/'supplement/Reweighted_Candidate_AIC.csv',index=False)
 hist=pd.read_csv(D/'L5_EventEcoFunction/L5_4_ThresholdAnalysis/L5_4A_Severity_Response_Thresholds.csv');ff=pd.DataFrame(fits)
 for response in ['Acomp_legacy','Acomp_damage']:
  a=ff[(ff.Response==response)&(ff.Weight_Scheme=='ORIGINAL_SQRT_EVENT_N')].iloc[0];b=hist[(hist.event_type=='CDH')&(hist.response==response)].iloc[0];assert abs(a.S_star-b.breakpoint_severity)<1e-12 and abs(a.Delta_AIC_Linear_Minus_Segmented-b.delta_AIC_linear_minus_segmented)<1e-8
 boot=pd.read_csv(M1/'R1M1_Spatial_Bootstrap_Breakpoints.csv');out=[]
 for (event,response,scale),z in boot.groupby(['Event','Response','Block_km']):
  out.append(dict(Event=event,Response=response,Block_km=scale,N_Replicates=len(z),S_CI025=z.S_star.quantile(.025),S_CI50=z.S_star.median(),S_CI975=z.S_star.quantile(.975),Q_Full_CI025=z.Q_star_Full.quantile(.025),Q_Full_CI975=z.Q_star_Full.quantile(.975),Candidate_Edge_Hit_Fraction=z.At_Search_Edge.mean(),Median_Delta_AIC=z.Delta_AIC.median(),Median_N_Blocks=z.N_Blocks.median(),Search_Definition='M1 full-supported candidate range, not historical search; do not assign these CIs to reweighted fits',Source=str(M1/'R1M1_Spatial_Bootstrap_Breakpoints.csv')))
 pd.DataFrame(out).to_csv(O/'supplement/M1_Existing_Spatial_Uncertainty.csv',index=False,encoding='utf-8-sig');print('BINS COMPLETE',len(rows),'rows',ff.to_string(index=False),flush=True)
if __name__=='__main__':main()
