from pathlib import Path
import json,time,gc,os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np
import pandas as pd
from pyproj import Transformer,CRS
from numba import njit
from model_utils import search
OUT=Path(r'.\EI_Reviewer1_Major1_Revision');CACHE=OUT/'cache'
NBOOT=350;SEED=20261109
PROJECT=CRS.from_proj4('+proj=aeqd +lat_0=35.5 +lon_0=107 +datum=WGS84 +units=m')
transform=Transformer.from_crs(4326,PROJECT,always_xy=True)

@njit(cache=True)
def med_boot(y,blocks,W):
    out=np.empty(len(W))
    for j in range(len(W)):
        total=0
        for i in range(len(y)):total+=W[j,blocks[i]]
        if total==0:out[j]=np.nan;continue
        k1=(total+1)//2;k2=total//2+1;cs=0;v1=0.;v2=0.;got=False
        for i in range(len(y)):
            cs+=W[j,blocks[i]]
            if not got and cs>=k1:v1=y[i];got=True
            if cs>=k2:v2=y[i];break
        out[j]=.5*(v1+v2)
    return out

def ecdf(sorted_values,s):return 100*np.searchsorted(sorted_values,s,side='right')/len(sorted_values)

def arrays(et):
    folder=CACHE/et;meta=json.loads((folder/'READY.json').read_text());n=meta['clean']
    names=['climate_cumulative_severity','Acomp_legacy_anomaly_days','Acomp_damage','valid_resistance_sample','x','y','row','col']
    arr={name:np.lib.format.open_memmap(folder/(name+'.npy'),mode='w+',dtype='float64',shape=(n,)) for name in names}
    k=0
    for p in sorted(folder.glob('clean_*.parquet')):
        d=pd.read_parquet(p,columns=names);nn=len(d)
        for name in names:arr[name][k:k+nn]=d[name]
        k+=nn
    for a in arr.values():a.flush()
    assert k==n
    return arr

def main():
    rep=[];curves=[];boots=[];mapping=[];splitrows=[];bindiag=[]
    for et in ['CDH','HEAT','HR']:
        folder=CACHE/et
        while not (folder/'READY.json').exists():time.sleep(5)
        a=arrays(et);s=a['climate_cumulative_severity'];n=len(s)
        meta=json.loads((folder/'READY.json').read_text())
        full=np.lib.format.open_memmap(folder/'all_severity_sorted.npy',mode='w+',dtype='float64',shape=(meta['rows'],))
        k=0
        for p in sorted(folder.glob('raw_severity_*.npz')):
            z=np.load(p);v=z['severity'];full[k:k+len(v)]=v;k+=len(v)
        full.sort();full.flush();clean=np.sort(np.asarray(s));np.save(folder/'clean_severity_sorted.npy',clean)
        x,y=transform.transform(a['x'],a['y']);blocks={};Ws={}
        for scale in [20,50]:
            labels=np.column_stack([np.floor(x/(scale*1000)),np.floor(y/(scale*1000))]).astype(np.int32)
            unique,bi=np.unique(labels,axis=0,return_inverse=True);blocks[scale]=bi.astype(np.int32);B=len(unique)
            rng=np.random.default_rng(SEED+scale);Ws[scale]=rng.multinomial(B,np.full(B,1/B),size=NBOOT).astype(np.int32)
            np.save(folder/f'blocks_{scale}.npy',bi.astype(np.int32))
            pd.DataFrame({'block':np.arange(B),'ix':unique[:,0],'iy':unique[:,1]}).to_csv(folder/f'block_registry_{scale}.csv',index=False)
            print(et,'blocks',scale,B,flush=True)
        # Fixed spatial holdout on 50-km blocks; selection independent of outcomes.
        B50=int(blocks[50].max()+1);rng=np.random.default_rng(SEED);discovery_blocks=rng.permutation(B50)[:B50//2]
        discovery=np.isin(blocks[50],discovery_blocks);np.save(folder/'discovery.npy',discovery)
        for nbins in [40,80,120]:
            edges=np.unique(np.quantile(clean,np.linspace(0,1,nbins+1)));edges[0]=-np.inf;edges[-1]=np.inf
            bi=np.clip(np.searchsorted(edges,s,side='right')-1,0,len(edges)-2)
            for response in ['Acomp_legacy','Acomp_damage','Damage_probability']:
                vals=a['Acomp_legacy_anomaly_days' if response=='Acomp_legacy' else 'Acomp_damage']
                valid=np.isfinite(vals)
                if response=='Acomp_damage':valid&=(vals>0)&(a['valid_resistance_sample']==1)
                xbin=[];ybin=[];count=[];booty={20:[],50:[]};bootn={20:[],50:[]}
                for b in range(len(edges)-1):
                    ix=np.flatnonzero((bi==b)&valid);nn=len(ix)
                    if nn<500:continue
                    xx=np.asarray(s[ix]);yy=np.asarray(vals[ix])
                    xc=float(np.mean(xx) if response=='Damage_probability' else np.median(xx))
                    yc=float(np.mean(yy>0) if response=='Damage_probability' else np.median(yy))
                    xbin.append(xc);ybin.append(yc);count.append(nn)
                    bindiag.append(dict(Event=et,Response=response,N_Bins_Requested=nbins,Bin=b,N=nn,S_Median=xc,Y=yc))
                    if nbins==80:
                        for scale in [20,50]:
                            block=blocks[scale][ix];W=Ws[scale];nnboot=W@np.bincount(block,minlength=W.shape[1])
                            if response=='Damage_probability':
                                sums=np.bincount(block,weights=(yy>0).astype(int),minlength=W.shape[1]);yb=(W@sums)/np.maximum(nnboot,1)
                            else:
                                order=np.argsort(yy);yb=med_boot(yy[order],block[order],W)
                            booty[scale].append(yb);bootn[scale].append(nnboot)
                xb=np.array(xbin);yb=np.array(ybin);nn=np.array(count)
                for mode in ['historical','full_supported']:
                    f=search(xb,yb,np.sqrt(nn),mode);qf=ecdf(full,f['c']);qc=ecdf(clean,f['c'])
                    rep.append(dict(Event=et,Response=response,Analysis=f'RAW_EXACT_MEDIANS_{nbins}_{mode}',Reproduction_Status='INDEPENDENT_RECONSTRUCTION',S_star=f['c'],Q_star_Full=qf,Q_star_Clean=qc,Delta_AIC=f['delta'],Search_Min=f['lo'],Search_Max=f['hi'],Search_Min_Q=ecdf(full,f['lo']),Search_Max_Q=ecdf(full,f['hi']),Distance_Lower_Q=qf-ecdf(full,f['lo']),Distance_Upper_Q=ecdf(full,f['hi'])-qf,n_bins=f['n'],Pre_Slope=f['beta'][1],Post_Slope=f['beta'][1]+f['beta'][2]))
                    for c,aa in zip(f['candidates'],f['aics']):curves.append(dict(Event=et,Response=response,Analysis=f'RAW_EXACT_MEDIANS_{nbins}_{mode}',Candidate_S=c,Candidate_Q_Full=ecdf(full,c),Candidate_Q_Clean=ecdf(clean,c),AIC=aa,Delta_AIC_Linear=f['linear_aic']-aa,Delta_AIC_From_Best=aa-f['aic']))
                if nbins==80:
                    for scale in [20,50]:
                        Y=np.array(booty[scale]).T;N=np.array(bootn[scale]).T
                        for j in range(NBOOT):
                            keep=np.isfinite(Y[j])&(N[j]>=500)
                            f=search(xb[keep],Y[j,keep],np.sqrt(N[j,keep]),'full_supported')
                            boots.append(dict(Event=et,Response=response,Block_km=scale,Replicate=j,S_star=f['c'],Q_star_Full=ecdf(full,f['c']),Q_star_Clean=ecdf(clean,f['c']),Delta_AIC=f['delta'],Pre_Slope=f['beta'][1],Post_Slope=f['beta'][1]+f['beta'][2],Search_Min=f['lo'],Search_Max=f['hi'],At_Search_Edge=bool(f['c']==f['lo'] or f['c']==f['hi']),N_Blocks=Ws[scale].shape[1],Bootstrap_Unit='complete spatial block; all pixel-events; exact multiplicity-weighted response medians; fixed full-cohort bin design'))
                    if response=='Acomp_legacy':
                        xd=[];yd=[];nd=[]
                        for b in range(len(edges)-1):
                            ix=np.flatnonzero((bi==b)&valid&discovery)
                            if len(ix)>=500:xd.append(np.median(s[ix]));yd.append(np.median(vals[ix]));nd.append(len(ix))
                        f=search(np.array(xd),np.array(yd),np.sqrt(nd),'full_supported')
                        splitrows.append(dict(Event=et,S_star=f['c'],Q_star_Full=ecdf(full,f['c']),Delta_AIC=f['delta'],N_Discovery_Blocks=len(discovery_blocks),N_Validation_Blocks=B50-len(discovery_blocks),Block_Overlap=0,Seed=SEED,Status='SPATIAL_HOLDOUT_DISCOVERY'))
                print(et,nbins,response,'full',f['c'],flush=True)
        h=pd.read_csv(OUT/'R1M1_Original_Breakpoint_Registry.csv');h=h[h.Event.eq(et)]
        for _,r in h.iterrows():mapping.append(dict(Event=et,Response=r.Response,S_star=r.S_star,Historical_Q=r.Q_star,Exact_Full_ECDF_Q=ecdf(full,r.S_star),Exact_Clean_ECDF_Q=ecdf(clean,r.S_star),Historical_CI_Low_Q_Full=ecdf(full,float(r.Breakpoint_CI.split(';')[0])),Historical_CI_High_Q_Full=ecdf(full,float(r.Breakpoint_CI.split(';')[1]))))
        del a,s,full,clean,blocks,Ws,x,y;gc.collect()
    old=pd.read_csv(OUT/'R1M1_Breakpoint_Replication.csv');pd.concat([old,pd.DataFrame(rep)],ignore_index=True).to_csv(OUT/'R1M1_Breakpoint_Replication.csv',index=False,encoding='utf-8-sig')
    old=pd.read_csv(OUT/'R1M1_Breakpoint_Candidate_AIC.csv');pd.concat([old,pd.DataFrame(curves)],ignore_index=True).to_csv(OUT/'R1M1_Breakpoint_Candidate_AIC.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(boots).to_csv(OUT/'R1M1_Spatial_Bootstrap_Breakpoints.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(mapping).to_csv(OUT/'audit/Percentile_Reference_Audit.csv',index=False)
    pd.DataFrame(splitrows).to_csv(OUT/'audit/Spatial_Holdout_Discovery.csv',index=False)
    pd.DataFrame(bindiag).to_csv(OUT/'audit/Independent_Exact_Bins.csv',index=False)
    print('INDEPENDENT BREAKPOINTS COMPLETE',flush=True)
if __name__=='__main__':main()
