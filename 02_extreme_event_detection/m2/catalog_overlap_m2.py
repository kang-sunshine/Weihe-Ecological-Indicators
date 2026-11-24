# -*- coding: utf-8 -*-
from pathlib import Path
import json,time,gc
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from numba import njit
OUT=Path(r'.\EI_Reviewer1_Major2_Revision');DB=Path(r'.\Weihe database V3');M1=Path(r'.\EI_Reviewer1_Major1_Revision')
@njit(cache=True)
def fill_intervals(grid,r,c,o,t):
    dup=0
    for i in range(len(r)):
        for k in range(o[i],t[i]+1):
            if grid[r[i],k,c[i]]:dup+=1
            grid[r[i],k,c[i]]=True
    return dup
@njit(cache=True)
def query_intervals(r,c,o,t,native,spans):
    z=np.zeros((len(r),2),np.int16)
    for i in range(len(r)):
        for k in range(o[i],t[i]+1):
            z[i,0]+=int(native[r[i],k,c[i]]);z[i,1]+=int(spans[r[i],k,c[i]])
    return z

def main():
    while not (OUT/'code/configuration.json').exists():time.sleep(2)
    cfg=json.loads((OUT/'code/configuration.json').read_text());shape=tuple(cfg['shape']);counts={};start=time.time()
    for et in ['CDH','HEAT','HR']:
        directory=OUT/'catalogues'/et;directory.mkdir(parents=True,exist_ok=True)
        span=np.lib.format.open_memmap(OUT/'cache'/f'span_{et}.npy',mode='w+',dtype='bool',shape=shape);span[:]=False
        n=0;duplicate=0;span_total=0;active=0;gaps=0
        p=DB/'L5_EventEcoFunction/L5_2_EventLifecycle'/f'L5_2_{et}_Independent_Event_Catalog.csv.gz'
        cols=['event_id','row','col','onset_index','termination_index','span_phases','active_phases_original','gap_filled_phases']
        for j,d in enumerate(pd.read_csv(p,usecols=cols,chunksize=100000)):
            r=d.row.to_numpy(np.int32);c=d.col.to_numpy(np.int32);o=d.onset_index.to_numpy(np.int32);t=d.termination_index.to_numpy(np.int32)
            ids=d.event_id.str.rsplit('_',n=1).str[-1].astype(np.int32).to_numpy();assert np.array_equal(ids,np.arange(n+1,n+len(d)+1))
            assert np.array_equal(t-o+1,d.span_phases) and np.array_equal(d.active_phases_original+d.gap_filled_phases,d.span_phases)
            duplicate+=int(fill_intervals(span,r,c,o,t));n+=len(d);span_total+=int(d.span_phases.sum());active+=int(d.active_phases_original.sum());gaps+=int(d.gap_filled_phases.sum())
            slim=pd.DataFrame(dict(Event_Num=ids,row=r.astype('int16'),col=c.astype('int16'),onset=o.astype('int16'),termination=t.astype('int16'),Span_Phases=d.span_phases.astype('int16'),Active_Phases=d.active_phases_original.astype('int16'),Gap_Phases=d.gap_filled_phases.astype('int16')))
            pq.write_table(pa.Table.from_pandas(slim,preserve_index=False),directory/f'part_{j:03d}.parquet',compression='zstd')
            if j%30==0:print('catalogue',et,j,n,'sec',round(time.time()-start,1),flush=True)
        assert duplicate==0
        expected=json.loads((M1/'cache'/et/'READY.json').read_text())['rows'];assert n==expected
        span.flush();del span
        counts[et]=dict(N_Events=n,Span_Phases=span_total,Active_Phases=active,Filled_Gap_Phases=gaps,Duplicate_Own_Interval_Phases=duplicate,Source=str(p))
    (OUT/'cache/CATALOGUES_READY.json').write_text(json.dumps(counts,indent=2),encoding='utf8')
    while not (OUT/'cache/NATIVE_READY.json').exists():time.sleep(3)
    native={et:np.load(OUT/'cache'/f'native_{et}.npy',mmap_mode='r') for et in counts};spans={et:np.load(OUT/'cache'/f'span_{et}.npy',mmap_mode='r') for et in counts};unknown=np.load(OUT/'cache/heat_unknown_soil.npy',mmap_mode='r')
    phase=[];pairs=[];ecount=[];classcounts={};cross={};ny=shape[0]
    for scope,maps in [('NATIVE_ACTIVE_PHASE',native),('MERGED_INTERVAL_PHASE_OCCUPANCY',spans)]:
        total={et:0 for et in counts};inter={(a,b):0 for a,b in [('CDH','HEAT'),('CDH','HR'),('HEAT','HR')]};viol=0
        for y0 in range(0,ny,8):
            arr={e:np.asarray(m[y0:y0+8]) for e,m in maps.items()}
            for e in counts:total[e]+=int(arr[e].sum())
            for a,b in inter:inter[a,b]+=int((arr[a]&arr[b]).sum())
            viol+=int((arr['CDH']&~arr['HEAT']).sum())
        assert viol==0
        for (a,b),v in inter.items():phase.append(dict(Scope=scope,Event_A=a,Event_B=b,N_A_Phases=total[a],N_B_Phases=total[b],N_Intersection=v,Fraction_A_Overlapped=v/total[a],Fraction_B_Overlapped=v/total[b],Jaccard=v/(total[a]+total[b]-v),Unit='same pixel, same recorded phase; merged scope includes filled gaps',Source='existing L5.0 masks' if scope.startswith('NATIVE') else 'existing L5.2 interval occupancy'))
        if scope.startswith('MERGED'):
            for et in counts:assert total[et]==counts[et]['Span_Phases']
    for anchor in ['CDH','HEAT','HR']:
        other=[e for e in counts if e!=anchor];sums={e:np.zeros(4,np.int64) for e in other}
        if anchor=='HEAT':
            size=counts[anchor]['N_Events']+1
            flag=np.lib.format.open_memmap(OUT/'cache/HEAT_Membership_By_EventNum.npy',mode='w+',dtype='uint8',shape=(size,));flag[:]=0
            outcat=OUT/'sensitivity/HEAT_Event_Classification';outcat.mkdir(exist_ok=True)
        for j,p in enumerate(sorted((OUT/'catalogues'/anchor).glob('*.parquet'))):
            d=pd.read_parquet(p);r=d.row.to_numpy(np.int32);c=d.col.to_numpy(np.int32);o=d.onset.to_numpy(np.int32);t=d.termination.to_numpy(np.int32)
            qq={e:query_intervals(r,c,o,t,native[e],spans[e]) for e in other}
            for e,z in qq.items():sums[e]+=np.array([int((z[:,0]>0).sum()),int((z[:,1]>0).sum()),int(z[:,0].sum()),int(z[:,1].sum())])
            if anchor=='HEAT':
                z=qq['CDH'];uk=query_intervals(r,c,o,t,unknown,unknown)[:,0]
                nonnative=z[:,0]==0;nonspan=z[:,1]==0
                values=(nonnative.astype(np.uint8))|(nonspan.astype(np.uint8)*2)|((uk==0).astype(np.uint8)*4)
                flag[d.Event_Num.to_numpy(int)]=values
                slim=d.copy();slim['HEAT_Only_No_CDH_Native_Within_Interval']=nonnative;slim['HEAT_Only_No_CDH_Merged_Interval']=nonspan;slim['Unknown_Soil_Assessment_Phases']=uk;slim['CDH_Native_Overlap_Phases']=z[:,0];slim['CDH_Merged_Overlap_Phases']=z[:,1];slim['HR_Native_Overlap_Phases']=qq['HR'][:,0];slim['HR_Merged_Overlap_Phases']=qq['HR'][:,1]
                pq.write_table(pa.Table.from_pandas(slim,preserve_index=False),outcat/p.name,compression='zstd')
        for e,z in sums.items():
            ecount.append(dict(Anchor_Event=anchor,Other_Event=e,N_Anchor_Events=counts[anchor]['N_Events'],N_With_Other_Native_Phase=int(z[0]),N_With_Other_Merged_Interval=int(z[1]),Fraction_Anchor_Overlap_Other_Native=z[0]/counts[anchor]['N_Events'],Fraction_Anchor_Overlap_Other_Merged=z[1]/counts[anchor]['N_Events'],Intersection_Phases_With_Other_Native=int(z[2]),Intersection_Phases_With_Other_Merged=int(z[3]),Overlap_Rule='at least one common phase at same pixel inside original inclusive onset..termination; intervals include filled gaps',Pair_Count_Status='not event-pair count; directional anchor-event fraction'))
        if anchor=='HEAT':
            flag.flush();vals=np.asarray(flag[1:]);classcounts=dict(N_HEAT_Events=len(vals),N_No_CDH_Native=int(((vals&1)>0).sum()),N_No_CDH_Merged=int(((vals&2)>0).sum()),N_No_CDH_Merged_And_Known_Soil=int((((vals&2)>0)&((vals&4)>0)).sum()),N_NativeVsMerged_Class_Disagreement=int((((vals&1)>0)!=((vals&2)>0)).sum()),N_With_Unknown_Soil=int(((vals&4)==0).sum()))
    pd.DataFrame(phase).to_csv(OUT/'M2_Phase_Overlap.csv',index=False,encoding='utf-8-sig');pd.DataFrame(ecount).to_csv(OUT/'M2_Merged_Event_Overlap.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([dict(Event=e,**v) for e,v in counts.items()]).to_csv(OUT/'audit/Catalogue_Structure_Counts.csv',index=False)
    (OUT/'cache/OVERLAP_READY.json').write_text(json.dumps(dict(classification=classcounts,catalogues=counts),indent=2),encoding='utf8')
    print('OVERLAP COMPLETE',classcounts,flush=True)
if __name__=='__main__':main()
