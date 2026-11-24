# -*- coding: utf-8 -*-
from pathlib import Path
import os,json,time,gc
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from numba import njit
OUT=Path(r'.\EI_Reviewer1_Major5_Revision');M1=Path(r'.\EI_Reviewer1_Major1_Revision')
VARIANTS=['PEAK_ONLY_ACTIVE_NEXT_EXCLUSIVE','PEAK_CATALOG_NEXT_EXCLUSIVE','PRIMARY_POSTPEAK_CATALOG_INCLUSIVE','FIRST_MISSING_GAP_SENSITIVITY','NO_PEAK_RULE_CATALOG_SENSITIVITY']

@njit(cache=True)
def extract(cols,term,peak,baseline,active_next,td,a,onsets):
    n=len(cols);out=np.full((n,5,10),-1,np.int16);meta=np.zeros((n,6),np.int16);nt=len(td)
    for i in range(n):
        col=cols[i];t0=term[i];pk=peak[i];b=baseline[i];endday=td[-1]-td[t0];nxt=-1
        for k in range(t0+1,nt):
            day=td[k]-td[t0]
            if day>120:break
            if onsets[k,col]:nxt=day;break
        meta[i,0]=nxt
        meta[i,1]=int(active_next[i]>=0 and (nxt<0 or active_next[i]<nxt))
        complete=endday>=120 and nxt<0
        scheduled=0;finite=0
        for k in range(t0,nt):
            day=td[k]-td[t0]
            if day>120:break
            scheduled+=1
            if np.isfinite(a[k,col]):finite+=1
            else:complete=False
        meta[i,2]=int(complete);meta[i,3]=scheduled;meta[i,4]=finite
        meta[i,5]=int(nxt>=0 and td[pk]-td[t0]>=nxt)
        for v in range(5):
            nn=active_next[i] if v==0 else nxt
            cday=min(120,endday)
            if nn>=0:cday=min(cday,int(nn))
            reason=1 if nn>=0 and nn<=120 and nn<=endday else (2 if endday>=120 else 3)
            exclusive=v<=1
            rec=-1;confirm=-1;first=-1;consec=0;lastvalid=-1;lastscheduled=-1;missing=0;pending=0;gap_stop=False
            for k in range(t0,nt):
                day=td[k]-td[t0]
                if day>120:break
                if nn>=0 and day>=nn:break
                if exclusive and day>=cday:break
                if not exclusive and day>cday:break
                lastscheduled=day;value=a[k,col]
                if not np.isfinite(value):
                    missing+=1;consec=0;first=-1
                    if v==3:
                        reason=4;gap_stop=True;break
                    continue
                lastvalid=day
                if v!=4 and k<=pk:continue
                if value<=b:
                    if consec==0:first=day
                    consec+=1;pending=1
                    if consec>=2:rec=first;confirm=day;pending=0;break
                else:consec=0;first=-1;pending=0
            dur=rec if rec>=0 else cday
            if rec>=0:reason=0
            elif gap_stop or lastvalid<lastscheduled:
                reason=4;dur=max(0,lastvalid)
            valid=lastvalid>=0
            out[i,v,0]=dur;out[i,v,1]=reason;out[i,v,2]=int(valid);out[i,v,3]=rec;out[i,v,4]=confirm;out[i,v,5]=pending;out[i,v,6]=missing;out[i,v,7]=lastvalid;out[i,v,8]=lastscheduled;out[i,v,9]=int(rec==0)
    return out,meta

def add_counts(target,block,group,duration,reason,valid):
    ix=(((block[valid]*2+group[valid])*5+reason[valid])*121+duration[valid]).astype(int)
    target+=np.bincount(ix,minlength=target.size).reshape(target.shape)

def main():
    while not (OUT/'cache/ONSETS_READY.json').exists():time.sleep(3)
    dates=np.load(OUT/'cache/dates.npy');td=np.load(OUT/'cache/time_days.npy');a=np.load(M1/'cache/Acomp_by_row.npy',mmap_mode='r');on=np.load(OUT/'cache/Catalogue_Onsets_Union.npy',mmap_mode='r')
    datens=dates.astype('datetime64[ns]');year=dates.astype('datetime64[Y]').astype(int)+1970
    for et in ['CDH','HEAT','HR']:
        destination=OUT/'event_records'/et;destination.mkdir(exist_ok=True)
        counts={};diag={};years=np.zeros((2,25,5),np.int64);all_features=np.zeros((2,10),np.int64)
        grids={sc:np.load(OUT/'cache'/f'{et}_pixel_blocks_{sc}.npy') for sc in [20,50]}
        for sc in [20,50]:
            B=int(grids[sc].max()+1)
            for v in VARIANTS+['M1_CACHED']:
                counts[sc,v]=np.zeros((B,2,5,121),np.int64)
            diag[sc]=np.zeros((B,2,5),np.int64)
        full=np.load(M1/'cache'/et/'all_severity_sorted.npy',mmap_mode='r');nbase=0;nwrite=0;start=time.time()
        cols=['row','col','x','y','climate_cumulative_severity','Acomp_pre','Acomp_peak_lag_days','onset_idx','term_idx','event_num','followup','reason','valid','true_next_day']
        for j,p in enumerate(sorted((M1/'cache'/et).glob('recovery_*.parquet'))):
            d=pd.read_parquet(p,columns=cols);sev=d.climate_cumulative_severity.to_numpy(float)
            q=100*np.searchsorted(full,sev,side='right')/len(full);keep=q>=75
            if not keep.any():continue
            d=d.loc[keep].reset_index(drop=True);qq=q[keep];nbase+=len(d);g=np.where(qq<90,0,1).astype(np.int32)
            target_days=td[d.onset_idx.to_numpy(int)]+d.Acomp_peak_lag_days.to_numpy(float);peaks=np.searchsorted(td,target_days)
            assert (peaks<len(td)).all() and np.array_equal(td[peaks],target_days)
            z=np.zeros((len(d),5,10),np.int16);meta=np.zeros((len(d),6),np.int16)
            for row,idx in d.groupby('row').indices.items():
                sub=d.iloc[idx]
                zz,mm=extract(sub.col.to_numpy(np.int32),sub.term_idx.to_numpy(np.int32),peaks[idx].astype(np.int32),sub.Acomp_pre.to_numpy(np.float32),sub.true_next_day.to_numpy(np.int16),td,a[int(row)],on[int(row)])
                z[idx]=zz;meta[idx]=mm
            for sc in [20,50]:
                block=grids[sc][d.row.to_numpy(int),d.col.to_numpy(int)];assert (block>=0).all()
                for v,name in enumerate(VARIANTS):add_counts(counts[sc,name],block,g,z[:,v,0],z[:,v,1],z[:,v,2]==1)
                add_counts(counts[sc,'M1_CACHED'],block,g,d.followup.to_numpy(int),d.reason.to_numpy(int),d.valid.to_numpy()==1)
                ix=((block*2+g)*5+z[:,2,1]).astype(int);diag[sc]+=np.bincount(ix,minlength=diag[sc].size).reshape(diag[sc].shape)
                d[f'Block_{sc}km']=block
            yy=year[d.onset_idx.to_numpy(int)]-2000;ix=((g*25+yy)*5+z[:,2,1]).astype(int);years+=np.bincount(ix,minlength=years.size).reshape(years.shape)
            feats=np.column_stack([np.ones(len(d),int),z[:,2,2],z[:,2,9],meta[:,1],meta[:,2],meta[:,5],z[:,2,5],z[:,2,6]>0,meta[:,0]>=0,z[:,2,1]==4])
            for k in range(2):all_features[k]+=feats[g==k].sum(axis=0)
            d['Event_ID']=[f'{et}_{int(v):010d}' for v in d.event_num];d['Pixel_ID']=d.row*703+d.col;d['Event_Type']=et;d['Severity']=d.climate_cumulative_severity;d['Full_ECDF_Q']=qq;d['Severity_Group']=np.where(g==0,'High','Extreme')
            d['Event_Onset']=datens[d.onset_idx.to_numpy(int)];d['Event_Termination']=datens[d.term_idx.to_numpy(int)];d['Ecological_Peak_Date']=datens[peaks]
            t0=td[d.term_idx.to_numpy(int)];origin=datens[d.term_idx.to_numpy(int)]
            def date_offset(days):
                result=origin+days.astype('timedelta64[D]');result[days<0]=np.datetime64('NaT');return result
            d['Recovery_Date']=date_offset(z[:,2,3]);d['Recovery_Confirmation_Date']=date_offset(z[:,2,4]);d['Next_Extreme_Onset']=date_offset(meta[:,0]);d['Observation_End_Date']=datens[-1]
            d['Last_Valid_Followup_Observation']=date_offset(z[:,2,7]);d['Valid_Followup']=z[:,2,2];d['Followup_Days']=z[:,2,0];d['Outcome_Code']=z[:,2,1]
            labels=np.array(['RECOVERED','NEXT_EVENT_INTERRUPTION','ADMINISTRATIVE_CENSORING','ADMINISTRATIVE_CENSORING','INCOMPLETE_OBSERVATION']);d['Censoring_Reason']=labels[z[:,2,1]]
            d['Administrative_Subtype']=np.where(z[:,2,1]==2,'HORIZON_120',np.where(z[:,2,1]==3,'RECORD_END',''))
            d['Complete_NoNext_120']=meta[:,2];d['M1_Active_Phase_Before_New_Catalogue_Onset']=meta[:,1];d['Peak_At_Or_After_Next_Onset']=meta[:,5];d['Missing_Phases_Before_Outcome']=z[:,2,6];d['Unconfirmed_Single_Recovery_Observation']=z[:,2,5]
            for v,name in enumerate(VARIANTS):
                d[name+'_Duration']=z[:,v,0];d[name+'_Outcome']=z[:,v,1];d[name+'_Valid']=z[:,v,2]
            pq.write_table(pa.Table.from_pandas(d,preserve_index=False),destination/(p.stem+'.parquet'),compression='zstd');nwrite+=len(d)
            if j%30==0:print(et,j,nwrite,'elapsed',round(time.time()-start,1),flush=True)
            del d,z,meta;gc.collect()
        arrays={f'counts_{sc}_{name}':c for (sc,name),c in counts.items()}
        arrays.update({f'diag_{sc}':v for sc,v in diag.items()});arrays['year_diagnostics']=years;arrays['features']=all_features
        np.savez_compressed(OUT/'cache'/f'{et}_Aggregates.npz',**arrays)
        (OUT/'cache'/f'{et}_EXTRACT_READY.json').write_text(json.dumps(dict(Event=et,N_Base_High_Extreme=nbase,N_Written=nwrite,variants=VARIANTS,feature_columns=['N_base','N_valid','N_Trec0','N_active_before_new_onset','N_complete_noNext120','N_peak_after_next','N_unconfirmed_single','N_with_missing_before_outcome','N_next_onset_within120','N_incomplete']),indent=2),encoding='utf8')
        print(et,'EXTRACTION READY',nwrite,flush=True)
if __name__=='__main__':main()
