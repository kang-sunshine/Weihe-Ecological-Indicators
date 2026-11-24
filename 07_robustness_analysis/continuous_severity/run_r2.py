# -*- coding: utf-8 -*-
"""Read-only reuse of audited event metrics; no raster extraction or model training."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
import ast, json, hashlib, platform, sys, time, gc
import numpy as np
import pandas as pd

R=Path(r'.'); O=R/'EI_Figure6_R2_ContinuousRecovery'
M1=R/'EI_Reviewer1_Major1_Revision'; M2=R/'EI_Reviewer1_Major2_Revision'
M5=R/'EI_Reviewer1_Major5_Revision'; MIN=R/'EI_Reviewer1_Minor2_SingleIndicator'
PRIMARY='PRIMARY_POSTPEAK_CATALOG_INCLUSIVE'
CHECKS=[]; REPRO=[]; ER=[]; FR=[]; DAILY=[]; BOOT=[]

def check(name, ok, detail=''):
    CHECKS.append(dict(Check=name,Pass=bool(ok),Detail=str(detail)))
    if not ok: raise AssertionError(name+': '+str(detail))

def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for a in iter(lambda:f.read(8*1024*1024),b''):h.update(a)
    return h.hexdigest()

def source_functions(p,names):
    tree=ast.parse(p.read_text(encoding='utf-8-sig'))
    ns={'np':np}
    selected=[f for f in tree.body if isinstance(f,ast.FunctionDef) and f.name in names]
    assert len(selected)==len(names)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(p),'exec'),ns)
    return ns

def snapshots():
    roots=[R/n for n in ['Weihe database','Weihe database V2','Weihe database V3',
        'analysis_project','EI_Figure6_Redraw_DataPack',
        'EI_Reviewer1_Minor2_SingleIndicator']]
    roots += sorted(R.glob('EI_Reviewer1_Major*_Revision'))
    rows=[]
    for root in roots:
        if not root.exists():continue
        for p in root.rglob('*'):
            if p.is_file():
                st=p.stat();rows.append(dict(Path=str(p),Size=st.st_size,Mtime_ns=st.st_mtime_ns))
    return pd.DataFrame(rows).sort_values('Path').reset_index(drop=True)

def save(d,path):pd.DataFrame(d).to_csv(O/path,index=False,encoding='utf-8-sig')

def prepare():
    for n in ['data','scripts','qc','reports']: (O/n).mkdir(parents=True,exist_ok=True)
    before=snapshots();before.to_csv(O/'qc/Protected_Files_Before.csv',index=False)
    cfg=json.loads((M5/'code/configuration.json').read_text())
    cfg.update(Output=str(O),Severity_Bin_Width=5,Last_Bin_Includes_100=True,
        Reference='all finite original L53 hazard events, not clean-pre or recoverable subset',
        Recovery_Metric_Source='Minor2 native Acomp caches, independent all-hazard replay gated against M5',
        HEAT_Legacy_Version='original historical L53, not M5 observed-posttermination legacy',
        HEAT_Candidate_Policy='fixed archived original HEAT candidate grid; >=4 binned medians on each side; no expanded range',
        HEAT_Binning='exact medians on frozen original historical 80-quantile edges; M1 independent exact-median sensitivity implementation',
        HEAT_Uncertainty='point fits and spatial support diagnostics only; no new breakpoint bootstrap')
    (O/'scripts/configuration.json').write_text(json.dumps(cfg,indent=2),encoding='utf8')
    sources=[M5/'code/survival_m5.py',M5/'code/extract_m5.py',M5/'code/analyse_m5.py',M5/'code/configuration.json',
        M5/'M5_Figure6ef_SourceData.csv',M5/'M5_Spatial_Recovery_Uncertainty.csv',
        M1/'R1M1_Reproducible_Code/model_utils.py',M1/'R1M1_Reproducible_Code/independent_breakpoints.py',
        M1/'R1M1_Breakpoint_Replication.csv',M1/'R1M1_Breakpoint_Candidate_AIC.csv',
        M2/'code/catalog_overlap_m2.py',M2/'M2_HEAT_Only_Sample_Counts.csv',M2/'M2_HEAT_Only_Response_Comparison.csv',
        M2/'cache/HEAT_Membership_By_EventNum.npy',M2/'cache/HEAT_Clean_ClassFlags.npy',M2/'cache/HEAT_Clean_Responses.npy',
        MIN/'code/minor2_core.py',MIN/'cache/ACOMP_GATE.json',MIN/'audit/Acomp_Gate_Verification.csv',
        R/'EI_Reviewer1_Major6_Revision/M6_Decision_Report.md',R/'EI_Figure6_Redraw_DataPack/F6_QC.md',
        R/'EI_Figure6_Redraw_DataPack/F6_EF_Recovery_Contrasts.csv']
    for h in ['CDH','HEAT','HR']:
        sources += [MIN/f'cache/{h}/Acomp.npy', MIN/f'cache/{h}/Full_ECDF_Q.npy',MIN/f'cache/{h}/Block50.npy',
            MIN/f'paired/{h}_Fixed_Clean_Events.parquet',M1/f'cache/{h}/all_severity_sorted.npy',
            M5/f'cache/{h}_pixel_blocks_50.npy',M5/f'cache/{h}_Aggregates.npz']
    sources += [R/'EI_Reviewer1_Major8_Revision/cache/M8B/HEAT/Original_Historical_Bin_Edges.npy',
        R/'Weihe database V3/L5_EventEcoFunction/L5_4_ThresholdAnalysis/L5_4A_Severity_Response_Thresholds.csv']
    rows=[]
    for p in sources:
        check('source_exists_'+p.name,p.is_file(),p)
        st=p.stat();rows.append(dict(Path=str(p),Size=st.st_size,Mtime_ns=st.st_mtime_ns,SHA256=digest(p)))
    save(rows,'qc/Key_Input_SHA256.csv')
    env=dict(Python=sys.version,Executable=sys.executable,Platform=platform.platform(),Numpy=np.__version__,Pandas=pd.__version__)
    (O/'qc/Runtime_Environment.json').write_text(json.dumps(env,indent=2),encoding='utf8')
    check('independent_Acomp_replay_gate',json.loads((MIN/'cache/ACOMP_GATE.json').read_text())['Pass'])
    return cfg

def continuous(cfg):
    est=source_functions(M5/'code/survival_m5.py',['estimates'])['estimates']
    formal=pd.read_csv(M5/'M5_Figure6ef_SourceData.csv')
    spatial=pd.read_csv(M5/'M5_Spatial_Recovery_Uncertainty.csv')
    prevplot=pd.read_csv(R/'EI_Figure6_Redraw_DataPack/F6_EF_Recovery_Contrasts.csv')
    support=[]
    for h in ['CDH','HEAT','HR']:
        t=time.time(); folder=MIN/'cache'/h
        e=pd.read_parquet(MIN/f'paired/{h}_Fixed_Clean_Events.parquet',columns=['event_num','row','col','climate_cumulative_severity']).astype({'event_num':'int32','row':'int16','col':'int16'})
        ids=e.event_num.to_numpy();s=e.climate_cumulative_severity.to_numpy(np.float64)
        q=np.load(folder/'Full_ECDF_Q.npy',mmap_mode='r');v=np.load(folder/'Acomp.npy',mmap_mode='r');b=np.load(folder/'Block50.npy',mmap_mode='r')
        grid=np.load(M5/f'cache/{h}_pixel_blocks_50.npy',mmap_mode='r');full=np.load(M1/f'cache/{h}/all_severity_sorted.npy',mmap_mode='r')
        B=int(grid.max()+1);N=len(e);pix=e.row.to_numpy(np.int32)*703+e.col.to_numpy(np.int32)
        check(h+'_cache_shapes',v.shape==(N,18) and len(q)==N and len(b)==N)
        check(h+'_original_unique_ID',len(np.unique(ids))==N and ids.min()>0)
        check(h+'_M5_block_labels',np.array_equal(b,grid[e.row.to_numpy(),e.col.to_numpy()]) and b.min()>=0)
        check(h+'_full_severity_finite_sorted',np.isfinite(full).all() and (np.diff(full)>=0).all())
        for i in range(0,N,250000):
            check(h+'_exact_full_ECDF_'+str(i),np.array_equal(q[i:i+250000],100*np.searchsorted(full,s[i:i+250000],side='right')/len(full)))
        bn=np.minimum((np.asarray(q)//5).astype(np.uint8),19)
        check(h+'_exclusive_exhaustive_bins',len(bn)==N and bn.max()<=19 and ((q>=0)&(q<=100)).all())
        valid=np.asarray(v[:,9]==1);base=np.isfinite(v[:,1]);check(h+'_followup_requires_baseline_peak',not (valid&~base).any())
        dur=v[valid,7].astype(np.int32);oc=v[valid,8].astype(np.int32)
        check(h+'_valid_outcome_times',(dur>=0).all() and (dur<=120).all() and (oc>=0).all() and (oc<5).all())
        flat=(((b[valid]*20+bn[valid])*5+oc)*121+dur).astype(np.int64)
        C=np.bincount(flat,minlength=B*20*5*121).reshape(B,20,5,121)
        check(h+'_eligible_counts_close',int(C.sum())==int(valid.sum()))
        np.savez_compressed(O/f'data/{h}_Event_Bin_Assignment.npz',event_num=ids,severity_bin=bn,baseline_peak_eligible=base,followup_eligible=valid)
        np.savez_compressed(O/f'data/{h}_Block_Bin_Outcome_Day_Counts.npz',counts=C)
        W=np.random.default_rng(cfg['seed']+50).multinomial(B,np.full(B,1/B),size=cfg['bootstrap_total'])
        check(h+'_same_Minor2_M5_weights',np.array_equal(W,np.load(folder/'Bootstrap_50km_Weights.npy')))
        np.save(O/f'data/{h}_Bootstrap_50km_Weights.npy',W)
        pt=est(C.sum(axis=0));fb=est((W@C.reshape(B,-1)).reshape(len(W),20,5,121))
        # Re-pool original records, never average bin-wise survival estimators.
        G=np.stack([C[:,15:18].sum(axis=1),C[:,18:20].sum(axis=1)],axis=1)
        saved=np.load(M5/f'cache/{h}_Aggregates.npz')[f'counts_50_{PRIMARY}']
        check(h+'_entire_M5_block_group_outcome_day_tensor',np.array_equal(G,saved))
        gp=est(G.sum(axis=0));gb=est((W@G.reshape(B,-1)).reshape(len(W),2,5,121))
        for gi,label in enumerate(['High','Extreme']):
            old=formal[(formal.Event==h)&(formal.Severity_Group==label)].iloc[0]
            groupmask=(q>=75)&(q<90) if gi==0 else q>=90
            for metric,value,ref in [('N_events',int((base&groupmask).sum()),old.N_events),('N_survival_eligible',int(G[:,gi].sum()),old.N_survival_eligible),('N_risk_120',gp['N_risk_120'][gi],old.N_risk_120),('P120',gp['P120'][gi],old.P120),('RMPT120',gp['RMPT120'][gi],old.RMPT120)]:
                err=float(value-ref);ok=abs(err)<1e-10
                REPRO.append(dict(Event=h,Comparison=label,Metric=metric,Recalculated=value,M5_Reference=ref,Difference=err,Pass=ok,Method='re-pooled original event counts, unchanged M5 estimates'))
                check(h+'_'+label+'_'+metric,ok,err)
        for metric in ['P120','RMPT120']:
            value=float(gp[metric][1]-gp[metric][0]);vals=gb[metric][:,1]-gb[metric][:,0]
            ref=spatial[(spatial.Event==h)&(spatial.Analysis==PRIMARY)&(spatial.Metric==metric)&(spatial.Comparison=='Extreme_minus_High')&(spatial.Block_km==50)].iloc[0]
            ci=np.nanquantile(vals,[.025,.975]);factor=100 if metric=='P120' else 1
            plot=prevplot[(prevplot.Event==h)&(prevplot.Metric==('KM_P120' if metric=='P120' else 'KM_RMPT120'))].iloc[0]
            for item,new,old in [('Point',value,ref.Estimate),('CI_Lower',ci[0],ref.CI_Lower),('CI_Upper',ci[1],ref.CI_Upper),('Figure6_Display',value*factor,plot.Difference)]:
                err=float(new-old);ok=abs(err)<1e-10
                REPRO.append(dict(Event=h,Comparison='Extreme_minus_High',Metric=metric+'_'+item,Recalculated=new,M5_Reference=old,Difference=err,Pass=ok))
                check(h+'_'+metric+'_'+item,ok,err)
        for j in range(20):
            mask=valid&(bn==j);clean=(bn==j);counts=C[:,j].sum(axis=0);nn=int(counts.sum());out=counts.sum(axis=1).astype(int)
            nr=int(pt['N_risk_120'][j]);risk=pt['Risk'][j];pos=np.flatnonzero(risk>0);last=int(pos[-1]) if len(pos) else None
            row=dict(event_type=h,bin_left=5*j,bin_right=5*j+5,bin_midpoint=5*j+2.5,
                bin_rule='[left,right)' if j<19 else '[95,100]',n_events=nn,n_clean_events=int(clean.sum()),
                n_baseline_peak=int((clean&base).sum()),n_excluded_no_valid_followup=int((clean&base&~valid).sum()),
                n_pixels=len(np.unique(pix[mask])),n_blocks=int(np.count_nonzero(C[:,j].sum(axis=(1,2)))),
                n_recovered=int(out[0]),n_censored_new_event=int(out[1]),n_censored_other=int(out[2:].sum()),
                n_censored_admin120=int(out[2]),n_censored_record_end=int(out[3]),n_censored_missing=int(out[4]),
                censoring_rate=float(1-out[0]/nn) if nn else np.nan,n_risk_120=nr,
                last_supported_day=last,n_risk_last_supported=int(risk[last]) if last is not None else 0,
                n_risk_day119=int(risk[119]),n_peak_at_or_after_next=int((mask&(v[:,17]==1)).sum()),
                ECDF_reference='ALL_FINITE_ORIGINAL_L53_HAZARD_FLOAT64_RIGHT_UPPER_RANK',n_ECDF_reference=len(full),
                bootstrap_block_km=50,bootstrap_n=len(W),bootstrap_seed=cfg['seed']+50,n_resampled_whole_blocks=B,
                Version=PRIMARY,Source=str(folder/'Acomp.npy'),endpoint_rule='M5: risk120>0 or KM_S120<=1e-14',
                Uncertainty='pointwise percentile CI; fixed original event ECDF; common whole-block weights')
            check(h+'_bin_'+str(j)+'_count_closure',nn==int(out.sum()) and nn==int(mask.sum()) and int(out[0])+int(out[1])+int(out[2:].sum())==nn)
            for metric,container in [('P120',ER),('RMPT120',FR)]:
                vals=fb[metric][:,j];good=np.isfinite(vals);nv=int(good.sum());value=float(pt[metric][j])
                ci=np.quantile(vals[good],[.025,.975]) if nv>=cfg['bootstrap_min_valid'] and np.isfinite(value) else [np.nan,np.nan]
                status='EMPTY_BIN' if nn==0 else 'NOT_ESTIMABLE_ENDPOINT' if not np.isfinite(value) else 'PARTIAL_BOOTSTRAP_SUPPORT' if nv<cfg['bootstrap_min_valid'] else 'ESTIMABLE'
                rr=dict(**row,**{metric:value},CI_lower=ci[0],CI_upper=ci[1],bootstrap_valid_n=nv,bootstrap_success_rate=nv/len(W),estimate_status=status,
                    missing_reason='no at-risk subject at120 and positive remaining KM survival' if status=='NOT_ESTIMABLE_ENDPOINT' else 'fewer than200 valid whole-block replicates' if status=='PARTIAL_BOOTSTRAP_SUPPORT' else 'no valid followup in bin' if status=='EMPTY_BIN' else '')
                container.append(rr)
                for rep,val in enumerate(vals):BOOT.append(dict(event_type=h,bin_left=5*j,bin_right=5*j+5,replicate=rep,metric=metric,estimate=val,endpoint_estimable=bool(np.isfinite(val))))
                maxv=1 if metric=='P120' else 120
                check(h+'_'+str(j)+'_'+metric+'_range',not np.isfinite(value) or 0<=value<=maxv)
                check(h+'_'+str(j)+'_'+metric+'_CI_range',not np.isfinite(ci).all() or 0<=ci[0]<=ci[1]<=maxv)
                check(h+'_'+str(j)+'_'+metric+'_missing_endpoint_preserved',np.isfinite(value) or (np.isnan(ci).all() and nr==0))
            for day in range(121):DAILY.append(dict(event_type=h,bin_left=5*j,bin_right=5*j+5,day=day,n_risk=int(risk[day]),n_recovered=int(counts[0,day]),n_censored=int(counts[1:,day].sum()),KM_S=float(pt['KM_S'][j,day]),KM_Recovery=float(1-pt['KM_S'][j,day]),Endpoint120_estimable=bool(np.isfinite(pt['P120'][j]))))
        support.append(dict(Event=h,N_Full_ECDF=len(full),N_Clean=N,N_Baseline_Peak=int(base.sum()),N_Valid_Followup=int(valid.sum()),N_50km_FullBlocks=B,N_Assigned=N,Event_ID_SHA256=hashlib.sha256(ids.tobytes()).hexdigest(),Severity_SHA256=hashlib.sha256(s.tobytes()).hexdigest()))
        save(ER,'data/F6_R2_E_Continuous_P120.csv');save(FR,'data/F6_R2_F_Continuous_RMPT120.csv');save(REPRO,'qc/F6_R2_M5_Reproduction_Check.csv')
        print('CONTINUOUS',h,'N',int(valid.sum()),'seconds',round(time.time()-t,1),flush=True)
        del e,s,v,q,b,grid,full,C,G,fb,gb,pt,gp,mask,valid,base,bn,pix,flat;gc.collect()
    save(support,'qc/Event_Cohort_Closure.csv');save(DAILY,'data/F6_R2_Daily_KM.csv');save(BOOT,'data/F6_R2_Bootstrap_Replicates.csv')

def heat_exclusive():
    t=time.time();utils=source_functions(M1/'R1M1_Reproducible_Code/model_utils.py',['fit']);fit=utils['fit']
    mat=np.load(M2/'cache/HEAT_Clean_Responses.npy',mmap_mode='r');flags=np.load(M2/'cache/HEAT_Clean_ClassFlags.npy',mmap_mode='r')
    membership=np.load(M2/'cache/HEAT_Membership_By_EventNum.npy',mmap_mode='r');allf=np.load(M2/'cache/HEAT_All_ClassFlags.npy',mmap_mode='r')
    full=np.load(M1/'cache/HEAT/all_severity_sorted.npy',mmap_mode='r')
    e=pd.read_parquet(MIN/'paired/HEAT_Fixed_Clean_Events.parquet',columns=['event_num','row','col','climate_cumulative_severity']).astype({'event_num':'int32','row':'int16','col':'int16'})
    s=e.climate_cumulative_severity.to_numpy(np.float64);ids=e.event_num.to_numpy();pix=e.row.to_numpy(np.int32)*703+e.col.to_numpy(np.int32);blocks=np.load(MIN/'cache/HEAT/Block50.npy',mmap_mode='r')
    check('HEAT_M2_severity_ID_alignment',np.array_equal(s,mat[:,0]) and np.array_equal(flags,membership[ids]))
    only=(flags&2)>0;overlap=~only
    counts=pd.read_csv(M2/'M2_HEAT_Only_Sample_Counts.csv');des=pd.read_csv(M2/'M2_HEAT_Only_Response_Comparison.csv')
    groups={'ALL_HEAT':np.ones(len(e),bool),'HEAT_ONLY_NO_CDH_MERGED_INTERVAL':only,'HEAT_WITH_CDH_MERGED_OVERLAP':overlap}
    countrows=[]
    for label,mask in groups.items():
        rawmask=np.ones(len(allf),bool) if label=='ALL_HEAT' else ((allf&2)>0 if label.startswith('HEAT_ONLY') else (allf&2)==0)
        for cohort,n in [('ALL_ORIGINAL_HEAT',int(rawmask.sum())),('CLEAN_PRE_FINITE_SEVERITY',int(mask.sum()))]:
            expected=int(counts[(counts.Cohort==cohort)&(counts.Group==label)].iloc[0].N_Events)
            check(label+'_'+cohort+'_M2_count',n==expected,str(n))
            countrows.append(dict(Group=label,Cohort=cohort,N=n,M2_Reference=expected))
        vals=mat[mask,3];fv=vals[np.isfinite(vals)];ref=des[(des.Cohort=='CLEAN_PRE_FINITE_SEVERITY')&(des.Group==label)&(des.Response=='Acomp_legacy_anomaly_days')].iloc[0]
        check(label+'_M2_legacy_n',len(fv)==int(ref.N_Valid))
        check(label+'_M2_legacy_median',np.isclose(np.median(fv),ref.Median,rtol=0,atol=1e-10))
    check('HEAT_complete_disjoint_partition',int(only.sum())+int(overlap.sum())==len(e))
    save(countrows,'qc/HEAT_M2_Count_Reproduction.csv')
    edges=np.load(R/'EI_Reviewer1_Major8_Revision/cache/M8B/HEAT/Original_Historical_Bin_Edges.npy')
    bn=np.clip(np.searchsorted(edges,s,side='right')-1,0,len(edges)-2)
    ac=pd.read_csv(M1/'R1M1_Breakpoint_Candidate_AIC.csv');cc=np.sort(ac[(ac.Event=='HEAT')&(ac.Response=='Acomp_legacy')&(ac.Analysis=='SAVED_BIN_historical')].Candidate_S.unique())
    original=pd.read_csv(R/'Weihe database V3/L5_EventEcoFunction/L5_4_ThresholdAnalysis/L5_4A_Severity_Response_Thresholds.csv')
    old=original[(original.event_type=='HEAT')&(original.response=='Acomp_legacy')].iloc[0]
    out=[dict(Event='HEAT',Group='ALL_HEAT',Analysis='ARCHIVED_ORIGINAL_SAMPLED_BIN_REFERENCE',S_star=old.breakpoint_severity,Q_full=100*np.searchsorted(full,old.breakpoint_severity,side='right')/len(full),Historical_Q=old.breakpoint_percentile if 'breakpoint_percentile' in old else np.nan,Delta_AIC=old.delta_AIC_linear_minus_segmented,Search_Min=cc.min(),Search_Max=cc.max(),Edge_Hit_Lower=False,Edge_Hit_Upper=True,Status='EDGE_CONSTRAINED',Uncertainty_Status='historical candidate at upper boundary; not independently identified threshold',Response_Version='ORIGINAL_L53_LEGACY',Source=str(R/'Weihe database V3/L5_EventEcoFunction/L5_4_ThresholdAnalysis/L5_4A_Severity_Response_Thresholds.csv'))]
    bins=[];curves=[]
    for label,mask in groups.items():
        valid=mask&np.isfinite(mat[:,3]);nb=int(valid.sum());x=[];y=[];ns=[];sub=s[mask];sv=s[valid]
        check(label+'_finite_legacy_nonempty',nb>0)
        for j in range(len(edges)-1):
            sel=valid&(bn==j);n=int(sel.sum())
            if n<500:continue
            xx=float(np.median(s[sel]));yy=float(np.median(mat[sel,3]));x.append(xx);y.append(yy);ns.append(n)
            bct=np.bincount(blocks[sel],minlength=int(blocks.max()+1))
            bins.append(dict(Group=label,Bin=j,N_Events=n,S_Median=xx,Q_full=100*np.searchsorted(full,xx,side='right')/len(full),Legacy_Median=yy,N_50km_Blocks=int(np.count_nonzero(bct)),Max_Block_Event_Share=float(bct.max()/n),Estimator='exact medians, original historical fixed bin edges; M1 audited exact-median sensitivity'))
        x=np.array(x);y=np.array(y);ns=np.array(ns);eligible=np.array([c for c in cc if (x<c).sum()>=4 and (x>c).sum()>=4]);bct=np.bincount(blocks[valid],minlength=int(blocks.max()+1));B=int(np.count_nonzero(bct))
        common=dict(Event='HEAT',Group=label,N_Clean_Events=int(mask.sum()),N_Legacy_Events=nb,N_Pixels=len(np.unique(pix[valid])),N_50km_Blocks=B,N_Bins=len(x),N_Eligible_Candidates=len(eligible),Search_Min_Original=cc.min(),Search_Max_Original=cc.max(),Search_Min=float(eligible.min()) if len(eligible) else np.nan,Search_Max=float(eligible.max()) if len(eligible) else np.nan,
            Severity_Min=float(sv.min()),Severity_Q05=float(np.quantile(sv,.05)),Severity_Median=float(np.median(sv)),Severity_Q95=float(np.quantile(sv,.95)),Severity_Max=float(sv.max()),Max_Block_Event_Share=float(bct.max()/nb),
            Full_ECDF_Reference_N=len(full),Q_Reference='full original HEAT catalogue float64 right ECDF',Q_Only_Reference='HEAT-only clean-pre finite-severity population, not finite-legacy subset',
            Original_Reference_S=old.breakpoint_severity,Original_Reference_Delta_AIC=old.delta_AIC_linear_minus_segmented,Response_Version='ORIGINAL_L53_LEGACY',Source=str(M2/'cache/HEAT_Clean_Responses.npy'),
            Weight='sqrt(full valid event bin count) on RSS; AIC k_linear2,k_segmented4',Median_Estimator='exact, not original bounded20k sampled medians',
            Uncertainty_Status='POINT_FIT_ONLY; no new spatial breakpoint bootstrap; location stability NOT VERIFIED',Minimum_Support='>=500/bin, >=12 bins, >=4 bins each side, >=10 50km blocks',
            Candidate_Policy='archived original candidate grid, not expanded or rescaled to subgroup quantiles')
        for method,cand in [('FIXED_ORIGINAL_REFERENCE',np.array([old.breakpoint_severity])),('FREE_WITHIN_ORIGINAL_CANDIDATES',eligible)]:
            row=dict(**common,Analysis=method)
            candidates=np.array([c for c in cand if (x<c).sum()>=4 and (x>c).sum()>=4])
            if len(x)<12 or B<10 or not len(candidates) or np.ptp(y)<=1e-12:
                row.update(Status='NOT ESTIMABLE',Missing_Reason='insufficient original-range tail bins/spatial support or constant response');out.append(row);continue
            b0,a0,r0=fit(x,y,np.sqrt(ns)); fits=[fit(x,y,np.sqrt(ns),c) for c in candidates];aics=np.array([a[1] for a in fits]);i=int(np.argmin(aics));st=float(candidates[i]);coef,aic,rss=fits[i];delta=float(a0-aic)
            low=bool(st==eligible.min());up=bool(st==eligible.max());ol=bool(st==cc.min());ou=bool(st==cc.max())
            row.update(S_star=st,Q_full=100*np.searchsorted(full,st,side='right')/len(full),Q_HEAT_Only=100*np.mean(s[only]<=st),Q_Within_Group_Clean=100*np.mean(sub<=st),Linear_AIC=a0,Segmented_AIC=aic,Delta_AIC=delta,Slope_Pre=coef[1],Slope_Post=coef[1]+coef[2],Slope_Change=coef[2],Edge_Hit_Lower=low,Edge_Hit_Upper=up,Original_Edge_Hit_Lower=ol,Original_Edge_Hit_Upper=ou,Support_Level='STRONG' if delta>=10 else 'MODERATE' if delta>=4 else 'UNSUPPORTED',Status='NO_SUPPORTED_BREAKPOINT' if delta<4 else 'FIXED_REFERENCE_COMPARISON' if method.startswith('FIXED') else 'EDGE_CONSTRAINED' if low or up else 'CONDITIONAL_POINT_CANDIDATE',Delta_S_From_Original=st-old.breakpoint_severity)
            out.append(row)
            if method.startswith('FREE'):
                for c,a in zip(candidates,aics):curves.append(dict(Group=label,S_candidate=c,Q_full=100*np.searchsorted(full,c,side='right')/len(full),AIC=a,Delta_AIC_Linear=a0-a,Delta_AIC_Best=a-aic))
    save(out,'data/F6_R2_HEAT_Exclusive_Breakpoint.csv');save(bins,'data/F6_R2_HEAT_Legacy_Bins.csv');save(curves,'data/F6_R2_HEAT_Candidate_AIC.csv')
    print('HEAT POINT FITS',len(out),'seconds',round(time.time()-t,1),flush=True)

def finish():
    a=pd.read_csv(O/'qc/Protected_Files_Before.csv');b=snapshots();b.to_csv(O/'qc/Protected_Files_After.csv',index=False)
    check('all_protected_files_unchanged',a.equals(b),f'{len(a)} before / {len(b)} after; size and mtime_ns')
    hashes=pd.read_csv(O/'qc/Key_Input_SHA256.csv');check('key_input_SHA256_unchanged',all(digest(Path(r.Path))==r.SHA256 for r in hashes.itertuples()))
    er=pd.read_csv(O/'data/F6_R2_E_Continuous_P120.csv');fr=pd.read_csv(O/'data/F6_R2_F_Continuous_RMPT120.csv')
    check('60_hazard_bins_each',len(er)==60 and len(fr)==60 and set(er.event_type)=={'CDH','HEAT','HR'})
    common=[c for c in er.columns if c not in ['P120','CI_lower','CI_upper']]
    check('P120_RMPT_same_bins_population_status',er[common].equals(fr[common]))
    check('M5_reproduction_all_pass',pd.read_csv(O/'qc/F6_R2_M5_Reproduction_Check.csv').Pass.all())
    save(CHECKS,'qc/Verification_Checks.csv')
    (O/'qc/COMPUTATION_READY.json').write_text(json.dumps(dict(Checks=len(CHECKS),All_Pass=all(r['Pass'] for r in CHECKS),Protected_Files=len(a),Key_SHA256=len(hashes)),indent=2),encoding='utf8')
    print('COMPUTATION READY',len(CHECKS),'checks',flush=True)

if __name__=='__main__':
    cfg=prepare();continuous(cfg);heat_exclusive();finish()
