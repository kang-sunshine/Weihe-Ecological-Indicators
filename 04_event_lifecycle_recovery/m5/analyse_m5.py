# -*- coding: utf-8 -*-
from pathlib import Path
import os,json,time,gc
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np
import pandas as pd
from survival_m5 import estimates,logrank,numerical_tests
OUT=Path(r'.\EI_Reviewer1_Major5_Revision');M1=Path(r'.\EI_Reviewer1_Major1_Revision')
PRIMARY='PRIMARY_POSTPEAK_CATALOG_INCLUSIVE';GAP='FIRST_MISSING_GAP_SENSITIVITY';UNC='UNCONTAMINATED_PEAK_SUBSET';COMP='COMPLETE_UNINTERRUPTED_120_SUBSET'
VARIANTS=['M1_CACHED','PEAK_ONLY_ACTIVE_NEXT_EXCLUSIVE','PEAK_CATALOG_NEXT_EXCLUSIVE',PRIMARY,GAP,'NO_PEAK_RULE_CATALOG_SENSITIVITY']
METRICS=['P120','RMPT120','AJ_Recovery120','AJ_Next120']
SEED=20261109;NB=250;MIN_VALID=200

def supplemental_counts(et,Bs):
    cs={(scale,name):np.zeros((Bs[scale],2,5,121),np.int64) for scale in [20,50] for name in [UNC,COMP]}
    for p in sorted((OUT/'event_records'/et).glob('*.parquet')):
        d=pd.read_parquet(p,columns=['Severity_Group','Followup_Days','Outcome_Code','Valid_Followup','Peak_At_Or_After_Next_Onset','Complete_NoNext_120','Block_20km','Block_50km'])
        g=d.Severity_Group.eq('Extreme').to_numpy(int);tt=d.Followup_Days.to_numpy(int);oo=d.Outcome_Code.to_numpy(int);valid=d.Valid_Followup.eq(1).to_numpy()
        for scale in [20,50]:
            b=d[f'Block_{scale}km'].to_numpy(int)
            for name,mask in [(UNC,valid&d.Peak_At_Or_After_Next_Onset.eq(0).to_numpy()),(COMP,valid&d.Complete_NoNext_120.eq(1).to_numpy())]:
                ix=(((b[mask]*2+g[mask])*5+oo[mask])*121+tt[mask]).astype(int)
                cs[scale,name]+=np.bincount(ix,minlength=cs[scale,name].size).reshape(cs[scale,name].shape)
    return cs

def endpoint_status(risk,p):return 'ENDPOINT_NOT_ESTIMABLE' if not np.isfinite(p) else ('LOW_ENDPOINT_SUPPORT' if risk<20 else 'ESTIMABLE')

def main():
    assert numerical_tests()
    group=[];comparisons=[];spatial=[];curves=[];logs=[];byyear=[];byblock=[];stability=[];features=[];bridge=[];replicates=[]
    historic=pd.read_csv(M1/'audit/Historical_Recovery_Replication.csv');header=pd.read_csv(M1/'M1_Closeout_Minimal/M1_Figure6_Recovery_Comparison.csv')
    for et in ['CDH','HEAT','HR']:
        z=np.load(OUT/'cache'/f'{et}_Aggregates.npz');Bs={sc:z[f'counts_{sc}_{PRIMARY}'].shape[0] for sc in [20,50]}
        cs={(sc,name):z[f'counts_{sc}_{name}'] for sc in [20,50] for name in VARIANTS};cs.update(supplemental_counts(et,Bs))
        ff=z['features'];diag=z['diag_20'];yy=z['year_diagnostics'];meta=json.loads((OUT/'cache'/f'{et}_EXTRACT_READY.json').read_text())
        for g,label in enumerate(['High','Extreme']):
            features.append(dict(Event=et,Severity_Group=label,**{k:int(ff[g,j]) for j,k in enumerate(meta['feature_columns'])}))
            for yr in range(25):
                v=yy[g,yr];n=int(v.sum())
                byyear.append(dict(Event=et,Group=label,Onset_Year=2000+yr,N=n,N_recovered=int(v[0]),N_next=int(v[1]),N_admin=int(v[2]+v[3]),N_incomplete=int(v[4]),Next_Rate=v[1]/n if n else np.nan,Admin_Rate=(v[2]+v[3])/n if n else np.nan,Incomplete_Rate=v[4]/n if n else np.nan))
            for sc in [20,50]:
                D=z[f'diag_{sc}']
                for b in range(Bs[sc]):
                    v=D[b,g];n=int(v.sum())
                    byblock.append(dict(Event=et,Group=label,Block_km=sc,Block_ID=b,N=n,N_recovered=int(v[0]),N_next=int(v[1]),N_admin=int(v[2]+v[3]),N_incomplete=int(v[4]),Next_Rate=v[1]/n if n else np.nan,Admin_Rate=(v[2]+v[3])/n if n else np.nan,Incomplete_Rate=v[4]/n if n else np.nan))
        for name in VARIANTS+[UNC,COMP]:
            C=cs[20,name].sum(axis=0);f=estimates(C)
            for g,label in enumerate(['High','Extreme']):
                v=C[g].sum(axis=1).astype(int);n=int(v.sum());Nbase=int(ff[g,0]) if name==PRIMARY else n
                counts_all=diag[:,g].sum(axis=0).astype(int) if name==PRIMARY else v
                group.append(dict(Event=et,Analysis=name,Severity_Group=label,ECDF_Reference='ALL_FINITE_L53_PER_HAZARD_FLOAT64_RIGHT_UPPER_RANK',Q_Lower=75 if g==0 else 90,Q_Upper=90 if g==0 else 100,Group_Boundary='75<=q<90' if g==0 else '90<=q<=100',N_events=Nbase,N_survival_eligible=n,N_excluded_no_valid_origin=Nbase-n,N_recovered=int(counts_all[0]),N_next_event=int(counts_all[1]),N_admin_censored=int(counts_all[2]+counts_all[3]),N_admin_120=int(counts_all[2]),N_admin_record_end=int(counts_all[3]),N_incomplete=int(counts_all[4]),Censoring_rate=1-counts_all[0]/Nbase if Nbase else np.nan,Survival_Censoring_Rate=1-v[0]/n if n else np.nan,Recovery_probability_120d=float(f['P120'][g]),P120=float(f['P120'][g]),RMPT120=float(f['RMPT120'][g]),RMPT120_Trapezoid=float(f['RMPT_Trapezoid'][g]),N_risk_120=int(f['N_risk_120'][g]),AJ_Recovery_Before_Next_120d=float(f['AJ_Recovery120'][g]),AJ_Next_Before_Recovery_120d=float(f['AJ_Next120'][g]),Endpoint_Status=endpoint_status(f['N_risk_120'][g],f['P120'][g]),Estimator_Status='RULE_DEFINED_CONDITIONAL_OBSERVATIONAL',RMPT_Integration='RIGHT_CONTINUOUS_STEP_0_TO120',Selection_Status='NO_NEW_ONSET_IN120_AND_ALL_SCHEDULED_PHASES_FINITE_SELECTION_BIASED' if name==COMP else ('CONDITIONAL_ON_PEAK_PRECEDING_NEXT_ONSET_SELECTION_BIASED' if name==UNC else 'ORIGINAL_CLEAN_PRE_BASELINE_AND_FINITE_PEAK_COHORT')))
                if name in [PRIMARY,GAP]:
                    for day in range(121):curves.append(dict(Event=et,Analysis=name,Group=label,Day=day,KM_S=float(f['KM_S'][g,day]),KM_Recovery_Probability=float(1-f['KM_S'][g,day]),AJ_Recovery_CIF=float(f['AJ_Recovery_Curve'][g,day]),AJ_Next_CIF=float(f['AJ_Next_Curve'][g,day]),N_risk=int(f['Risk'][g,day])))
            lr=logrank(C);logs.append(dict(Event=et,Analysis=name,**lr,Inference='NAIVE_EVENT_LEVEL_LOGRANK; dependent pixel-events; KM censoring assumptions; not spatially robust significance',P_Underflow=lr['Logrank_p']==0))
        # A common random block weight set for all necessary estimands, not repeated experiments.
        for scale in [20,50]:
            B=Bs[scale];rng=np.random.default_rng(SEED+scale);W=rng.multinomial(B,np.full(B,1/B),size=NB)
            for name in [PRIMARY,GAP,UNC,COMP]:
                C=cs[scale,name];t0=time.time()
                pilot=(W[:50]@C.reshape(B,-1)).reshape(50,2,5,121);fp=estimates(pilot);elapsed=time.time()-t0
                print(et,scale,name,'pilot50 sec',round(elapsed,3),flush=True)
                allc=(W@C.reshape(B,-1)).reshape(NB,2,5,121);fb=estimates(allc);point=estimates(C.sum(axis=0))
                for metric in METRICS:
                    for g,label in [(0,'High'),(1,'Extreme'),(-1,'Extreme_minus_High')]:
                        vals=fb[metric][:,g] if g>=0 else fb[metric][:,1]-fb[metric][:,0]
                        value=float(point[metric][g]) if g>=0 else float(point[metric][1]-point[metric][0]);valid=int(np.isfinite(vals).sum())
                        ci=np.nanquantile(vals,[.025,.975]) if valid>=MIN_VALID else [np.nan,np.nan]
                        record=dict(Event=et,Analysis=name,Metric=metric,Comparison=label,Block_km=scale,Estimate=value,CI_Lower=ci[0],CI_Upper=ci[1],Bootstrap_N=NB,Bootstrap_Valid=valid,N_Whole_Blocks=B,Unit='FULL_SPATIAL_BLOCK_WITH_ALL_PIXEL_EVENTS',CI_Method='pointwise percentile; fixed ECDF; common block weights',CI_Status='ESTIMABLE_CONDITIONAL' if valid>=MIN_VALID else 'PARTIAL_ENDPOINT_SUPPORT',Fraction_Negative=float(np.mean(vals[np.isfinite(vals)]<0)) if np.isfinite(vals).any() else np.nan,Fraction_Positive=float(np.mean(vals[np.isfinite(vals)]>0)) if np.isfinite(vals).any() else np.nan)
                        spatial.append(record)
                        if label=='Extreme_minus_High':
                            for j,val in enumerate(vals):replicates.append(dict(Event=et,Analysis=name,Block_km=scale,Replicate=j,Metric=metric,Contrast=val))
                            for nrep in [50,100,200,250]:
                                vv=vals[:nrep];ok=int(np.isfinite(vv).sum());cc=np.nanquantile(vv,[.025,.975]) if ok else [np.nan,np.nan]
                                stability.append(dict(Event=et,Analysis=name,Block_km=scale,Metric=metric,N_Replicates=nrep,N_Valid=ok,CI_Lower=cc[0],CI_Upper=cc[1],Pilot50_Seconds=elapsed))
                # KM and AJ are listed separately with their distinct estimands.
                for g,label in [(0,'High'),(1,'Extreme'),(-1,'Extreme_minus_High')]:
                    row=dict(Event=et,Analysis=name,Group_Or_Contrast=label,Bootstrap_Block_km=scale)
                    for metric in METRICS:
                        rr=next(v for v in reversed(spatial) if v['Event']==et and v['Analysis']==name and v['Metric']==metric and v['Comparison']==label and v['Block_km']==scale)
                        row[metric]=rr['Estimate'];row[metric+'_CI_Lower']=rr['CI_Lower'];row[metric+'_CI_Upper']=rr['CI_Upper'];row[metric+'_CI_Status']=rr['CI_Status']
                    row['KM_Estimand']='net recovery under independent interruption censoring assumption';row['AJ_Estimand']='observed recovery criterion before a new catalogue event onset, under administrative/missing censoring assumptions';row['Causal_Interpretation']='NONE';comparisons.append(row)
                if name==PRIMARY:
                    D=z[f'diag_{scale}'];Db=(W@D.reshape(B,-1)).reshape(NB,2,5);pp=D.sum(axis=0)
                    for metric,index in [('Next_Interruption_Rate',1),('Administrative_Censor_Rate',2),('Incomplete_Observation_Rate',4)]:
                        values=Db[:,:,index]/Db.sum(axis=2)
                        if index==2:values=(Db[:,:,2]+Db[:,:,3])/Db.sum(axis=2)
                        point_rates=pp[:,index]/pp.sum(axis=1) if index!=2 else (pp[:,2]+pp[:,3])/pp.sum(axis=1)
                        vals=values[:,1]-values[:,0];ci=np.quantile(vals,[.025,.975]);spatial.append(dict(Event=et,Analysis=name,Metric=metric,Comparison='Extreme_minus_High',Block_km=scale,Estimate=point_rates[1]-point_rates[0],CI_Lower=ci[0],CI_Upper=ci[1],Bootstrap_N=NB,Bootstrap_Valid=NB,N_Whole_Blocks=B,Unit='FULL_SPATIAL_BLOCK_WITH_ALL_PIXEL_EVENTS',CI_Method='percentile on all baseline-eligible group records',CI_Status='ESTIMABLE_CONDITIONAL'))
                del allc,fb,pilot,fp;gc.collect()
        # Ordered numerical bridge; effects are order-dependent bookkeeping, not causal attribution.
        with np.load(M1/'cache'/et/'survival_aggregated.npz') as m:
            oldq=m['historical_qbins'];oldgroups=[oldq[30:36].sum(axis=0),oldq[36:40].sum(axis=0)]
        oldc=np.zeros((2,5,121));
        for g in range(2):oldc[g,0]=oldgroups[g][0];oldc[g,2]=oldgroups[g][1]
        of=estimates(oldc)
        steps=[]
        for stage in range(8):
            if stage<=1:
                hh=historic[historic.Event.eq(et)&historic.Analysis.eq('HISTORICAL_ALGORITHM')&historic.Severity_Class.isin(['High_Q75_90','Extreme_Q90_100'])].copy()
                hh=hh.set_index('Severity_Class').loc[['High_Q75_90','Extreme_Q90_100']]
                values={'P120':hh.P120.to_numpy(),'RMPT120':hh['RMPT120_Historical_Trapezoid' if stage==0 else 'RMPT120'].to_numpy()}
            elif stage==2:values=of
            else:values=estimates(cs[20,VARIANTS[[0,1,2,3,4][stage-3]]].sum(axis=0))
            names=['HISTORICAL_B2_OLD_CUTS_TRAPEZOID','HISTORICAL_B2_OLD_CUTS_STEP','HISTORICAL_B2_FULL_ECDF_STEP','M1_CACHED_FULL_ECDF_STEP','PEAK_ONLY_ACTIVE_NEXT_EXCLUSIVE','PEAK_CATALOG_NEXT_EXCLUSIVE','PRIMARY_POSTPEAK_CATALOG_INCLUSIVE','FIRST_MISSING_GAP_SENSITIVITY']
            causes=['BASELINE','RMPT_INTEGRATION_ONLY','ECDF_GROUPING_ONLY','M1_TRUNCATION_NEXT_LOOKUP_AND_MISSING_HANDLING_COMBINED','POST_PEAK_REQUIREMENT_ONLY','NEW_CATALOGUE_ONSET_VS_ACTIVE_PHASE','INCLUSIVE_120_AND_RECORD_END_OBSERVATION_BOUNDARY','FIRST_MISSING_PHASE_CENSORING_SENSITIVITY']
            for metric in ['P120','RMPT120']:
                v=values[metric]
                for g,label in [(0,'High'),(1,'Extreme'),(-1,'Extreme_minus_High')]:
                    val=float(v[g]) if g>=0 else float(v[1]-v[0]);previous=next((r['Estimate'] for r in reversed(steps) if r['Metric']==metric and r['Group']==label),np.nan)
                    row=dict(Event=et,Step=stage,Analysis=names[stage],Group=label,Metric=metric,Estimate=val,Change_From_Previous=val-previous,Change_Component=causes[stage],ECDF_Grouping='HISTORICAL_L54B_SAMPLE_CUTS' if stage<2 else 'FULL_FINITE_HAZARD_ECDF',Followup_Origin='CLIMATE_TERMINATION; UNCHANGED_FROM_FORMAL_B2',Recovery_Peak_Rule='NOT_ENFORCED' if stage<4 else 'STRICTLY_AFTER_STORED_PEAK',Integration_Rule='DAILY_TRAPEZOID' if stage==0 else 'RIGHT_CONTINUOUS_STEP',Attribution_Limit='Ordered bridge, not causal decomposition; within M1 truncation vs missing subcomponents UNKNOWN without additional factorial reconstruction' if stage==3 else 'KNOWN_ORDERED_RULE_CHANGE')
                    steps.append(row)
        bridge.extend(steps)
        np.savez_compressed(OUT/'cache'/f'{et}_Supplemental_Subset_Counts.npz',**{f'{sc}_{name}':c for (sc,name),c in cs.items() if name in [UNC,COMP]})
        print(et,'ANALYSIS COMPLETE',flush=True)
    G=pd.DataFrame(group);U=pd.DataFrame(spatial)
    # Keep one row per group/analysis and CI columns for each scale.
    for i,r in G.iterrows():
        for sc in [20,50]:
            for metric,target in [('P120','P120'),('RMPT120','RMPT120'),('AJ_Recovery120','AJ_Recovery120')]:
                d=U[U.Event.eq(r.Event)&U.Analysis.eq(r.Analysis)&U.Metric.eq(metric)&U.Comparison.eq(r.Severity_Group)&U.Block_km.eq(sc)]
                if len(d):
                    G.loc[i,f'{target}_CI_Lower_{sc}km']=d.iloc[0].CI_Lower;G.loc[i,f'{target}_CI_Upper_{sc}km']=d.iloc[0].CI_Upper
    G.to_csv(OUT/'M5_Recovery_Group_Statistics.csv',index=False,encoding='utf-8-sig')
    U.to_csv(OUT/'M5_Spatial_Recovery_Uncertainty.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(comparisons).to_csv(OUT/'M5_KM_CompetingRisk_Comparison.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(bridge).to_csv(OUT/'M5_Historical_vs_Corrected_Recovery.csv',index=False,encoding='utf-8-sig')
    for filename,data in [('KM_AJ_Daily_Curves.csv',curves),('Logrank_High_Extreme.csv',logs),('Censoring_By_Year.csv',byyear),('Censoring_By_Block.csv',byblock),('Bootstrap_Stability.csv',stability),('Followup_Feature_Counts.csv',features),('Bootstrap_Contrast_Replicates.csv',replicates)]:pd.DataFrame(data).to_csv(OUT/'audit'/filename,index=False,encoding='utf-8-sig')
    print('ALL M5 STATISTICS READY',flush=True)
if __name__=='__main__':main()
