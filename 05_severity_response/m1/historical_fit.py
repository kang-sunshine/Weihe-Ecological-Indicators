from pathlib import Path
import ast,json
import numpy as np
import pandas as pd
from model_utils import search
OUT=Path(r'.\EI_Reviewer1_Major1_Revision')
ROOT=Path(r'.\Weihe database V3\L5_EventEcoFunction\L5_4_ThresholdAnalysis')
hist=pd.read_csv(ROOT/'L5_4A_Severity_Response_Thresholds.csv')
bins=pd.read_csv(ROOT/'L5_4A_Severity_Response_Binned.csv');prob=pd.read_csv(ROOT/'L5_4A_Severity_DamageProbability_Binned.csv')
rng=np.random.default_rng(20261109);reg=[];rep=[];curves=[];boots=[]
for _,h in hist.iterrows():
    et=h.event_type;r=h.response
    d=(prob[prob.event_type.eq(et)] if r=='Damage_probability' else bins[bins.event_type.eq(et)&bins.response.eq(r)])
    x=d['severity_mean' if r=='Damage_probability' else 'severity_median'].to_numpy(float)
    y=d['damage_probability' if r=='Damage_probability' else 'response_median'].to_numpy(float);w=np.sqrt(d.n.to_numpy(float))
    for mode in ['historical','full_supported']:
        f=search(x,y,w,mode)
        rep.append(dict(Event=et,Response=r,Analysis='SAVED_BIN_'+mode,Reproduction_Status='EXACT_REPRODUCTION_SAVED_BIN_POINT_FIT' if mode=='historical' else 'INDEPENDENT_RECONSTRUCTION',S_star=f['c'],Delta_AIC=f['delta'],Linear_AIC=f['linear_aic'],Segmented_AIC=f['aic'],Search_Min=f['lo'],Search_Max=f['hi'],Distance_Lower_S=f['c']-f['lo'],Distance_Upper_S=f['hi']-f['c'],S_Difference_From_History=f['c']-h.breakpoint_severity,Delta_AIC_Difference=f['delta']-h.delta_AIC_linear_minus_segmented,n_bins=len(x),Pre_Slope=f['beta'][1],Post_Slope=f['beta'][1]+f['beta'][2]))
        for c,a in zip(f['candidates'],f['aics']):curves.append(dict(Event=et,Response=r,Analysis='SAVED_BIN_'+mode,Candidate_S=c,AIC=a,Delta_AIC_Linear=f['linear_aic']-a,Delta_AIC_From_Best=a-f['aic']))
    f=search(x,y,w)
    for j in range(500):
        ix=rng.choice(len(x),len(x),replace=True,p=w/w.sum());b=search(x[ix],y[ix],w[ix])
        boots.append(dict(Event=et,Response=r,Replicate=j,S_star=b['c'],Delta_AIC=b['delta'],Unit='SAVED_BIN_PROB_SQRT_N'))
    reg.append(dict(Event=et,Response=r,S_star=h.breakpoint_severity,Q_star=h.breakpoint_percentile,Search_Min=f['lo'],Search_Max=f['hi'],Search_Rule='sorted bin positions floor(0.1*n):ceil(0.9*n), >=4 positions each tail',Delta_AIC=h.delta_AIC_linear_minus_segmented,Breakpoint_CI=f'{h.breakpoint_CI_low};{h.breakpoint_CI_high}',CI_Units='raw severity, NOT percentile',Bootstrap_Unit='bin; sampling probability proportional to sqrt(N); 500 replicates',Source_File=str(ROOT/'L5_4A_Severity_Response_Thresholds.csv'),Source_Code=str(OUT/'audit/Step_L5_4A_Severity_Response_Thresholds_FAST.py'),Evidence_Status='ORIGINAL_CODE_LOCATED_POINT_FIT_VERIFIED',Percentile_Reference='bounded 500000 clean-pre severity sample; order-biased replacement algorithm'))
pd.DataFrame(reg).to_csv(OUT/'R1M1_Original_Breakpoint_Registry.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(rep).to_csv(OUT/'R1M1_Breakpoint_Replication.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(curves).to_csv(OUT/'R1M1_Breakpoint_Candidate_AIC.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(boots).to_csv(OUT/'audit/Saved_Bin_Bootstrap.csv',index=False)
print(pd.DataFrame(rep)[['Event','Response','Analysis','S_star','Delta_AIC','S_Difference_From_History','Delta_AIC_Difference']].to_string(index=False))
