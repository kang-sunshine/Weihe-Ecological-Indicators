# -*- coding: utf-8 -*-
"""Saved Figure9 data extraction and fixed-model spatial bootstrap; NO MODEL FITS."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import json,hashlib,sys,datetime,zipfile,shutil
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.model_selection import GroupKFold
from pyproj import Transformer
R=Path(r'.');O=R/'EI_Figure9_R1';DB=R/'Weihe database V3';P=R/'analysis_project'
M3=R/'EI_Reviewer1_Major3_Revision';M4=R/'EI_Reviewer1_Major4_Revision';M8=R/'EI_Reviewer1_Major8_Revision'
STEM=DB/'Figure9_FINAL_V5_V2Residual/Figure9_ConditionalGreeningBenefit_FINAL_V5'
CHECK=[]
def ck(name,v,detail=''):
    CHECK.append(dict(Check=name,Pass=bool(v),Detail=str(detail)))
    if not v:
        pd.DataFrame(CHECK).to_csv(O/'qc/F9_Verification_Checks.csv',index=False)
        (O/'qc/F9_VERSION_CONFLICT_STOP.txt').write_text(name+'\n'+str(detail)+'\nSTOP: no historical results replaced.',encoding='utf8')
        raise AssertionError((name,detail))
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for a in iter(lambda:f.read(8*1024*1024),b''):h.update(a)
    return h.hexdigest()
def save(data,name):pd.DataFrame(data).to_csv(O/name,index=False,encoding='utf-8-sig')
def read(p,**kw):return pd.read_csv(p,float_precision='round_trip',**kw)
def snapshot():
    roots=[R/n for n in ['Weihe database','Weihe database V2','Weihe database V3','analysis_project','weihe/Weihe']]
    roots += [p for p in R.glob('EI_*') if p.is_dir() and p!=O]
    result=[]
    for root in roots:
        for p in root.rglob('*'):
            if p.is_file():
                st=p.stat();result.append(dict(Path=str(p),Size=st.st_size,Mtime_ns=st.st_mtime_ns))
    return pd.DataFrame(result).sort_values('Path').reset_index(drop=True)
def gz(frame,path):frame.to_csv(O/path,index=False,compression={'method':'gzip','compresslevel':6,'mtime':0})
def main():
    planpath=O/'qc/F9_C_PeakContrast_Plan.json';plan=json.loads(planpath.read_text());planhash=sha(planpath)
    ck('no_prior_output_replacement',not (O/'EI_Figure9_R1_DataPackage.zip').exists())
    before=snapshot();save(before,'qc/F9_Protected_Files_Before.csv')
    paths=[Path(str(STEM)+'_PixelResults.csv.gz'),Path(str(STEM)+'_HCL_BinnedBenefit.csv'),Path(str(STEM)+'_QC.txt'),
        P/'绘图代码/greening_model.py',P/'analysis_code/greening_model.py',M4/'paired/Frozen_Model_Pixels.parquet',M4/'cache/Historical_OOF.npy',
        M4/'oof/V0_LEGACY_LEGACY_Acomp_original_FIXED.parquet',M4/'splits/V0_LEGACY_LEGACY.npz',M4/'splits/V2_O0.npz',M4/'code/configuration.json',M4/'audit/Grid_Metadata.json',M4/'audit/Historical_Reproduction.json',M4/'M4_Spatial_CV_Comparison.csv',
        M8/'paired/Full_Model_Greening_Contrasts.parquet',M8/'cache/M8B/Spatial_Grid.npz',M8/'code/m8b_events.py',M8/'code/m8b_greening.py',
        M3/'M3_Greening_Contrast_Rule_Audit.md',R/'EI_Figure7_Redraw_DataPack/geometry/Basin_Boundary.geojson',
        R/'EI_Figure8_R1/data/F8_QC.txt']
    inputs=[]
    for p in paths:
        ck('input_exists_'+p.name,p.is_file(),p);st=p.stat()
        inputs.append(dict(Path=str(p),Size=st.st_size,Mtime_ns=st.st_mtime_ns,Modified_UTC=datetime.datetime.fromtimestamp(st.st_mtime,datetime.timezone.utc).isoformat(),SHA256=sha(p)))
    save(inputs,'qc/F9_Input_Manifest.csv')
    d=read(paths[0]);curve=read(paths[1]);frozen=pd.read_parquet(M4/'paired/Frozen_Model_Pixels.parquet');n=len(d)
    ck('formal_N153932',n==153932 and len(frozen)==n)
    ck('unique_valid_grid_pixel',not d.duplicated(['row','col']).any() and d.row.between(0,413).all() and d.col.between(0,702).all())
    ck('source_row_order_matches_frozen',np.array_equal(d[['row','col']],frozen[['row','col']]))
    ck('source_target_matches_M4',np.allclose(d.Residual_damage,frozen.Residual_damage,atol=1e-12,rtol=0))
    ck('all_formal_values_finite',np.isfinite(d.select_dtypes('number')).all().all())
    ck('historical_groups461',d.block20.nunique()==461 and np.array_equal(d.block20,frozen.block20))
    ck('historical_blockID_formula',np.array_equal(d.block20,(d.row//20)*1000+d.col//20))
    hist=np.load(M4/'cache/Historical_OOF.npy');old_oof=pd.read_parquet(M4/'oof/V0_LEGACY_LEGACY_Acomp_original_FIXED.parquet');split=np.load(M4/'splits/V0_LEGACY_LEGACY.npz');folds=split['Fold']
    ck('OOF_source_M4_matches',np.allclose(d.OOF_pred,hist,atol=1e-12,rtol=0) and np.allclose(d.OOF_pred,old_oof.OOF,atol=1e-12,rtol=0))
    ck('historical_folds_match_stored',np.array_equal(folds,frozen.Formal_Fold) and np.array_equal(folds,old_oof.Fold) and np.array_equal(split['Group'],d.block20))
    recompute=np.zeros(n,np.uint8)
    for k,(tr,te) in enumerate(GroupKFold(5).split(np.zeros((n,1)),groups=d.block20),1):
        recompute[te]=k;ck(f'fold{k}_no_group_overlap',not np.intersect1d(d.block20.to_numpy()[tr],d.block20.to_numpy()[te]).size)
    ck('fold_assignment_reproduced_without_fit',np.array_equal(folds,recompute))
    gridmeta=json.loads((M4/'audit/Grid_Metadata.json').read_text());xx=np.array(gridmeta['Axes']['x']);yy=np.array(gridmeta['Axes']['y'])
    lon=xx[d.col.to_numpy(int)];lat=yy[d.row.to_numpy(int)]
    ck('coordinates_exact_grid',np.array_equal(lon,frozen.Longitude) and np.array_equal(lat,frozen.Latitude))
    pixelid=d.row.to_numpy(np.int32)*703+d.col.to_numpy(np.int32)
    a=pd.DataFrame(dict(Pixel_ID=pixelid,Original_Source_Order=frozen.Original_Source_Order,row=d.row,col=d.col,Longitude=lon,Latitude=lat,Observed_Residual_Damage=d.Residual_damage,OOF_Prediction=d.OOF_pred,Fold=folds,Historical_Block20=d.block20,Valid_Mask=True))
    gz(a,'data/F9_A_OOF.csv.gz')
    metrics=[]
    ref=read(M4/'M4_Spatial_CV_Comparison.csv')
    for fold in [0,1,2,3,4,5]:
        mask=np.ones(n,bool) if fold==0 else folds==fold;y=d.Residual_damage.to_numpy()[mask];p=d.OOF_pred.to_numpy()[mask]
        rr=dict(Analysis='HISTORICAL_20x20_GROUPKFold_FIXED_RESPONSE',Scope='POOLED' if fold==0 else 'FOLD',Fold=fold,N_Test=len(y),N_Train=np.nan if fold==0 else n-len(y),N_Test_Groups=d.loc[mask,'block20'].nunique(),N_Train_Groups=np.nan if fold==0 else d.loc[~mask,'block20'].nunique(),R2=r2_score(y,p),RMSE=np.sqrt(mean_squared_error(y,p)),MAE=mean_absolute_error(y,p),Response_SD=np.std(y),OOF_Coverage=1.,Buffer_km=0,Block_Definition='historical20x20 analysis pixels, not metric20km',Source=str(paths[0]))
        metrics.append(rr)
        if fold==0:
            ck('historical_R2_recalculated',abs(rr['R2']-.3930276528778419)<1e-12)
            old=ref[(ref.Configuration=='V0_LEGACY_LEGACY')&(ref.Response=='Acomp_original')&(ref.Layer=='FIXED_HISTORICAL_RESPONSE')].iloc[0]
            ck('historical_metrics_all_match_M4',max(abs(rr[k]-old[k]) for k in ['R2','RMSE','MAE'])<1e-12)
    save(metrics,'data/F9_A_Metrics.csv')
    # Saved eight-feature full model output, never load or fit estimator weights.
    oldb=pd.read_parquet(M8/'paired/Full_Model_Greening_Contrasts.parquet')
    ck('B_source_matches_M8_T0',np.array_equal(d[['row','col']],oldb[['row','col']]) and np.allclose(d.GreeningBenefit,oldb.T0_Greening_Contrast,atol=1e-12,rtol=0))
    g20,g80=np.quantile(d.NIRv_trend,[.2,.8]);ck('P20_P80_formal',np.allclose([g20,g80],[.013746835,.031125259],atol=1e-12,rtol=0))
    zv=(d.VPD_base-d.VPD_base.mean())/d.VPD_base.std(ddof=0);zs=(d.RootSM_base-d.RootSM_base.mean())/d.RootSM_base.std(ddof=0);h=zv-zs
    ck('HCL_source_formula_reproduces',np.allclose(h,d.HCL,atol=1e-12,rtol=0),float(np.max(abs(h-d.HCL))))
    bins,edges=pd.qcut(d.HCL,15,labels=False,duplicates='drop',retbins=True);bins=bins.to_numpy(np.int16)+1
    ck('predeclared_HCL_intervals_unchanged',np.array_equal(edges,np.array(plan['All_HCL_Edges'])))
    ck('predeclared_original_curve_unchanged',sha(paths[1])==plan['Curve_SHA256'])
    sp=np.load(M8/'cache/M8B/Spatial_Grid.npz');blocks=sp['Block50_Index'][d.row.to_numpy(int),d.col.to_numpy(int)];B=len(sp['Block50_Labels']);cfg=json.loads((M4/'code/configuration.json').read_text())
    xm,ym=Transformer.from_crs(4326,cfg['Projection_Proj4'],always_xy=True).transform(lon,lat)
    bi=np.floor(xm/50000).astype(int);bj=np.floor(ym/50000).astype(int);labels=bi*10000+bj
    ck('existing_50km_grid_true_metric',np.array_equal(labels,sp['Block50_Labels'][blocks]) and (blocks>=0).all())
    cv50=np.load(M4/'splits/V2_O0.npz');_,tmp=np.unique(labels,return_inverse=True);ck('M8_and_M4_50km_partition_match',np.array_equal(tmp,cv50['Group']))
    benefit=d.GreeningBenefit.to_numpy();pos=benefit>1e-12;neg=benefit< -1e-12;zero=abs(benefit)<=1e-12;domain=np.zeros(n,np.int8);domain[(d.HCL<=0)&pos]=1;domain[(d.HCL>0)&pos]=2;domain[(d.HCL<=0)&neg]=3;domain[(d.HCL>0)&neg]=4
    ck('original_five_domains_exact',np.array_equal(domain,d.Domain))
    ck('B_sign_categories_exact',tuple(map(int,[pos.sum(),neg.sum(),zero.sum()]))==(71781,17583,64568))
    maps=pd.DataFrame(dict(Pixel_ID=pixelid,row=d.row,col=d.col,Longitude=lon,Latitude=lat,B=benefit,HCL=d.HCL,HCL_bin=bins,Valid_Mask=True,Support_Mask=True,Historical_Block20=d.block20,Block50_Index=blocks,Block50_Label=labels,Block50_i=bi,Block50_j=bj,B_Sign_Class=np.where(pos,'Positive',np.where(neg,'Negative','Numerical zero')),Original_Domain=domain))
    gz(maps,'data/F9_BD_PixelMaps.csv.gz')
    ck('A_BD_identical_ID_coordinates',np.array_equal(a[['Pixel_ID','row','col','Longitude','Latitude']],maps[['Pixel_ID','row','col','Longitude','Latitude']]))
    # Original15-bin SE interval, independently verified, preserved separately from newbootstrap.
    curve=curve.sort_values('bin').reset_index(drop=True)
    ck('original15_grid_bins',np.array_equal(curve.bin,np.arange(1,16)))
    for j,row in curve.iterrows():
        mask=bins==row.bin;sub=d.loc[mask];bm=sub.groupby('block20').GreeningBenefit.mean();se=bm.std(ddof=1)/np.sqrt(len(bm))
        ck(f'original_HCL_bin{j+1}_values_counts',abs(sub.HCL.mean()-row.HCL)<1e-12 and abs(sub.GreeningBenefit.mean()-row.Benefit)<1e-12 and abs(se-row.Benefit_SE_block)<1e-12 and len(sub)==row.n_pixels and len(bm)==row.n_blocks)
    curve['CI_lower']=curve.Benefit-1.96*curve.Benefit_SE_block;curve['CI_upper']=curve.Benefit+1.96*curve.Benefit_SE_block
    curve['HCL_lower']=edges[:-1];curve['HCL_upper']=edges[1:];curve['Original_CI_Method']='pixel mean +/-1.96*SE of historical20x20 block means; NOT bootstrap'
    curve['n_blocks50']=[len(np.unique(blocks[bins==j])) for j in range(1,16)]
    flat=blocks*15+bins-1;counts=np.bincount(flat,minlength=B*15).reshape(B,15);sums=np.bincount(flat,weights=benefit,minlength=B*15).reshape(B,15)
    ck('block_counts_close_to_original_bins',np.array_equal(counts.sum(axis=0),curve.n_pixels))
    ck('peak_selection_exploratory',int(curve.loc[curve.Benefit.idxmax(),'bin'])==plan['Peak_Bin'])
    peak,left,right=plan['Peak_Bin']-1,plan['Left_Bin']-1,plan['Right_Bin']-1
    common=np.flatnonzero((counts[:,[left,peak,right]]>0).all(axis=1));ck('common_spatial_support_exists',len(common)>=plan['Minimum_Comparison_Blocks'],len(common))
    # Plan was written before bootstrap; do not revise intervals/support to chase significance.
    ck('plan_hash_locked_before_bootstrap',sha(planpath)==planhash)
    (O/'qc/F9_C_Bootstrap_Run_Start.json').write_text(json.dumps(dict(Started_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),Plan_SHA256=planhash,No_Prior_Test_Results=True),indent=2),encoding='utf8')
    contrastrows=[];replicates=[];draws={}
    for scope,select in [('FULL_ORIGINAL_BIN_COHORTS_SHARED_BLOCK_WEIGHTS',np.arange(B)),('COMMON_BLOCKS_THREE_BINS',common)]:
        cb=counts[select];sb=sums[select];bb=len(select);W=np.random.default_rng(plan['Seed']).multinomial(bb,np.full(bb,1/bb),size=plan['Bootstrap_N']);draws[scope+'_weights']=W;draws[scope+'_block_indices']=select
        den=W@cb;num=W@sb;means=np.divide(num,den,out=np.full_like(num,np.nan,dtype=float),where=den>0)
        pt=np.divide(sb.sum(axis=0),cb.sum(axis=0),out=np.full(15,np.nan),where=cb.sum(axis=0)>0)
        if scope.startswith('FULL'):
            ck('full_cohort_bootstrap_points_match_curve',np.allclose(pt,curve.Benefit,atol=1e-12,rtol=0))
            for j in range(15):
                valid=np.isfinite(means[:,j]);ci=np.quantile(means[valid,j],[.025,.975]) if valid.sum()>=plan['Minimum_Valid_Replicates'] else [np.nan,np.nan]
                curve.loc[j,'Bootstrap50_CI_lower']=ci[0];curve.loc[j,'Bootstrap50_CI_upper']=ci[1];curve.loc[j,'Bootstrap50_valid_n']=int(valid.sum());curve.loc[j,'Bootstrap50_failed_n']=int((~valid).sum())
        for name,base in [('peak-minus-left',means[:,left]),('peak-minus-right',means[:,right]),('peak-minus-two-neighbor-reference',.5*(means[:,left]+means[:,right]))]:
            basepoint=pt[left] if name=='peak-minus-left' else pt[right] if name=='peak-minus-right' else .5*(pt[left]+pt[right]);value=float(pt[peak]-basepoint);v=means[:,peak]-base;valid=np.isfinite(v);nv=int(valid.sum());ci=np.quantile(v[valid],[.025,.975]) if nv>=plan['Minimum_Valid_Replicates'] else [np.nan,np.nan]
            status='NOT_ESTIMABLE' if not np.isfinite(ci).all() else 'NOT_SUPPORTED' if ci[0]<=0<=ci[1] else 'EXPLORATORY_SUPPORTED_POSITIVE' if ci[0]>0 else 'EXPLORATORY_SUPPORTED_NEGATIVE'
            record=dict(Comparison=name,Support_Domain=scope,Estimate=value,CI_lower=ci[0],CI_upper=ci[1],Unit='Acomp residual damage',Status=status,Peak_bin=peak+1,Left_bin=left+1,Right_bin=right+1,Peak_HCL=float(curve.HCL.iloc[peak]),Left_HCL=float(curve.HCL.iloc[left]),Right_HCL=float(curve.HCL.iloc[right]),Peak_B=pt[peak],Left_B=pt[left],Right_B=pt[right],N_peak_pixels=int(cb[:,peak].sum()),N_left_pixels=int(cb[:,left].sum()),N_right_pixels=int(cb[:,right].sum()),N_peak_blocks=int((cb[:,peak]>0).sum()),N_left_blocks=int((cb[:,left]>0).sum()),N_right_blocks=int((cb[:,right]>0).sum()),N_common_all3_blocks=len(common),N_draw_domain_blocks=bb,N_occupied_draw_domain_blocks=int((cb.sum(axis=1)>0).sum()),Bootstrap_N=len(W),Bootstrap_valid_n=nv,Bootstrap_failed_n=len(W)-nv,Block_km=50,Seed=plan['Seed'],Exploratory_peak_selection=True,Selection_adjusted=False,Inference='fixed model, fixed HCL bins, shared whole-block draws; not causal or peak-selection adjusted')
            contrastrows.append(record)
            for rep,(val,ok) in enumerate(zip(v,valid)):replicates.append(dict(Support_Domain=scope,Comparison=name,Replicate=rep,Estimate=val,Valid=bool(ok)))
        print('BOOTSTRAP',scope,'drawblocks',bb,'occupied',int((cb.sum(axis=1)>0).sum()),flush=True)
    np.savez_compressed(O/'qc/F9_C_Bootstrap_Draws.npz',**draws,Block_Bin_Counts=counts,Block_Bin_Sums=sums,Block50_Labels=sp['Block50_Labels'])
    save(curve,'data/F9_C_HCL_Curve.csv');save(contrastrows,'data/F9_C_PeakContrast.csv');gz(pd.DataFrame(replicates),'data/F9_C_Bootstrap_Replicates.csv.gz')
    # Replay ONLY the existing display arithmetic as an explicitly untested display reference.
    gx=np.linspace(curve.HCL.min(),curve.HCL.max(),200);gy=make_interp_spline(curve.HCL,curve.Benefit,k=3)(gx);gse=make_interp_spline(curve.HCL,curve.Benefit_SE_block,k=3)(gx)
    save(pd.DataFrame(dict(HCL=gx,Original_Display_B=gy,Original_Display_CI_lower=gy-1.96*gse,Original_Display_CI_upper=gy+1.96*gse,Method='exact replay of existing cubic spline display; not independent HCL estimates or bootstrap CI')),'data/F9_C_OriginalDisplayReference.csv')
    # One compact shared boundary and grid-coordinate reference, no raster/model copying.
    boundary=R/'EI_Figure7_Redraw_DataPack/geometry/Basin_Boundary.geojson';shutil.copyfile(boundary,O/'map_assets/Basin_Boundary.geojson')
    georef=dict(CRS='EPSG:4326',Grid_shape=[414,703],X_centres=xx.tolist(),Y_centres=yy.tolist(),X_step=float(np.median(np.diff(xx))),Y_step=float(np.median(np.diff(yy))),Source=str(M4/'audit/Grid_Metadata.json'),Original_Grid_Source=str(DB/'Weihe_L3_ClimateStress_1000m_2000_2024_V3_1_FINAL.nc'),Original_Basin_Source=str(R/'weihe/Weihe/Export_Output_Weihe.shp'),Boundary_Shared_Reference=str(boundary),Boundary_SHA256=sha(boundary),Source_plot_extent_uses_centres=[float(xx.min()),float(xx.max()),float(yy.min()),float(yy.max())],Missing_grid_cells='not in153932-pixel table; must be masked, never classed as numerical zero',Bootstrap_Projection=cfg['Projection_Proj4'],Bootstrap_Origin_m=[0,0],Bootstrap_Block_m=50000,Historical_Block20='(row//20)*1000+col//20, nominal ~20km; not this projected50km grid')
    (O/'map_assets/F9_Grid_Reference.json').write_text(json.dumps(georef,indent=2),encoding='utf8')
    # Preserve explicit conflicts rather than replacing either original CI or display curve.
    save([dict(Item='PanelC_original_interval_label',Figure_Script=str(P/'绘图代码/greening_model.py'),Observed='label says95% Block bootstrap CI; executed formula is cubic-spline interpolation of SE and mean +/-1.96*SE',Action='LABEL_CONFLICT_RECORDED; original CI values retained; new50km bootstrap in separate columns/files; no historical replacement',Status='PARTIAL'),dict(Item='PanelC_curve_display_version',Figure_Script=str(P/'绘图代码/greening_model.py'),Observed='formalV5 source has15 empirical bins; current layout draws200-point cubic spline from those bins',Action='original15-bin data exported; existing display arithmetic replay separately labelled; no peak test on spline maxima',Status='RECORDED')],'qc/F9_Source_Conflicts.csv')
    ck('no_plan_changed_after_inference',sha(planpath)==planhash)
    ck('all_original_sources_unchanged',all(sha(Path(v['Path']))==v['SHA256'] for v in inputs))
    after=snapshot();ck('all_protected_files_metadata_unchanged',before.equals(after),len(before));save(after,'qc/F9_Protected_Files_After.csv')
    ck('all_bootstrap_weight_sums',all(np.all(v.sum(axis=1)==v.shape[1]) for k,v in draws.items() if k.endswith('_weights')))
    ck('six_contrasts_with_real_results',len(contrastrows)==6 and all(r['Bootstrap_valid_n']+r['Bootstrap_failed_n']==1000 for r in contrastrows))
    save(CHECK,'qc/F9_Verification_Checks.csv')
    stats=dict(N=n,N_Historical_Groups=461,N_Bootstrap_Domain_Blocks=B,N_Bootstrap_Model_Occupied_Blocks=len(np.unique(blocks)),N_Common_PeakNeighbor_Blocks=len(common),G20=float(g20),G80=float(g80),VPD_mean=float(d.VPD_base.mean()),VPD_sd_ddof0=float(d.VPD_base.std(ddof=0)),RootSM_mean=float(d.RootSM_base.mean()),RootSM_sd_ddof0=float(d.RootSM_base.std(ddof=0)),QC_Checks=len(CHECK),Protected_Files=len(before),Key_Inputs=len(inputs),Plan_SHA256=planhash,Curve_Method_Conflict=True,New_Model_Fits=0,New_Spatial_CV_Runs=0,Status='PARTIAL')
    (O/'qc/F9_Extraction_Result.json').write_text(json.dumps(stats,indent=2),encoding='utf8')
    print('FIGURE9 EXTRACTION READY',len(CHECK),'checks',flush=True)
    print(pd.DataFrame(contrastrows)[['Comparison','Support_Domain','Estimate','CI_lower','CI_upper','Status','Bootstrap_valid_n']].to_string(index=False),flush=True)
if __name__=='__main__':main()
