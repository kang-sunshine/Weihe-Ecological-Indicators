# -*- coding: utf-8 -*-
"""
c | Severity-adjusted residual damage + Figure 8 source data

Key correction relative to V1:
The CDH damage breakpoint S* is still inherited unchanged from locked Section 3.3,
but the nuisance severity-damage coefficients are estimated directly from the
formal EVENT cohort at that fixed S*. Therefore d_res is a genuine event-level
OLS residual around the fixed-break conditional mean, rather than a residual
around a regression fitted to binned medians.

Scientific chain:
3.1 CDH severity -> 3.2 A_comp -> 3.3 Acomp_damage + locked S*
-> fixed-S* event-level nuisance regression -> event residual
-> pixel aggregation -> Figure 8.
"""
import os, glob, gzip, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import xarray as xr

ROOT=r".\Weihe database V3"
L5=os.path.join(ROOT,"L5_EventEcoFunction")
EVENT_FILE=os.path.join(L5,"L5_3_SeverityResilience","L5_3_CDH_Event_Severity_Resilience_Metrics.csv.gz")
THRESHOLD_FILE=os.path.join(L5,"L5_4_ThresholdAnalysis","L5_4A_Severity_Response_Thresholds.csv")

OUT=os.path.join(L5,"L6_SeverityAdjustedResidual_FINAL_V2")
FIG8=os.path.join(L5,"Figures_FINAL","Figure08_Greening_Hydroclimate_Resilience_FINAL_V2")
os.makedirs(OUT,exist_ok=True); os.makedirs(FIG8,exist_ok=True)

EVENT_OUT=os.path.join(OUT,"L6_0_CDH_Event_SeverityAdjusted_DamageResidual_FINAL_V2.csv.gz")
PIXEL_OUT=os.path.join(OUT,"L6_0_CDH_Pixel_ResidualDamage_GreeningHydroclimate_FINAL_V2.csv.gz")
MODEL_OUT=os.path.join(OUT,"L6_0_CDH_EventLevel_FixedBreakpoint_Model_FINAL_V2.csv")
QC_OUT=os.path.join(OUT,"L6_0_CDH_ResidualDamage_FINAL_V2_QC.txt")
FG=os.path.join(FIG8,"Figure08_GreeningGradient_SourceData.csv")
FB=os.path.join(FIG8,"Figure08_GreeningBenefit_SourceData.csv")
FS=os.path.join(FIG8,"Figure08_ResponseSurface_SourceData.csv")

NY,NX=414,703
CHUNK=750_000
MIN_EVENTS_PER_PIXEL=3
BLOCK_PIX=20

def banner(s):
    print("\n"+"="*112); print(s); print("="*112,flush=True)

def formal_arrays(ch):
    sev=pd.to_numeric(ch.climate_cumulative_severity,errors="coerce").to_numpy(float)
    dam=pd.to_numeric(ch.Acomp_damage,errors="coerce").to_numpy(float)
    row=pd.to_numeric(ch.row,errors="coerce").to_numpy(float)
    col=pd.to_numeric(ch.col,errors="coerce").to_numpy(float)
    clean=pd.to_numeric(ch.clean_pre_any_event,errors="coerce").to_numpy(float)
    valid=pd.to_numeric(ch.valid_resistance_sample,errors="coerce").to_numpy(float)
    ok=(clean==1)&(valid==1)&np.isfinite(sev)&np.isfinite(dam)&(dam>0)&np.isfinite(row)&np.isfinite(col)&(row>=0)&(row<NY)&(col>=0)&(col<NX)
    return ok,sev,dam,row,col

def design(sev,sstar):
    return np.column_stack([np.ones(len(sev)),sev,np.maximum(0.,sev-sstar)])

def decile(v,edges): return np.searchsorted(edges,v,side="right")+1

def block_se(z):
    x=z.groupby("block20",observed=True).Residual_damage.mean()
    return (float(x.std(ddof=1)/np.sqrt(len(x))) if len(x)>=2 else np.nan),int(len(x))

# 1 | inherit locked breakpoint
banner("1 | INHERIT LOCKED CDH DAMAGE-MAGNITUDE BREAKPOINT")
if not os.path.exists(EVENT_FILE): raise FileNotFoundError(EVENT_FILE)
thr=pd.read_csv(THRESHOLD_FILE)
r=thr[(thr.event_type=="CDH")&(thr.response=="Acomp_damage")]
if len(r)!=1: raise RuntimeError(f"Expected one CDH/Acomp_damage threshold row, got {len(r)}")
r=r.iloc[0]
SSTAR=float(r.breakpoint_severity); PCTL=float(r.breakpoint_percentile); EVIDENCE=str(r.threshold_evidence)
print(f"S*={SSTAR:.10f} | percentile={PCTL:.6f} | evidence={EVIDENCE}")

# 2 | first pass: exact event-level fixed-break OLS
banner("2 | PASS 1 — EVENT-LEVEL FIXED-BREAK REGRESSION")
cols=["event_id","row","col","climate_cumulative_severity","Acomp_damage",
      "clean_pre_any_event","valid_resistance_sample"]
XtX=np.zeros((3,3),float); Xty=np.zeros(3,float)
nfit=0; sy=sy2=0.0; total=0
for k,ch in enumerate(pd.read_csv(EVENT_FILE,compression="gzip",usecols=cols,chunksize=CHUNK,low_memory=False),1):
    total+=len(ch); ok,sev,dam,_,_=formal_arrays(ch)
    sev=sev[ok]; dam=dam[ok]
    if len(sev):
        X=design(sev,SSTAR)
        XtX += X.T@X; Xty += X.T@dam
        nfit += len(dam); sy += float(dam.sum()); sy2 += float(np.dot(dam,dam))
    print(f"pass1 chunk {k:02d} | total={total:,} | formal={nfit:,}",flush=True)

beta=np.linalg.solve(XtX,Xty)
B0,BPRE,BHINGE=[float(v) for v in beta]; BPOST=BPRE+BHINGE
# exact OLS diagnostics from sufficient statistics
sse=float(sy2-2*beta@Xty+beta@XtX@beta)
ybar=sy/nfit; sst=float(sy2-nfit*ybar*ybar)
r2=1-sse/sst
pd.DataFrame([{"event_type":"CDH","response":"Acomp_damage","model":"event_level_fixed_break_OLS",
"breakpoint_severity":SSTAR,"breakpoint_percentile":PCTL,"threshold_evidence":EVIDENCE,
"intercept":B0,"pre_break_slope":BPRE,"hinge_slope_change":BHINGE,
"post_break_slope":BPOST,"n_events":nfit,"r2":r2,"sse":sse}]).to_csv(MODEL_OUT,index=False)
print(f"Dhat(S)={B0:.10f}+{BPRE:.10f}S+{BHINGE:.10f}max(0,S-S*)")
print(f"post slope={BPOST:.10f} | R2={r2:.6f}")

# 3 | second pass: residuals + pixel aggregation
banner("3 | PASS 2 — RESIDUALS + PIXEL AGGREGATION")
ncell=NY*NX
cnt=np.zeros(ncell,np.int64); sm=np.zeros(ncell); ss=np.zeros(ncell)
first=True; total2=kept=0; res_sum=res_ss=0.0
with gzip.open(EVENT_OUT,"wt",encoding="utf-8",newline="",compresslevel=1) as gz:
    for k,ch in enumerate(pd.read_csv(EVENT_FILE,compression="gzip",usecols=cols,chunksize=CHUNK,low_memory=False),1):
        total2+=len(ch); ok,sev,dam,row,col=formal_arrays(ch)
        sev=sev[ok]; dam=dam[ok]; rr=row[ok].astype(np.int32); cc=col[ok].astype(np.int32)
        if not len(sev):
            print(f"pass2 chunk {k:02d} | total={total2:,} | kept={kept:,}",flush=True); continue
        expected=design(sev,SSTAR)@beta
        resid=dam-expected
        idx=rr*NX+cc
        cnt+=np.bincount(idx,minlength=ncell)
        sm +=np.bincount(idx,weights=resid,minlength=ncell)
        ss +=np.bincount(idx,weights=resid*resid,minlength=ncell)
        res_sum+=float(resid.sum()); res_ss+=float(np.dot(resid,resid))
        out=pd.DataFrame({"event_id":ch.loc[ok,"event_id"].to_numpy(),"row":rr,"col":cc,
            "climate_cumulative_severity":sev,"Acomp_damage":dam,
            "expected_Acomp_damage":expected,"damage_residual":resid})
        out.to_csv(gz,index=False,header=first); first=False; kept+=len(out)
        print(f"pass2 chunk {k:02d} | total={total2:,} | kept={kept:,}",flush=True)

shape=(NY,NX)
cnt2=cnt.reshape(shape); sum2=sm.reshape(shape); ss2=ss.reshape(shape)
mean=np.divide(sum2,cnt2,out=np.full(shape,np.nan),where=cnt2>0)
var=np.divide(ss2,cnt2,out=np.full(shape,np.nan),where=cnt2>0)-mean**2
sd=np.sqrt(np.where(var<0,0,var))
event_res_mean=res_sum/kept
event_res_sd=np.sqrt(max(res_ss/kept-event_res_mean**2,0))

# 4 | covariates
banner("4 | LOAD PIXEL GREENING / ROOTSM COVARIATES")
def find_covariates():
    hits=glob.glob(os.path.join(ROOT,"**","L6_2_PixelLevel_ResilienceDriverDataset.csv.gz"),recursive=True)
    if hits:
        fp=hits[0]; h=pd.read_csv(fp,compression="gzip",nrows=2)
        nt=next((c for c in ["NIRv_trend","NIRv_trend_per_decade"] if c in h.columns),None)
        rt=next((c for c in ["RootSM_base","RootSM_mean","RootSM"] if c in h.columns),None)
        if nt and rt and {"row","col"}.issubset(h.columns):
            q=pd.read_csv(fp,compression="gzip",usecols=["row","col",nt,rt])
            return q.rename(columns={nt:"NIRv_trend",rt:"RootSM_base"}),fp
    hits=glob.glob(os.path.join(ROOT,"**","*Greening*Hydroclimate*Proxy*Maps*.nc"),recursive=True)
    for fp in hits:
        try:
            ds=xr.open_dataset(fp)
            nt=next((c for c in ["NIRv_trend_per_decade","NIRv_trend"] if c in ds),None)
            rt=next((c for c in ["RootSM_base","RootSM_mean"] if c in ds),None)
            if nt and rt:
                a=np.asarray(ds[nt].values,float); r=np.asarray(ds[rt].values,float); ds.close()
                if a.shape==shape and r.shape==shape:
                    yy,xx=np.indices(shape)
                    return pd.DataFrame({"row":yy.ravel(),"col":xx.ravel(),"NIRv_trend":a.ravel(),"RootSM_base":r.ravel()}),fp
            ds.close()
        except Exception: pass
    raise FileNotFoundError("No usable pixel covariate source found.")

cov,cov_source=find_covariates()
cov["row"]=pd.to_numeric(cov.row,errors="coerce"); cov["col"]=pd.to_numeric(cov.col,errors="coerce")
cov=cov.dropna(subset=["row","col"]).copy(); cov["row"]=cov.row.astype(int); cov["col"]=cov.col.astype(int)
cov=cov[(cov.row>=0)&(cov.row<NY)&(cov.col>=0)&(cov.col<NX)].drop_duplicates(["row","col"])
yy,xx=np.indices(shape)
pix=pd.DataFrame({"row":yy.ravel(),"col":xx.ravel(),"Residual_damage":mean.ravel(),
                  "Residual_damage_SD":sd.ravel(),"n_events":cnt2.ravel()})
pix=pix.merge(cov,on=["row","col"],how="left")
pix=pix[(pix.n_events>=MIN_EVENTS_PER_PIXEL)&np.isfinite(pix.Residual_damage)&
        np.isfinite(pix.NIRv_trend)&np.isfinite(pix.RootSM_base)].copy()

# 5 | pixel quantiles
banner("5 | PIXEL-BASED MOISTURE / GREENING CLASSES")
g_edges=np.quantile(pix.NIRv_trend,np.arange(.1,1,.1))
m_edges=np.quantile(pix.RootSM_base,np.arange(.1,1,.1))
m_terc=np.quantile(pix.RootSM_base,[1/3,2/3])
pix["greening_decile"]=decile(pix.NIRv_trend.to_numpy(),g_edges)
pix["RootSM_decile"]=decile(pix.RootSM_base.to_numpy(),m_edges)
pix["moisture_regime"]=np.where(pix.RootSM_base<=m_terc[0],"Dry",
                         np.where(pix.RootSM_base<=m_terc[1],"Intermediate","Wet"))
pix["block20"]=(pix.row//BLOCK_PIX).astype(int)*1000+(pix.col//BLOCK_PIX).astype(int)
pix.to_csv(PIXEL_OUT,index=False,compression={"method":"gzip","compresslevel":1})

# 6 | source tables
banner("6 | BUILD FIGURE 8 SOURCE TABLES")
A=[]
for (reg,gdec),z in pix.groupby(["moisture_regime","greening_decile"],observed=True):
    se,nb=block_se(z)
    A.append({"moisture_regime":reg,"greening_decile":int(gdec),
        "damage_residual_mean":float(z.Residual_damage.mean()),"damage_residual_SE":se,
        "damage_n":int(len(z)),"spatial_block_n":nb})
A=pd.DataFrame(A).sort_values(["moisture_regime","greening_decile"]); A.to_csv(FG,index=False)

B=[]
for mdec,z in pix.groupby("RootSM_decile",observed=True):
    lo=z[z.greening_decile<=2]; hi=z[z.greening_decile>=9]
    ld=float(lo.Residual_damage.mean()) if len(lo) else np.nan
    hd=float(hi.Residual_damage.mean()) if len(hi) else np.nan
    lb=lo.groupby("block20").Residual_damage.mean(); hb=hi.groupby("block20").Residual_damage.mean()
    common=lb.index.intersection(hb.index); dif=(lb.loc[common]-hb.loc[common]).to_numpy(float) if len(common) else np.array([])
    se=float(dif.std(ddof=1)/np.sqrt(len(dif))) if len(dif)>=2 else np.nan
    B.append({"RootSM_decile":int(mdec),"low_greening_damage":ld,"high_greening_damage":hd,
        "greening_resilience_benefit":ld-hd,"greening_resilience_benefit_SE":se,
        "low_greening_n":int(len(lo)),"high_greening_n":int(len(hi)),
        "matched_spatial_block_n":int(len(common))})
B=pd.DataFrame(B).sort_values("RootSM_decile"); B.to_csv(FB,index=False)

C=(pix.groupby(["RootSM_decile","greening_decile"],observed=True).Residual_damage.agg(["mean","size"])
   .reset_index().rename(columns={"mean":"damage_residual","size":"n"}))
C.to_csv(FS,index=False)

# 7 | QC
banner("7 | QC")
pixel_mean=float(pix.Residual_damage.mean())
lines=["="*106,"STEP L6.0 FINAL V2 | EVENT-LEVEL FIXED-BREAK RESIDUAL + FIGURE 8 SOURCE DATA","="*106,
f"Event source                   : {EVENT_FILE}",f"Threshold source               : {THRESHOLD_FILE}",
f"Covariate source               : {cov_source}","",
"LINK TO LOCKED METHODS",
"3.1 input                      : CDH climate_cumulative_severity",
"3.3 input                      : event-level Acomp_damage",
f"Locked CDH damage breakpoint   : {SSTAR:.10f}",
f"Breakpoint percentile          : {PCTL:.6f}",
f"Threshold evidence             : {EVIDENCE}","",
"FIXED-BREAK EVENT-LEVEL OLS",
f"Intercept                      : {B0:+.10f}",
f"Pre-break slope                : {BPRE:+.10f}",
f"Hinge slope change             : {BHINGE:+.10f}",
f"Post-break slope               : {BPOST:+.10f}",
f"R2                             : {r2:.8f}",
f"Formal regression events       : {nfit:,}","",
"RESIDUAL QC",
f"Event residual mean            : {event_res_mean:+.12f}",
f"Event residual SD              : {event_res_sd:.10f}",
f"Pixels retained (>=3 events)   : {len(pix):,}",
f"Equal-pixel residual mean      : {pixel_mean:+.10f}","",
"FIGURE 8 CLASSIFICATION",
f"RootSM tercile 1               : {m_terc[0]:.10f}",
f"RootSM tercile 2               : {m_terc[1]:.10f}",
"Low greening                   : pixel greening deciles 1-2",
"High greening                  : pixel greening deciles 9-10",
"Uncertainty                    : SE across 20-km spatial-block means","",
f"Panel A rows                   : {len(A)}",f"Panel B rows                   : {len(B)}",
f"Panel C cells                  : {len(C)} / 100",
f"Positive benefit fraction      : {(B.greening_resilience_benefit>0).mean()*100:.2f}%",
f"Negative benefit fraction      : {(B.greening_resilience_benefit<0).mean()*100:.2f}%","",
"INTERPRETATION LOCKS",
"1. S* is inherited from locked Section 3.3; only nuisance coefficients are re-estimated at fixed S*.",
"2. d_res is an event-level OLS residual around the fixed-break conditional mean.",
"3. Figure 8 uses equal-weight retained pixels, not event-weighted groups.",
"4. NIRv trend is an associational greening proxy, not a causal treatment.",
"5. Deciles/terciles are empirical classes, not physical thresholds.",
"6. Continuous thresholds remain reserved for ALE analysis.","",
f"Model output                   : {MODEL_OUT}",f"Event residuals                : {EVENT_OUT}",
f"Pixel residuals                : {PIXEL_OUT}",f"Figure 8 A                     : {FG}",
f"Figure 8 B                     : {FB}",f"Figure 8 C                     : {FS}",
"FINAL STATUS: COMPLETE"]
with open(QC_OUT,"w",encoding="utf-8") as f:f.write("\n".join(lines))
print("\n".join(lines))
