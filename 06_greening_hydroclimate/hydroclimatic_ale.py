# -*- coding: utf-8 -*-
"""
Figure 10 | Nonlinear hydroclimatic regulation of severity-adjusted ecological damage
FINAL V1 | L6.0 FINAL V2 residual

Role in manuscript
------------------
Figure 8 : empirical moisture-conditioned greening association
Figure 9 : multivariate conditional greening benefit + spatial domains
Figure 10: spatially validated driver importance + ALE nonlinear responses
           + robust diagnostic transition ranges + NIRv x RootSM interaction

Scientific safeguards
---------------------
1. Target = NEW L6.0 FINAL V2 pixel Residual_damage.
2. Same predictors + same HistGradientBoosting hyperparameters as locked Figure 9.
3. Predictive skill is inherited/recomputed with 5-fold ~20-km spatial GroupKFold.
4. Driver importance = spatial OOF permutation importance, not training importance.
5. 1D ALE is used for nonlinear conditional response under correlated predictors.
6. Diagnostic transition = best continuous one-break segmented fit to ALE curve,
   restricted to the feature's central Q15-Q85 range.
7. Robustness = spatial-block model-refit bootstrap + 20/25/30-bin sensitivity.
8. Thresholds are diagnostic nonlinear transition ranges, NOT deterministic tipping points.
9. Second-order ALE = NIRv trend x RootSM interaction beyond additive main effects.
"""

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ======================================================================================
# 0 | PATHS / SETTINGS
# ======================================================================================
ROOT = r".\Weihe database V3"
L5 = os.path.join(ROOT, "L5_EventEcoFunction")

NEW_RESID = os.path.join(
    L5, "L6_SeverityAdjustedResidual_FINAL_V2",
    "L6_0_CDH_Pixel_ResidualDamage_GreeningHydroclimate_FINAL_V2.csv.gz"
)
OLD_DRIVER = os.path.join(
    L5, "L6_Mechanisms", "L6_2_SpatialML",
    "L6_2_PixelLevel_ResilienceDriverDataset.csv.gz"
)

OUT = os.path.join(ROOT, "Figure10_FINAL_V1_V2Residual")
os.makedirs(OUT, exist_ok=True)
STEM = os.path.join(OUT, "Figure10_Nonlinear_Hydroclimatic_Regulation_FINAL_V1")

IMP_OUT = STEM + "_SpatialPermutationImportance.csv"
ALE_OUT = STEM + "_MainALE_30bins.csv"
BOOT_OUT = STEM + "_BlockBootstrapThresholds.csv"
STAB_OUT = STEM + "_ThresholdStability.csv"
INT_OUT = STEM + "_NIRv_RootSM_SecondOrderALE.csv"
QC_OUT = STEM + "_QC.txt"

FEATURES = [
    "NIRv_trend", "RootSM_base", "DeepSM_base", "VPD_base",
    "Tmax_base", "Temp_base", "Precip_base", "RAD_base"
]
DISPLAY = {
    "NIRv_trend":"Greening trend",
    "RootSM_base":"Root-zone soil moisture",
    "DeepSM_base":"Deep soil moisture",
    "VPD_base":"VPD",
    "Tmax_base":"Maximum temperature",
    "Temp_base":"Mean temperature",
    "Precip_base":"Precipitation",
    "RAD_base":"Radiation"
}
UNITS = {
    "NIRv_trend":r"NIRv trend (decade$^{-1}$)",
    "RootSM_base":r"RootSM (m$^3$ m$^{-3}$)",
    "DeepSM_base":r"DeepSM (m$^3$ m$^{-3}$)",
    "VPD_base":"VPD (kPa)",
    "Tmax_base":"Tmax (°C)",
    "Temp_base":"Temperature (°C)",
    "Precip_base":"Precipitation",
    "RAD_base":r"Radiation (MJ m$^{-2}$)"
}
FOCAL = ["RootSM_base","VPD_base","Precip_base","RAD_base"]

MODEL_KW = dict(
    loss="squared_error", learning_rate=0.05, max_iter=280,
    max_leaf_nodes=15, min_samples_leaf=100,
    l2_regularization=0.15, random_state=42
)

N_ALE = 30
BOOT_N = 30
BOOT_MAX_ROWS = 150_000
PERM_REPEATS = 3
RNG = np.random.default_rng(20260911)

# ======================================================================================
# 1 | STYLE / HELPERS
# ======================================================================================
mpl.rcParams.update({
    "font.family":"Times New Roman","font.weight":"bold",
    "axes.labelweight":"bold","axes.titleweight":"bold",
    "axes.linewidth":0.9,"font.size":9.5,
    "axes.titlesize":10.8,"axes.labelsize":9.5,
    "xtick.labelsize":8.2,"ytick.labelsize":8.2,
    "legend.fontsize":7.8,"pdf.fonttype":42,"ps.fonttype":42
})

COL = {
    "RootSM_base":"#3F75A8",
    "VPD_base":"#B65343",
    "Precip_base":"#4E8E78",
    "RAD_base":"#B18D4B"
}

def banner(s):
    print("\n"+"="*112); print(s); print("="*112, flush=True)

def bold(ax):
    for t in ax.get_xticklabels()+ax.get_yticklabels():
        t.set_fontweight("bold")

def weighted_lstsq(X,y,w):
    sw=np.sqrt(np.asarray(w,float))
    return np.linalg.lstsq(X*sw[:,None], np.asarray(y,float)*sw, rcond=None)[0]

def percentile_of_value(v, x):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    return 100.0*np.mean(x<=v)

# ======================================================================================
# 2 | 1D ALE
# ======================================================================================
def ale_1d(model, X, j, n_bins=30):
    """
    First-order ALE with quantile bins.
    Returns one row per occupied bin:
    x = bin midpoint in original units;
    ALE = centered cumulative local effect;
    count = number of samples in bin.
    """
    x=X[:,j]
    edges=np.unique(np.quantile(x,np.linspace(0,1,n_bins+1)))
    if len(edges)<5:
        raise RuntimeError("Too few unique ALE edges.")
    nb=len(edges)-1
    ids=np.clip(np.searchsorted(edges,x,side="right")-1,0,nb-1)

    local=np.full(nb,np.nan)
    count=np.zeros(nb,int)
    xpos=np.full(nb,np.nan)

    for b in range(nb):
        idx=np.where(ids==b)[0]
        count[b]=len(idx)
        if len(idx)==0: continue
        X0=X[idx].copy(); X1=X[idx].copy()
        X0[:,j]=edges[b]; X1[:,j]=edges[b+1]
        local[b]=np.mean(model.predict(X1)-model.predict(X0))
        xpos[b]=0.5*(edges[b]+edges[b+1])

    valid=np.isfinite(local)
    if valid.sum()<4:
        raise RuntimeError("Too few occupied ALE bins.")

    # cumulative effect at bin centers
    loc=np.where(np.isfinite(local),local,0.0)
    cum=np.cumsum(loc)-0.5*loc
    # weighted centering
    center=np.average(cum[valid],weights=count[valid])
    ale=cum-center

    return pd.DataFrame({
        "bin":np.arange(nb),
        "x":xpos,
        "ALE":ale,
        "count":count,
        "edge_low":edges[:-1],
        "edge_high":edges[1:]
    }).query("count > 0 and x == x").reset_index(drop=True)

# ======================================================================================
# 3 | DIAGNOSTIC ALE BREAKPOINT
# ======================================================================================
def ale_breakpoint(curve, raw_x):
    """
    Continuous segmented regression:
    ALE = a + b*x + c*max(0,x-k)
    Candidate k restricted to original feature Q15-Q85.
    Weighted by ALE-bin sample count.
    """
    q=curve.dropna(subset=["x","ALE","count"]).copy()
    x=q.x.to_numpy(float); y=q.ALE.to_numpy(float); w=q["count"].to_numpy(float)

    lo,hi=np.quantile(raw_x,[.15,.85])
    candidates=x[(x>=lo)&(x<=hi)]
    if len(candidates)<4: return None

    best=None
    for k in candidates:
        X=np.column_stack([np.ones_like(x),x,np.maximum(0,x-k)])
        b=weighted_lstsq(X,y,w)
        pred=X@b
        rss=float(np.sum(w*(y-pred)**2))
        if best is None or rss<best["rss"]:
            best={"threshold_value":float(k),"rss":rss,
                  "intercept":float(b[0]),"slope_pre":float(b[1]),
                  "slope_change":float(b[2]),"slope_post":float(b[1]+b[2])}
    best["threshold_percentile"]=percentile_of_value(best["threshold_value"],raw_x)
    return best

# ======================================================================================
# 4 | SECOND-ORDER ALE INTERACTION
# ======================================================================================
def ale_2d_interaction(model, X, j, k, bins=12):
    """
    Second-order ALE interaction approximation:
    mean local second differences -> 2D cumulative surface ->
    weighted removal of row/column main components and global mean.
    """
    xj=X[:,j]; xk=X[:,k]
    ej=np.unique(np.quantile(xj,np.linspace(0,1,bins+1)))
    ek=np.unique(np.quantile(xk,np.linspace(0,1,bins+1)))
    nj,nk=len(ej)-1,len(ek)-1

    ij=np.clip(np.searchsorted(ej,xj,side="right")-1,0,nj-1)
    ik=np.clip(np.searchsorted(ek,xk,side="right")-1,0,nk-1)

    delta=np.full((nj,nk),np.nan); count=np.zeros((nj,nk),int)

    for a in range(nj):
        for b in range(nk):
            idx=np.where((ij==a)&(ik==b))[0]
            count[a,b]=len(idx)
            if len(idx)==0: continue
            X00=X[idx].copy(); X10=X[idx].copy(); X01=X[idx].copy(); X11=X[idx].copy()
            X00[:,j]=ej[a];   X00[:,k]=ek[b]
            X10[:,j]=ej[a+1]; X10[:,k]=ek[b]
            X01[:,j]=ej[a];   X01[:,k]=ek[b+1]
            X11[:,j]=ej[a+1]; X11[:,k]=ek[b+1]
            delta[a,b]=np.mean(model.predict(X11)-model.predict(X10)-model.predict(X01)+model.predict(X00))

    d=np.where(np.isfinite(delta),delta,0.0)
    H=np.cumsum(np.cumsum(d,axis=0),axis=1)
    valid=count>0

    # remove weighted row/column means, then global center
    row=np.zeros(nj); col=np.zeros(nk)
    for a in range(nj):
        m=valid[a]
        row[a]=np.average(H[a,m],weights=count[a,m]) if m.any() else 0
    for b in range(nk):
        m=valid[:,b]
        col[b]=np.average(H[m,b],weights=count[m,b]) if m.any() else 0
    H2=H-row[:,None]-col[None,:]
    ctr=np.average(H2[valid],weights=count[valid])
    H2=H2-ctr
    H2[~valid]=np.nan

    xmid=0.5*(ej[:-1]+ej[1:])
    ymid=0.5*(ek[:-1]+ek[1:])
    return xmid,ymid,H2,count

# ======================================================================================
# 5 | DATA
# ======================================================================================
banner("LOAD V2 RESIDUAL + COVARIATES")
for fp in [NEW_RESID,OLD_DRIVER]:
    if not os.path.exists(fp): raise FileNotFoundError(fp)

new=pd.read_csv(NEW_RESID,compression="gzip",low_memory=False)
old_head=pd.read_csv(OLD_DRIVER,compression="gzip",nrows=2)
use_old=["row","col","DeepSM_base","VPD_base","Tmax_base","Temp_base","Precip_base","RAD_base"]
miss=[c for c in use_old if c not in old_head.columns]
if miss: raise RuntimeError(f"Missing covariates: {miss}")
old=pd.read_csv(OLD_DRIVER,compression="gzip",usecols=use_old,low_memory=False)

d=new.merge(old,on=["row","col"],how="left",validate="one_to_one")
need=["Residual_damage","block20"]+FEATURES
for c in need: d[c]=pd.to_numeric(d[c],errors="coerce")
d=d.dropna(subset=need).copy()
d["block20"]=d.block20.astype(int)

X=d[FEATURES].to_numpy(float)
y=d.Residual_damage.to_numpy(float)
groups=d.block20.to_numpy()
print(f"Rows={len(d):,} | blocks={d.block20.nunique():,}")

# ======================================================================================
# 6 | SPATIAL OOF + PERMUTATION IMPORTANCE
# ======================================================================================
banner("SPATIAL GROUP OOF + PERMUTATION IMPORTANCE")
gkf=GroupKFold(n_splits=5)
oof=np.full(len(d),np.nan)
imp={f:[] for f in FEATURES}
fold_stats=[]

for fold,(tr,te) in enumerate(gkf.split(X,y,groups),1):
    m=HistGradientBoostingRegressor(**MODEL_KW).fit(X[tr],y[tr])
    base=m.predict(X[te]); oof[te]=base
    mse0=mean_squared_error(y[te],base)
    fold_stats.append((fold,r2_score(y[te],base),mse0**0.5,mean_absolute_error(y[te],base)))

    for j,f in enumerate(FEATURES):
        vals=[]
        for rr in range(PERM_REPEATS):
            xp=X[te].copy()
            xp[:,j]=xp[RNG.permutation(len(te)),j]
            vals.append(mean_squared_error(y[te],m.predict(xp))-mse0)
        imp[f].append(np.mean(vals))
    print(f"fold {fold} | R2={fold_stats[-1][1]:.4f} | RMSE={fold_stats[-1][2]:.5f}",flush=True)

OOF_R2=r2_score(y,oof); OOF_RMSE=mean_squared_error(y,oof)**0.5; OOF_MAE=mean_absolute_error(y,oof)
imp_df=pd.DataFrame({
    "feature":FEATURES,
    "display_name":[DISPLAY[f] for f in FEATURES],
    "delta_MSE_mean":[np.mean(imp[f]) for f in FEATURES],
    "delta_MSE_sd":[np.std(imp[f],ddof=1) for f in FEATURES]
}).sort_values("delta_MSE_mean",ascending=False)
pos=np.clip(imp_df.delta_MSE_mean.to_numpy(),0,None)
imp_df["relative_importance_pct"]=100*pos/pos.sum() if pos.sum()>0 else np.nan
imp_df.to_csv(IMP_OUT,index=False)

print(f"OOF R2={OOF_R2:.4f} | RMSE={OOF_RMSE:.5f} | MAE={OOF_MAE:.5f}")
print(imp_df.to_string(index=False))

# final interpretive model: same algorithm as Figure 9
banner("TRAIN FINAL INTERPRETIVE MODEL")
model=HistGradientBoostingRegressor(**MODEL_KW).fit(X,y)
train_r2=r2_score(y,model.predict(X))
print(f"Training R2={train_r2:.4f}; scientific predictive performance remains spatial OOF R2={OOF_R2:.4f}")

# ======================================================================================
# 7 | MAIN ALE + BIN SENSITIVITY
# ======================================================================================
banner("MAIN 1D ALE + BIN SENSITIVITY")
all_ale=[]; main_rows=[]; bin_rows=[]

for f in FOCAL:
    j=FEATURES.index(f)
    c=ale_1d(model,X,j,N_ALE)
    c["feature"]=f; all_ale.append(c)
    bp=ale_breakpoint(c,X[:,j])
    main_rows.append({"feature":f,**bp})
    print(f"{f}: main Q={bp['threshold_percentile']:.2f}, value={bp['threshold_value']:.6g}")

    for nb in [20,25,30]:
        cc=ale_1d(model,X,j,nb)
        bpp=ale_breakpoint(cc,X[:,j])
        bin_rows.append({"feature":f,"n_bins":nb,**bpp})

ale_df=pd.concat(all_ale,ignore_index=True)
ale_df.to_csv(ALE_OUT,index=False)
main_df=pd.DataFrame(main_rows)
bin_df=pd.DataFrame(bin_rows)

# ======================================================================================
# 8 | SPATIAL-BLOCK MODEL-REFIT BOOTSTRAP
# ======================================================================================
banner("SPATIAL-BLOCK MODEL-REFIT BOOTSTRAP")
block_to_idx={b:np.where(groups==b)[0] for b in np.unique(groups)}
blocks=np.array(list(block_to_idx.keys()))
boot_rows=[]

for b in range(BOOT_N):
    samp=RNG.choice(blocks,size=len(blocks),replace=True)
    idx=np.concatenate([block_to_idx[z] for z in samp])
    if len(idx)>BOOT_MAX_ROWS:
        idx=RNG.choice(idx,size=BOOT_MAX_ROWS,replace=False)

    mb=HistGradientBoostingRegressor(**MODEL_KW).fit(X[idx],y[idx])
    for f in FOCAL:
        j=FEATURES.index(f)
        c=ale_1d(mb,X[idx],j,N_ALE)
        bp=ale_breakpoint(c,X[:,j])  # percentile relative to full retained population
        if bp is not None:
            boot_rows.append({"boot":b+1,"feature":f,**bp})
    if (b+1)==1 or (b+1)%5==0:
        print(f"bootstrap {b+1:02d}/{BOOT_N}",flush=True)

boot_df=pd.DataFrame(boot_rows)
boot_df.to_csv(BOOT_OUT,index=False)

# ======================================================================================
# 9 | STABILITY SUMMARY
# ======================================================================================
banner("THRESHOLD STABILITY")
stab=[]
for f in FOCAL:
    m=main_df[main_df.feature.eq(f)].iloc[0]
    z=boot_df[boot_df.feature.eq(f)]
    bs=bin_df[bin_df.feature.eq(f)]
    q=z.threshold_percentile.to_numpy(float)
    val=z.threshold_value.to_numpy(float)
    bins_q=bs.threshold_percentile.to_numpy(float)

    q25,q50,q75=np.quantile(q,[.25,.5,.75])
    q025,q975=np.quantile(q,[.025,.975])
    v025,v50,v975=np.quantile(val,[.025,.5,.975])
    p5=float(np.mean(np.abs(q-float(m.threshold_percentile))<=5))
    bin_range=float(np.max(bins_q)-np.min(bins_q))
    iqr=float(q75-q25)

    if (p5>=.60) and (iqr<=15) and (bin_range<=10):
        status="stable"
    elif (p5>=.40) and (iqr<=25) and (bin_range<=20):
        status="moderate"
    else:
        status="uncertain"

    stab.append({
        "feature":f,"display_name":DISPLAY[f],
        "main_threshold_value":m.threshold_value,
        "main_threshold_percentile":m.threshold_percentile,
        "bootstrap_value_median":v50,
        "bootstrap_value_CI_low":v025,
        "bootstrap_value_CI_high":v975,
        "bootstrap_Q_median":q50,
        "bootstrap_Q_IQR_low":q25,
        "bootstrap_Q_IQR_high":q75,
        "bootstrap_Q_CI_low":q025,
        "bootstrap_Q_CI_high":q975,
        "P_within_mainQ_plusminus5":p5,
        "bin_Q_min":float(np.min(bins_q)),
        "bin_Q_max":float(np.max(bins_q)),
        "bin_Q_range":bin_range,
        "slope_pre":m.slope_pre,
        "slope_post":m.slope_post,
        "stability":status,
        "n_boot":len(z)
    })

stab_df=pd.DataFrame(stab)
stab_df.to_csv(STAB_OUT,index=False)
print(stab_df[["display_name","bootstrap_value_median","bootstrap_Q_median",
               "bootstrap_Q_IQR_low","bootstrap_Q_IQR_high",
               "P_within_mainQ_plusminus5","bin_Q_range","stability"]].to_string(index=False))

# ======================================================================================
# 10 | SECOND-ORDER ALE: GREENING x ROOTSM
# ======================================================================================
banner("SECOND-ORDER ALE | NIRv TREND x ROOTSM")
jg=FEATURES.index("NIRv_trend"); js=FEATURES.index("RootSM_base")
gx,sx,H,CNT=ale_2d_interaction(model,X,jg,js,bins=12)

rr=[]
for i,sv in enumerate(sx):
    for j,gv in enumerate(gx):
        rr.append({"RootSM":sv,"NIRv_trend":gv,"ALE_interaction":H[j,i] if H.shape==(len(gx),len(sx)) else H[i,j],
                   "count":CNT[j,i] if CNT.shape==(len(gx),len(sx)) else CNT[i,j]})
int_df=pd.DataFrame(rr)
int_df.to_csv(INT_OUT,index=False)
print(f"Interaction range: {np.nanmin(H):+.6f} to {np.nanmax(H):+.6f}")

# ======================================================================================
# 11 | FIGURE
# ======================================================================================
banner("DRAW FIGURE 10")
fig=plt.figure(figsize=(12.5,7.5),facecolor="white")
gs=fig.add_gridspec(2,3,left=.065,right=.985,bottom=.09,top=.965,wspace=.28,hspace=.30)

# a | spatial permutation importance
ax=fig.add_subplot(gs[0,0])
ii=imp_df.sort_values("relative_importance_pct",ascending=True)
ax.barh(ii.display_name,ii.relative_importance_pct,color="#6F8191",height=.68)
ax.set_xlabel("Spatial OOF permutation importance (%)")
ax.set_title("a  Spatial driver importance",loc="left",pad=5)
ax.spines[["top","right"]].set_visible(False); bold(ax)

# b-e | ALE
panel_map=[("RootSM_base","b"),("VPD_base","c"),("Precip_base","d"),("RAD_base","e")]
positions=[gs[0,1],gs[0,2],gs[1,0],gs[1,1]]

for (f,letter),posn in zip(panel_map,positions):
    ax=fig.add_subplot(posn)
    q=ale_df[ale_df.feature.eq(f)].sort_values("x")
    st=stab_df[stab_df.feature.eq(f)].iloc[0]
    ax.plot(q.x,q.ALE,color=COL[f],lw=2.25,marker="o",ms=3.5,mfc="white")
    ax.axhline(0,color=".55",lw=.75)

    med=st.bootstrap_value_median
    lo=st.bootstrap_value_CI_low; hi=st.bootstrap_value_CI_high
    ax.axvspan(lo,hi,color=COL[f],alpha=.10,lw=0)
    ls="-" if st.stability=="stable" else "--"
    ax.axvline(med,color=COL[f],lw=1.35,ls=ls)
    ax.set_xlabel(UNITS[f]); ax.set_ylabel("ALE on residual damage")
    ax.set_title(f"{letter}  {DISPLAY[f]}",loc="left",pad=5)
    ax.text(.03,.95,
            f"Q~{st.bootstrap_Q_median:.1f} | {st.stability}\n"
            f"95% Q: {st.bootstrap_Q_CI_low:.1f}–{st.bootstrap_Q_CI_high:.1f}",
            transform=ax.transAxes,ha="left",va="top",fontsize=7.8,
            bbox=dict(fc="white",ec="none",alpha=.78,pad=1.5))
    ax.spines[["top","right"]].set_visible(False); bold(ax)

# f | second-order ALE interaction
ax=fig.add_subplot(gs[1,2])
# H axes are [greening_bin, rootsm_bin]
Z=H.T if H.shape==(len(gx),len(sx)) else H
v=np.nanpercentile(np.abs(Z[np.isfinite(Z)]),98)
im=ax.imshow(Z,origin="lower",aspect="auto",
             extent=[gx.min(),gx.max(),sx.min(),sx.max()],
             cmap="RdBu_r",norm=TwoSlopeNorm(vmin=-v,vcenter=0,vmax=v),
             interpolation="nearest")
ax.set_xlabel("NIRv trend (decade$^{-1}$)")
ax.set_ylabel(r"RootSM (m$^3$ m$^{-3}$)")
ax.set_title("f  Greening × RootSM interaction",loc="left",pad=5)
bold(ax)
cb=fig.colorbar(im,ax=ax,orientation="horizontal",fraction=.05,pad=.13)
cb.set_label("Second-order ALE interaction",fontweight="bold")
for t in cb.ax.get_xticklabels():t.set_fontweight("bold")

for ext in ["png","pdf"]:
    fig.savefig(STEM+"."+ext,dpi=300,bbox_inches="tight",pad_inches=.04,facecolor="white")
plt.close(fig)

# ======================================================================================
# 12 | QC
# ======================================================================================
lines=[
"="*110,
"FIGURE 10 | NONLINEAR HYDROCLIMATIC REGULATION | FINAL V1 QC",
"="*110,
f"Target source              : {NEW_RESID}",
f"Covariate source           : {OLD_DRIVER}",
f"Usable equal-weight pixels : {len(d):,}",
f"Spatial groups (~20 km)    : {d.block20.nunique():,}",
"",
"MODEL CONSISTENCY WITH FIGURE 9",
"Algorithm                  : HistGradientBoostingRegressor",
f"Features                   : {FEATURES}",
"Hyperparameters            : identical to Figure 9 FINAL V5",
f"Spatial OOF R2             : {OOF_R2:.6f}",
f"Spatial OOF RMSE           : {OOF_RMSE:.6f}",
f"Spatial OOF MAE            : {OOF_MAE:.6f}",
f"Full-model train R2        : {train_r2:.6f}",
"Predictive performance must be reported from spatial OOF, not training R2.",
"",
"THRESHOLD METHOD",
"1D response                : first-order ALE",
f"Main ALE bins              : {N_ALE}",
"Breakpoint model           : continuous one-break segmented regression on ALE bins",
"Candidate search           : feature central Q15-Q85 only",
"Curve weighting            : ALE-bin sample counts",
f"Spatial block bootstraps   : {BOOT_N}",
"Bin sensitivity            : 20 / 25 / 30 bins",
"Threshold meaning          : diagnostic nonlinear transition; NOT deterministic tipping point",
"",
"STABILITY RULES",
"stable   : P(|Qboot-Qmain|<=5)>=0.60 AND bootstrap Q-IQR<=15 AND bin-Q-range<=10",
"moderate : P>=0.40 AND Q-IQR<=25 AND bin-Q-range<=20",
"uncertain: otherwise",
"",
"THRESHOLD SUMMARY",
stab_df[["display_name","bootstrap_value_median","bootstrap_value_CI_low","bootstrap_value_CI_high",
         "bootstrap_Q_median","bootstrap_Q_CI_low","bootstrap_Q_CI_high",
         "slope_pre","slope_post","stability"]].to_string(index=False),
"",
"SPATIAL PERMUTATION IMPORTANCE",
imp_df[["display_name","delta_MSE_mean","relative_importance_pct"]].to_string(index=False),
"",
"SECOND-ORDER ALE",
"Interaction               : NIRv_trend x RootSM_base",
f"Interaction minimum       : {np.nanmin(H):+.8f}",
f"Interaction maximum       : {np.nanmax(H):+.8f}",
f"Interaction median        : {np.nanmedian(H):+.8f}",
"",
"INTERPRETATION LOCKS",
"1. Figure 10 uses NEW L6.0 FINAL V2 Residual_damage.",
"2. Figure 9 spatial OOF remains the primary predictive validation.",
"3. ALE describes model-based conditional nonlinear associations, not causal effects.",
"4. VPD enters uWUE upstream; its ALE must not be interpreted as an independent causal effect.",
"5. NIRv trend is a long-term greening proxy, not restoration treatment intensity.",
"6. Wide/unstable bootstrap ranges must be reported as regime uncertainty, not hidden.",
"7. Only stable/moderate transitions may be emphasized in the main text.",
"8. Second-order ALE is interaction beyond additive main effects.",
"",
f"Importance CSV            : {IMP_OUT}",
f"Main ALE CSV              : {ALE_OUT}",
f"Bootstrap CSV             : {BOOT_OUT}",
f"Stability CSV             : {STAB_OUT}",
f"Interaction CSV           : {INT_OUT}",
f"PNG                       : {STEM+'.png'}",
f"PDF                       : {STEM+'.pdf'}",
"FINAL STATUS: READY FOR SCIENTIFIC/VISUAL QC"
]
with open(QC_OUT,"w",encoding="utf-8") as f:f.write("\n".join(lines))
print("\n".join(lines))
