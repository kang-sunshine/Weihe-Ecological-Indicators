# -*- coding: utf-8 -*-
"""
Figure 9 | Conditional greening benefit under hydroclimatic limitation
FINAL V5 | Rebuilt on L6.0 FINAL V2 residual damage

Scientific role
---------------
Figure 8: empirical moisture-conditioned greening association.
Figure 9: multivariate conditional association + spatialization.
Figure 10: ALE-based nonlinear environmental response/thresholds.

Key safeguards
--------------
1. Target is NEW L6.0 V2 pixel Residual_damage, not legacy L6.2 Residual_damage.
2. Old L6.2 is used only for environmental covariates.
3. Equal pixel weights.
4. Predictive performance is 5-fold spatial GroupKFold OOF using ~20-km blocks.
5. Conditional greening benefit is model-based association, not causal CATE.
6. HCL and diagnostic domains are relative descriptors, not physical thresholds.
7. No spatial smoothing/interpolation.
"""

import os, glob, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, ListedColormap
from matplotlib.patches import Patch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ======================================================================================
# 0 | PATHS
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

OUT = os.path.join(ROOT, "Figure9_FINAL_V5_V2Residual")
os.makedirs(OUT, exist_ok=True)
STEM = os.path.join(OUT, "Figure9_ConditionalGreeningBenefit_FINAL_V5")
QC = STEM + "_QC.txt"
DATA_OUT = STEM + "_PixelResults.csv.gz"
CURVE_OUT = STEM + "_HCL_BinnedBenefit.csv"

SHP_DIR = r".\weihe\Weihe"
DPI = 300
NY, NX = 414, 703

FEATURES = [
    "NIRv_trend", "RootSM_base", "DeepSM_base", "VPD_base",
    "Tmax_base", "Temp_base", "Precip_base", "RAD_base"
]
ENV = [x for x in FEATURES if x != "NIRv_trend"]

MODEL_KW = dict(
    loss="squared_error",
    learning_rate=0.05,
    max_iter=280,
    max_leaf_nodes=15,
    min_samples_leaf=100,
    l2_regularization=0.15,
    random_state=42
)

# ======================================================================================
# 1 | STYLE
# ======================================================================================
mpl.rcParams.update({
    "font.family":"Times New Roman","font.weight":"bold",
    "axes.labelweight":"bold","axes.titleweight":"bold",
    "axes.linewidth":0.9,"font.size":10,
    "axes.titlesize":11.3,"axes.labelsize":10,
    "xtick.labelsize":8.5,"ytick.labelsize":8.5,
    "pdf.fonttype":42,"ps.fonttype":42
})

def banner(s):
    print("\n" + "="*108); print(s); print("="*108, flush=True)

def bold(ax):
    for t in ax.get_xticklabels()+ax.get_yticklabels(): t.set_fontweight("bold")

# ======================================================================================
# 2 | INPUT + MERGE
# ======================================================================================
banner("FIGURE 9 V5 | LOAD NEW V2 RESIDUAL + ENVIRONMENTAL COVARIATES")
for fp in [NEW_RESID, OLD_DRIVER]:
    if not os.path.exists(fp): raise FileNotFoundError(fp)

new = pd.read_csv(NEW_RESID, compression="gzip", low_memory=False)
old_head = pd.read_csv(OLD_DRIVER, compression="gzip", nrows=2)
need_old = ["row","col","DeepSM_base","VPD_base","Tmax_base","Temp_base","Precip_base","RAD_base"]
miss = [c for c in need_old if c not in old_head.columns]
if miss: raise RuntimeError(f"Missing old-driver covariates: {miss}")

old = pd.read_csv(OLD_DRIVER, compression="gzip", usecols=need_old, low_memory=False)
d = new.merge(old, on=["row","col"], how="left", validate="one_to_one")

need = ["Residual_damage","block20"] + FEATURES
for c in need:
    d[c] = pd.to_numeric(d[c], errors="coerce")
d = d.dropna(subset=need).copy()
d["row"] = d.row.astype(int); d["col"] = d.col.astype(int)
d["block20"] = d.block20.astype(int)

print(f"Usable pixels: {len(d):,}")
print("Target mean / SD:", d.Residual_damage.mean(), d.Residual_damage.std())

X = d[FEATURES].to_numpy(float)
y = d.Residual_damage.to_numpy(float)
groups = d.block20.to_numpy()

# ======================================================================================
# 3 | SPATIAL OOF VALIDATION
# ======================================================================================
banner("5-FOLD SPATIAL GROUP OOF VALIDATION")
gkf = GroupKFold(n_splits=5)
oof = np.full(len(d), np.nan)
fold_rows=[]

for fold,(tr,te) in enumerate(gkf.split(X,y,groups),1):
    m = HistGradientBoostingRegressor(**MODEL_KW)
    m.fit(X[tr],y[tr])
    p = m.predict(X[te])
    oof[te]=p
    fold_rows.append({
        "fold":fold,"n_train":len(tr),"n_test":len(te),
        "r2":r2_score(y[te],p),
        "rmse":mean_squared_error(y[te],p)**0.5,
        "mae":mean_absolute_error(y[te],p)
    })
    print(f"fold {fold}: R2={fold_rows[-1]['r2']:.4f} | RMSE={fold_rows[-1]['rmse']:.5f}",flush=True)

OOF_R2 = r2_score(y,oof)
OOF_RMSE = mean_squared_error(y,oof)**0.5
OOF_MAE = mean_absolute_error(y,oof)
print(f"OOF overall: R2={OOF_R2:.4f} | RMSE={OOF_RMSE:.5f} | MAE={OOF_MAE:.5f}")

# final interpretive model
model = HistGradientBoostingRegressor(**MODEL_KW).fit(X,y)

# ======================================================================================
# 4 | CONDITIONAL GREENING BENEFIT
# ======================================================================================
banner("CONDITIONAL GREENING BENEFIT")
G20,G80 = np.quantile(d.NIRv_trend,[.20,.80])
iG = FEATURES.index("NIRv_trend")
Xlo=X.copy(); Xhi=X.copy()
Xlo[:,iG]=G20; Xhi[:,iG]=G80
pred_lo=model.predict(Xlo); pred_hi=model.predict(Xhi)
benefit=pred_lo-pred_hi
EPS_B=1e-12
pos = benefit > EPS_B
neg = benefit < -EPS_B
neu = np.abs(benefit) <= EPS_B

# relative hydroclimatic limitation
zv=(d.VPD_base-d.VPD_base.mean())/d.VPD_base.std(ddof=0)
zs=(d.RootSM_base-d.RootSM_base.mean())/d.RootSM_base.std(ddof=0)
hcl=(zv-zs).to_numpy(float)

d["OOF_pred"]=oof
d["GreeningBenefit"]=benefit
d["HCL"]=hcl

print(f"NIRv q20/q80: {G20:.8f} / {G80:.8f}")
print("Benefit P05/P50/P95:", np.quantile(benefit,[.05,.5,.95]))
print(f"Positive/neutral/negative: {pos.mean()*100:.2f}% / {neu.mean()*100:.2f}% / {neg.mean()*100:.2f}%")

# ======================================================================================
# 5 | TRANSPARENT DIAGNOSTIC DOMAINS
# ======================================================================================
# Natural zero-reference rules only:
# HCL <= 0 vs > 0; Benefit > 0 vs <= 0.
# These are diagnostic quadrants, NOT ecological thresholds.
domain=np.zeros(len(d),np.int8)  # 0 = no modelled greening contrast
domain[(hcl<=0)&pos] = 1
domain[(hcl> 0)&pos] = 2
domain[(hcl<=0)&neg] = 3
domain[(hcl> 0)&neg] = 4
d["Domain"]=domain

DOM_NAMES = {
    0:"No modelled greening contrast",
    1:"Low limitation / buffering association",
    2:"High limitation / retained benefit",
    3:"Low limitation / greening trade-off",
    4:"High limitation / hydroclimatic trade-off"
}
dom_pct={k:100*np.mean(domain==k) for k in DOM_NAMES}
for k,v in dom_pct.items(): print(f"{k} {DOM_NAMES[k]}: {v:.2f}%")

# ======================================================================================
# 6 | HCL-CONDITIONED BENEFIT CURVE
# ======================================================================================
# 15 equal-frequency HCL bins; uncertainty from 20-km block means.
d["HCL_bin"]=pd.qcut(d.HCL,15,labels=False,duplicates="drop")+1
curve=[]
for b,z in d.groupby("HCL_bin",observed=True):
    bm=z.groupby("block20",observed=True).agg(HCL=("HCL","mean"),B=("GreeningBenefit","mean"))
    curve.append({
        "bin":int(b),"HCL":float(z.HCL.mean()),"Benefit":float(z.GreeningBenefit.mean()),
        "Benefit_SE_block":float(bm.B.std(ddof=1)/np.sqrt(len(bm))) if len(bm)>1 else np.nan,
        "n_pixels":len(z),"n_blocks":len(bm)
    })
curve=pd.DataFrame(curve).sort_values("HCL")
curve.to_csv(CURVE_OUT,index=False)

# ======================================================================================
# 7 | GRID + BASIN
# ======================================================================================
benefit_grid=np.full((NY,NX),np.nan,np.float32)
domain_grid=np.full((NY,NX),np.nan,np.float32)
benefit_grid[d.row,d.col]=benefit.astype(np.float32)
domain_grid[d.row,d.col]=domain.astype(np.float32)

# coordinate source
cands=[
    os.path.join(L5,"Figure3_Hydroclimatic_CDHExposure_FINAL_LOCKED","Figure3_SpatialMetrics_FINAL_LOCKED.nc"),
    os.path.join(L5,"Figure3_HydroclimaticExtremeRegime_FINAL_V5","Figure3_SpatialMetrics_FINAL_V5.nc")
]
nc=next((f for f in cands if os.path.exists(f)),None)
if nc is None:
    hits=glob.glob(os.path.join(L5,"**","Figure3_SpatialMetrics_FINAL*.nc"),recursive=True)
    if not hits: raise FileNotFoundError("Cannot locate Figure 3 coordinate NetCDF.")
    nc=hits[-1]
with xr.open_dataset(nc) as ds:
    xv=ds.x.values; yv=ds.y.values
origin="lower" if yv[-1]>yv[0] else "upper"
extent=[xv.min(),xv.max(),yv.min(),yv.max()]

boundary=None
try:
    import geopandas as gpd
    fs=glob.glob(os.path.join(SHP_DIR,"**","*.shp"),recursive=True)
    if fs:
        boundary=gpd.read_file(fs[0])
        if boundary.crs is not None: boundary=boundary.to_crs(4326)
        boundary=boundary.boundary
except Exception as e:
    print("Boundary warning:",e)

# ======================================================================================
# 8 | FIGURE
# ======================================================================================
banner("DRAW FIGURE 9 FINAL V5")
fig=plt.figure(figsize=(12.3,7.3),facecolor="white")
gs=fig.add_gridspec(2,2,left=.07,right=.985,bottom=.095,top=.965,wspace=.22,hspace=.28)

# a | spatial OOF validation
ax=fig.add_subplot(gs[0,0])
lo=min(np.nanpercentile(y,1),np.nanpercentile(oof,1))
hi=max(np.nanpercentile(y,99),np.nanpercentile(oof,99))
hb=ax.hexbin(y,oof,gridsize=52,mincnt=1,cmap="Greys",bins="log")
ax.plot([lo,hi],[lo,hi],ls="--",lw=1.2,color="#A94442")
ax.set(xlim=(lo,hi),ylim=(lo,hi),
       xlabel="Observed residual damage",ylabel="Spatial OOF prediction")
ax.set_title("a  Spatial cross-validation",loc="left",pad=5)
ax.text(.04,.94,f"$R^2_{{OOF}}$ = {OOF_R2:.3f}\nRMSE = {OOF_RMSE:.3f}",
        transform=ax.transAxes,ha="left",va="top",fontsize=9.5,fontweight="bold")
ax.spines[["top","right"]].set_visible(False); bold(ax)

# b | conditional benefit map
ax=fig.add_subplot(gs[0,1])
vv=np.nanpercentile(np.abs(benefit),98)
im=ax.imshow(benefit_grid,extent=extent,origin=origin,interpolation="nearest",
             cmap="RdBu",norm=TwoSlopeNorm(vmin=-vv,vcenter=0,vmax=vv),aspect="auto")
if boundary is not None: boundary.plot(ax=ax,color="black",linewidth=.65)
ax.set_title("b  Conditional greening benefit",loc="left",pad=5)
ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)"); bold(ax)
cb=fig.colorbar(im,ax=ax,orientation="horizontal",fraction=.05,pad=.10)
cb.set_label(r"Benefit, $\hat D(G_{20})-\hat D(G_{80})$",fontweight="bold")
for t in cb.ax.get_xticklabels(): t.set_fontweight("bold")

# c | HCL-conditioned benefit
ax=fig.add_subplot(gs[1,0])
x=curve.HCL.to_numpy(); yy=curve.Benefit.to_numpy(); se=curve.Benefit_SE_block.to_numpy()
ax.fill_between(x,yy-1.96*se,yy+1.96*se,color="#4C7A82",alpha=.15,lw=0)
ax.plot(x,yy,color="#356E78",lw=2.3,marker="o",ms=4.5,mfc="white")
ax.axhline(0,color=".45",lw=.8); ax.axvline(0,color=".45",lw=.8,ls="--")
ax.set(xlabel=r"Relative hydroclimatic limitation, HCL = $Z(VPD)-Z(RootSM)$",
       ylabel="Conditional greening benefit")
ax.set_title("c  Hydroclimatic modulation of greening benefit",loc="left",pad=5)
ax.spines[["top","right"]].set_visible(False); bold(ax)

# d | diagnostic domains
ax=fig.add_subplot(gs[1,1])
cmap=ListedColormap(["#D9D9D9","#6A9F86","#B9A15F","#D98C62","#A94A42"])
im2=ax.imshow(domain_grid,extent=extent,origin=origin,interpolation="nearest",
              cmap=cmap,vmin=-.5,vmax=4.5,aspect="auto")
if boundary is not None: boundary.plot(ax=ax,color="black",linewidth=.65)
ax.set_title("d  Diagnostic greening–hydroclimate domains",loc="left",pad=5)
ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)"); bold(ax)

handles=[
    Patch(fc="#D9D9D9",label=f"No modelled contrast ({dom_pct[0]:.1f}%)"),
    Patch(fc="#6A9F86",label=f"Low limitation / benefit ({dom_pct[1]:.1f}%)"),
    Patch(fc="#B9A15F",label=f"High limitation / benefit ({dom_pct[2]:.1f}%)"),
    Patch(fc="#D98C62",label=f"Low limitation / trade-off ({dom_pct[3]:.1f}%)"),
    Patch(fc="#A94A42",label=f"High limitation / trade-off ({dom_pct[4]:.1f}%)")
]
ax.legend(handles=handles,loc="lower left",frameon=True,framealpha=.90,
          fontsize=7.7,borderpad=.5,handlelength=1.3)

for ext in ["png","pdf"]:
    fig.savefig(STEM+"."+ext,dpi=DPI,bbox_inches="tight",pad_inches=.04,facecolor="white")
plt.close(fig)

# save pixel outputs
d[["row","col","block20","Residual_damage","OOF_pred","NIRv_trend",
   "RootSM_base","VPD_base","HCL","GreeningBenefit","Domain"]].to_csv(
    DATA_OUT,index=False,compression={"method":"gzip","compresslevel":1}
)

# ======================================================================================
# 9 | QC
# ======================================================================================
lines=[
"="*106,
"FIGURE 9 | CONDITIONAL GREENING BENEFIT UNDER HYDROCLIMATIC LIMITATION | FINAL V5 QC",
"="*106,
f"New response source        : {NEW_RESID}",
f"Environmental covariates   : {OLD_DRIVER}",
f"Usable equal-weight pixels : {len(d):,}",
f"Spatial groups (~20 km)    : {d.block20.nunique():,}",
"",
"MODEL",
"Algorithm                  : HistGradientBoostingRegressor",
f"Features                   : {FEATURES}",
"Validation                 : 5-fold spatial GroupKFold OOF",
f"OOF R2                     : {OOF_R2:.6f}",
f"OOF RMSE                   : {OOF_RMSE:.6f}",
f"OOF MAE                    : {OOF_MAE:.6f}",
"",
"CONDITIONAL GREENING BENEFIT",
"Definition                 : prediction(G=q20) - prediction(G=q80), other covariates held pixel-specific",
f"NIRv trend q20             : {G20:.10f}",
f"NIRv trend q80             : {G80:.10f}",
f"Benefit P05                : {np.quantile(benefit,.05):+.8f}",
f"Benefit median             : {np.median(benefit):+.8f}",
f"Benefit P95                : {np.quantile(benefit,.95):+.8f}",
f"Benefit > 0                : {pos.mean()*100:.2f}%",
f"Benefit = 0 (numerical)    : {neu.mean()*100:.2f}%",
f"Benefit < 0                : {neg.mean()*100:.2f}%",
"",
"HYDROCLIMATIC LIMITATION",
"Definition                 : HCL = Z(VPD_base) - Z(RootSM_base)",
f"HCL median                 : {np.median(hcl):+.6f}",
"HCL=0 is a relative standardized reference, NOT a physical threshold.",
"",
"DIAGNOSTIC DOMAIN RULES",
"0 = |Benefit|<=1e-12       : No modelled greening contrast",
"1 = HCL<=0 and Benefit>0   : Low limitation / buffering association",
"2 = HCL>0  and Benefit>0   : High limitation / retained benefit",
"3 = HCL<=0 and Benefit<0   : Low limitation / greening trade-off",
"4 = HCL>0  and Benefit<0   : High limitation / hydroclimatic trade-off",
]
for k in range(0,5):
    lines.append(f"Domain {k} fraction          : {dom_pct[k]:.4f}%")
lines += [
"",
"INTERPRETATION LOCKS",
"1. Target is the NEW L6.0 FINAL V2 Residual_damage.",
"2. Old L6.2 Residual_damage is NOT used.",
"3. Benefit is a model-based conditional association, not causal CATE.",
"4. NIRv trend is a greening proxy, not restoration treatment intensity.",
"5. HCL and the diagnostic domains are relative classifications, not ecological tipping thresholds.",
"6. Exact-zero benefit means the fitted tree ensemble gives identical q20/q80 predictions for that pixel; it is not classified as trade-off.",
"7. No spatial smoothing/interpolation is used.",
"8. Continuous nonlinear thresholds are reserved for Figure 10 ALE.",
"",
f"Pixel output               : {DATA_OUT}",
f"HCL curve                  : {CURVE_OUT}",
f"PNG                        : {STEM+'.png'}",
f"PDF                        : {STEM+'.pdf'}",
"FINAL STATUS: READY FOR VISUAL/SCIENTIFIC QC"
]
with open(QC,"w",encoding="utf-8") as f: f.write("\n".join(lines))
print("\n".join(lines))
