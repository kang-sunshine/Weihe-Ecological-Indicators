# -*- coding: utf-8 -*-
import re,csv
import numpy as np
import rasterio
from pipeline_config import *

YEAR_RE=re.compile(r"_(\d{4})\.tif$",re.I)

def year_from_path(p):
    m=YEAR_RE.search(p.name)
    if not m: raise ValueError(p.name)
    return int(m.group(1))

def build_map(pattern):
    out={}
    for p in DATA_DIR.glob(pattern):
        y=year_from_path(p)
        if y in out: raise RuntimeError(f"Duplicate {y}")
        out[y]=p
    return out

def nearly(a,b,tol=1e-7):
    return len(a)==len(b) and all(abs(float(x)-float(y))<=tol for x,y in zip(a,b))

def sampled_stats(src):
    bands=np.linspace(1,src.count,min(8,src.count),dtype=int)
    vals=[]
    for b in bands:
        a=src.read(int(b),out_shape=(1,min(256,src.height),min(256,src.width)),
                   masked=True,resampling=rasterio.enums.Resampling.nearest)
        x=np.asarray(a.filled(np.nan),dtype=float); x[x<-900]=np.nan
        f=x[np.isfinite(x)]
        if f.size: vals.append((float(np.nanmin(f)),float(np.nanmax(f)),float(np.nanmean(f)),float(np.mean(np.isfinite(x)))))
    return vals[0] if len(vals)==1 else (min(v[0] for v in vals),max(v[1] for v in vals),float(np.mean([v[2] for v in vals])),float(np.mean([v[3] for v in vals]))) if vals else (np.nan,np.nan,np.nan,0)

def main():
    l2,l3=build_map(L2_PATTERN),build_map(L3_PATTERN)
    common=sorted(set(l2)&set(l3)); expected=list(range(START_YEAR,END_YEAR+1))
    problems=[]
    if common!=expected: problems.append(f"Paired years mismatch: {common}")
    rows=[]; ref=None
    for y in common:
        with rasterio.open(l2[y]) as s2,rasterio.open(l3[y]) as s3:
            if s2.count%len(L2_BAND_VARS): problems.append(f"{y} L2 band count invalid")
            if s3.count%len(L3_BAND_VARS): problems.append(f"{y} L3 band count invalid")
            n2=s2.count//len(L2_BAND_VARS); n3=s3.count//len(L3_BAND_VARS); exp=len(phases_for_year(y))
            if n2!=exp: problems.append(f"{y} L2 has {n2}; expected {exp}")
            if n3!=exp: problems.append(f"{y} L3 has {n3}; expected {exp}")
            if n2!=n3: problems.append(f"{y} L2/L3 mismatch")
            if ref is None: ref={"crs":s3.crs,"w":s3.width,"h":s3.height,"tr":tuple(s3.transform)}
            else:
                if s3.crs!=ref["crs"]: problems.append(f"{y} CRS mismatch")
                if (s3.width,s3.height)!=(ref["w"],ref["h"]): problems.append(f"{y} shape mismatch")
                if not nearly(tuple(s3.transform),ref["tr"]): problems.append(f"{y} transform mismatch")
            a,b=sampled_stats(s2),sampled_stats(s3)
            rows.append({"year":y,"l2_file":str(l2[y]),"l3_file":str(l3[y]),"l2_steps":n2,"l3_steps":n3,
                         "configured_phases":f"{phases_for_year(y)[0]}-{phases_for_year(y)[-1]}",
                         "l2_crs":str(s2.crs),"l3_crs":str(s3.crs),
                         "l2_width":s2.width,"l2_height":s2.height,"l3_width":s3.width,"l3_height":s3.height,
                         "l2_sample_min":a[0],"l2_sample_max":a[1],"l2_sample_mean":a[2],"l2_valid_fraction":a[3],
                         "l3_sample_min":b[0],"l3_sample_max":b[1],"l3_sample_mean":b[2],"l3_valid_fraction":b[3]})
    with open(MANIFEST_CSV,"w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    with open(AUDIT_TXT,"w",encoding="utf-8") as f:
        f.write("Weihe resilience pipeline — QC partial-year aware\n"+"="*70+"\n")
        f.write(f"Paired years: {common}\nReference CRS: {ref['crs']}\nReference shape: {ref['h']} x {ref['w']}\n")
        f.write(f"2000 configured phases: {phases_for_year(2000)[0]}-{phases_for_year(2000)[-1]} ({len(phases_for_year(2000))} steps)\n")
        f.write("2001-2024: phases 1-46\n\n")
        if problems:
            f.write("HARD QC FAILURES\n"+"\n".join("- "+p for p in problems))
        else:
            f.write("HARD QC: PASS\n")
            f.write("2000 is treated as an explicit partial year and is never padded.\n")
    print("[OK] Manifest:",MANIFEST_CSV); print("[OK] Audit:",AUDIT_TXT)
    if problems:
        for p in problems: print(" -",p)
        raise RuntimeError("Step1 failed")
    print("[PASS] Step1 passed.")

if __name__=="__main__":
    main()
