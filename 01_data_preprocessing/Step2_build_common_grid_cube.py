# -*- coding: utf-8 -*-
import gc
import numpy as np
import pandas as pd
import rasterio
import rioxarray as rxr
import netCDF4 as nc
from tqdm import tqdm
from pipeline_config import *

def date_from_phase(year,phase):
    return pd.Timestamp(f"{year}-01-01")+pd.Timedelta(days=(phase-1)*8)

def main():
    df=pd.read_csv(MANIFEST_CSV)
    for r in df.itertuples(index=False):
        exp=len(phases_for_year(int(r.year)))
        if int(r.l2_steps)!=exp or int(r.l3_steps)!=exp: raise RuntimeError(f"{r.year} manifest mismatch")
    ref=rxr.open_rasterio(str(df.iloc[0]["l3_file"])).isel(band=0).squeeze(drop=True)
    crs=ref.rio.crs; transform=ref.rio.transform(); x=ref.x.values.astype(float); y=ref.y.values.astype(float); ny,nx=ref.shape
    dates=[]; phases=[]; years=[]
    for yy in df["year"].astype(int):
        ph=phases_for_year(yy); dates += [date_from_phase(yy,p) for p in ph]; phases += ph; years += [yy]*len(ph)
    nt=len(dates)
    if CUBE_NC.exists(): CUBE_NC.unlink()
    ds=nc.Dataset(CUBE_NC,"w",format="NETCDF4")
    ds.title="Weihe 1-km 8-day common analytical cube"
    ds.partial_2000_note=f"2000 explicitly mapped to phases {phases_for_year(2000)[0]}-{phases_for_year(2000)[-1]}; verify source metadata."
    ds.createDimension("time",nt); ds.createDimension("y",ny); ds.createDimension("x",nx)
    tv=ds.createVariable("time","i8",("time",)); tv.units="days since 2000-01-01 00:00:00"; tv.calendar="proleptic_gregorian"
    tv[:]=nc.date2num(pd.DatetimeIndex(dates).to_pydatetime(),tv.units,tv.calendar)
    ds.createVariable("phase","i2",("time",))[:]=np.asarray(phases,dtype=np.int16)
    ds.createVariable("year","i2",("time",))[:]=np.asarray(years,dtype=np.int16)
    xv=ds.createVariable("x","f8",("x",)); yv=ds.createVariable("y","f8",("y",)); xv[:]=x; yv[:]=y
    if crs.is_geographic: xv.units="degrees_east"; yv.units="degrees_north"
    crsv=ds.createVariable("crs","i4"); crsv.spatial_ref=crs.to_wkt(); crsv.crs_wkt=crs.to_wkt()
    crsv.GeoTransform=" ".join(map(str,[transform.c,transform.a,transform.b,transform.f,transform.d,transform.e]))
    names=RAW_L2_KEEP+RAW_L3_KEEP+["WUE","iWUE","uWUE"]; vo={}
    for n in names:
        vo[n]=ds.createVariable(n,"f4",("time","y","x"),fill_value=np.nan,zlib=True,complevel=4,shuffle=True,
                                chunksizes=(1,min(256,ny),min(256,nx)))
        vo[n].grid_mapping="crs"
    t0=0
    for row in tqdm(df.itertuples(index=False),total=len(df),desc="Step2"):
        yy=int(row.year); nsteps=len(phases_for_year(yy))
        d2=rxr.open_rasterio(str(row.l2_file)); d3=rxr.open_rasterio(str(row.l3_file))
        d2a=d2.rio.reproject_match(ref,resampling=rasterio.enums.Resampling.bilinear,nodata=np.nan)
        a2=np.asarray(d2a.values,dtype=np.float32); a3=np.asarray(d3.values,dtype=np.float32); a2[a2<-900]=np.nan; a3[a3<-900]=np.nan
        n2=len(L2_BAND_VARS); n3=len(L3_BAND_VARS); sl=slice(t0,t0+nsteps)
        for n in RAW_L2_KEEP: vo[n][sl]=a2[L2_BAND_VARS.index(n)::n2][:nsteps]
        for n in RAW_L3_KEEP: vo[n][sl]=a3[L3_BAND_VARS.index(n)::n3][:nsteps]
        gpp=a2[L2_BAND_VARS.index("GPP")::n2][:nsteps]; et=a2[L2_BAND_VARS.index("ET")::n2][:nsteps]; vpd=a3[L3_BAND_VARS.index("VPD")::n3][:nsteps]
        valid=np.isfinite(gpp)&np.isfinite(et)&np.isfinite(vpd)&(et>1e-6)&(vpd>=0)
        w=np.full_like(gpp,np.nan); iw=np.full_like(gpp,np.nan); uw=np.full_like(gpp,np.nan)
        w[valid]=gpp[valid]/et[valid]; iw[valid]=gpp[valid]*vpd[valid]/et[valid]; uw[valid]=gpp[valid]*np.sqrt(vpd[valid])/et[valid]
        vo["WUE"][sl]=w; vo["iWUE"][sl]=iw; vo["uWUE"][sl]=uw
        t0+=nsteps
        d2.close(); d3.close(); del d2,d3,d2a,a2,a3,gpp,et,vpd,w,iw,uw; gc.collect()
    ds.close(); ref.close()
    print("[PASS]",CUBE_NC); print("[INFO] time steps:",nt)

if __name__=="__main__":
    main()
