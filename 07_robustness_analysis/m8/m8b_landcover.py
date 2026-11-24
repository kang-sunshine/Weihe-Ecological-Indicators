# -*- coding: utf-8 -*-
from pathlib import Path
import json,zipfile,hashlib,shutil,time
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
O=Path(r'.\EI_Reviewer1_Major8_Revision');R=O.parent
GROUPS={1:'Forest',2:'Grassland_Shrubland',3:'Cropland'}
def grouping(mode,crop14=False):
 out=np.full(mode.shape,4,np.uint8);out[mode==0]=0;out[np.isin(mode,[1,2,3,4,5])]=1;out[np.isin(mode,[6,7,10])]=2;out[mode==12]=3
 if crop14:out[mode==14]=3
 return out
def main():
 start=time.time()
 for folder in ['inputs/M8B_Landcover','audit/M8B','supplement/M8B','cache/M8B','paired/M8B','archive/M8A_Before_Landcover']:(O/folder).mkdir(parents=True,exist_ok=True)
 arc=O/'archive/M8A_Before_Landcover'
 for p in list(O.glob('M8_*.csv'))+list(O.glob('M8_*.md'))+[O/'M8_QC.txt',O/'README.md']:
  if p.is_file() and not (arc/p.name).exists():shutil.copy2(p,arc/p.name)
 for n in ['Verification_Checks.csv','Delivery_Manifest.csv']:
  if not (arc/n).exists():shutil.copy2(O/'audit'/n,arc/n)
 before=pd.read_csv(O/'audit/Inputs_Before.csv');meta=[]
 for z in before.itertuples(index=False):
  p=Path(z.Path);s=p.stat();assert s.st_size==z.Bytes and s.st_mtime_ns==int(z.Mtime_ns),(p,'changed sinceM8A');meta.append(dict(Path=str(p),Bytes=s.st_size,Mtime_ns=s.st_mtime_ns))
 pd.DataFrame(meta).to_csv(O/'audit/M8B/Inputs_Before.csv',index=False)
 frozen=[]
 for folder in ['models','oof','splits']:
  for p in sorted((O/folder).glob('*')):
   if p.is_file():s=p.stat();frozen.append(dict(File=p.relative_to(O).as_posix(),Bytes=s.st_size,Mtime_ns=s.st_mtime_ns,SHA256=hashlib.sha256(p.read_bytes()).hexdigest()))
 for p in [O/'M8_Terrain_Ablation.csv',O/'paired/Frozen_Model_Pixels_With_Terrain.parquet',O/'paired/Full_Model_Greening_Contrasts.parquet']:
  s=p.stat();frozen.append(dict(File=p.relative_to(O).as_posix(),Bytes=s.st_size,Mtime_ns=s.st_mtime_ns,SHA256=hashlib.sha256(p.read_bytes()).hexdigest()))
 pd.DataFrame(frozen).to_csv(O/'audit/M8B/Frozen_Terrain_Model_Manifest.csv',index=False)
 cfg=dict(Date='2026-11-11',Main_Year=2001,Check_Years=[2001,2010,2020],Groups=GROUPS,Main_IGBP={'Forest':[1,2,3,4,5],'Grassland_Shrubland':[6,7,10],'Cropland':[12]},Other_Classes_Separate=[8,9,11,13,14,15,16,17],Mixed='purity<0.5 flag retained;0.75purity sensitivity separately',Consistency='3period exactclass and broadgroup consistency, not all-year stability',Minimum_Bin_N=500,Minimum_Fit_Bins=12,Minimum_Tail_Bins=4,Minimum_Total_50km_Blocks=10,Minimum_Green_Group_Pixels=100,Minimum_Green_Group_50km_Blocks=10,Minimum_Endpoint_Arm_Pixels=50,Minimum_Endpoint_Arm_Blocks=5,Seed=20261111,Spatial_Block_km=50,Bootstrap_N=350,Minimum_Valid_Bootstrap=280,Original_Bin_Design='original hazard-specific80quantile sampled clean-severity edges, no type-specific rebinning',Candidate_Design='all archived original candidateSknots inside exact original range, >=4 supported bins each side; no expansion',AIC='linear k2, segmented k4 for both fixed and search to preserve historical comparability; fixed-knot k3 sensitivity also reported',Response='original2000-2020 standardization and historical L5.3 damage/legacy as M1/M6; not redefined M5 recovery',No_New_Model_Training=True,Greening_Main_Endpoints='original basin P20/P80 for exact full/OOF reuse; within-type P20/P80 from saved frozen models only, no refits',Support_Caveat='reconstructed0.1degree ERA5 distributed supports, not proof of native Gaussian independent nodes')
 (O/'code/M8B_configuration.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf8')
 zipfile_path=O/'drive-download-20261111T065447Z-1-001.zip';manifest=[]
 with zipfile.ZipFile(zipfile_path) as z:
  expected={f'M8_LC_{y}_IGBP_Fractions_1km.tif' for y in cfg['Check_Years']};assert {p.filename for p in z.infolist()}==expected
  for n in sorted(expected):
   b=z.read(n);p=O/'inputs/M8B_Landcover'/n
   if p.exists():assert hashlib.sha256(p.read_bytes()).digest()==hashlib.sha256(b).digest()
   else:p.write_bytes(b)
   manifest.append(dict(File=n,Bytes=len(b),SHA256=hashlib.sha256(b).hexdigest(),ZIP=zipfile_path.name))
 pd.DataFrame(manifest).to_csv(O/'audit/M8B/Export_File_Manifest.csv',index=False)
 grid=json.loads((O/'audit/Target_Grid.json').read_text());tr=rasterio.Affine(*grid['Transform'][:6]);shape=tuple(grid['Shape']);domain=np.load(R/'EI_Reviewer1_Major4_Revision/cache/Climate_Valid_Mask.npy');area=np.load(O/'cache/Pixel_Area_km2.npy');d=pd.read_parquet(O/'paired/Frozen_Model_Pixels_With_Terrain.parquet');all_mode=[];all_group=[];all_purity=[];valids=[];qc=[];prov=[];r,c=np.where(domain);pixel=pd.DataFrame(dict(row=r,col=c,Pixel_Area_km2=area[domain]))
 names=['igbp_mode']+[f'igbp_{k}_fraction' for k in range(1,18)]+['valid_fraction','purity_among_valid','missing_fraction']
 for yr in cfg['Check_Years']:
  p=O/'inputs/M8B_Landcover'/f'M8_LC_{yr}_IGBP_Fractions_1km.tif'
  with rasterio.open(p) as ds:
   assert ds.crs==rasterio.crs.CRS.from_epsg(4326);assert list(ds.descriptions)==names;assert ds.count==21;assert np.allclose([ds.transform.a,ds.transform.b,ds.transform.d,ds.transform.e],[tr.a,tr.b,tr.d,tr.e],rtol=0,atol=1e-12)
   offc=(tr.c-ds.transform.c)/tr.a;offr=(tr.f-ds.transform.f)/tr.e;assert abs(offc-round(offc))<1e-7 and abs(offr-round(offr))<1e-7;offc=int(round(offc));offr=int(round(offr));assert offc>=0 and offr>=0 and offc+shape[1]<=ds.width and offr+shape[0]<=ds.height
   a=ds.read(window=Window(offc,offr,shape[1],shape[0]),masked=True).filled(np.nan);prov.append(dict(Year=yr,File=str(p),Product='MODIS/061/MCD12Q1',Band='LC_Type1',Classification='IGBP1-17',Export_Script='code/M8_Minimal_GEE_Landcover_2001_2010_2020.js',Source_CRS=str(ds.crs),Source_Height=ds.height,Source_Width=ds.width,Target_Height=shape[0],Target_Width=shape[1],Offset_Rows=offr,Offset_Cols=offc,Transform=json.dumps(list(ds.transform)),NoData=ds.nodata,Bands=';'.join(ds.descriptions),Status='USER_EXPORTED_FILE_CONTENT_AND_GRID_VERIFIED',Interpolation='none; exact integer-offset window onto frozen grid; area fractions already aggregated inGEE',Limit='source collection attribution from user-run supplied export code; no independent GEE server manifest supplied'))
  mode0=a[0];f=a[1:18];vf=a[18];pur=a[19];missing=a[20];present=np.isfinite(mode0);ok=present&np.isfinite(f).all(axis=0)&np.isfinite(vf)&(vf>0);assert np.all(np.abs(mode0[present]-np.rint(mode0[present]))<1e-6);mode=np.where(ok,np.rint(mode0),0).astype(np.uint8);assert set(np.unique(mode)).issubset(set(range(18)));assert np.nanmin(f)>=-1e-6 and np.nanmax(f)<=1+1e-6
  err=np.nanmax(np.abs(np.sum(f,axis=0)-vf));perr=np.max(np.abs(np.max(f[:,ok],axis=0)/vf[ok]-pur[ok]));merr=np.nanmax(np.abs(vf+missing-1));arg=np.argmax(np.where(np.isfinite(f),f,-1),axis=0)+1;ties=ok&(mode!=arg);different=np.zeros(shape,bool)
  if ties.any():
   flat=np.flatnonzero(ties);chosen=f.reshape(17,-1)[mode.ravel()[flat]-1,flat];different.ravel()[flat]=np.abs(chosen-f.max(axis=0).ravel()[flat])>1e-6
  assert err<1e-5 and perr<1e-5 and merr<1e-5 and not different[domain].any();group=grouping(mode);all_mode.append(mode);all_group.append(group);all_purity.append(pur);valids.append(ok)
  qc.append(dict(Year=yr,N_Domain_Pixels=int(domain.sum()),N_Valid_Class_Pixels=int((domain&ok).sum()),N_NoData_Or_ZeroValid=int((domain&~ok).sum()),N_Formal_Model_Pixels=len(d),N_Model_Class_Missing=int((~ok[d.row.to_numpy(int),d.col.to_numpy(int)]).sum()),N_Mixed_Purity_Lt0p5=int((domain&ok&(pur<.5)).sum()),Fraction_Sum_MaxError=err,Purity_MaxError=perr,Valid_Plus_Missing_MaxError=merr,N_Mode_Tie_Choices_Differ=int((ties&domain).sum()),N_Mode_Not_Maximum=int((different&domain).sum()),Grid_Aligned=True,Resampled=False))
  pixel[f'IGBP_{yr}']=mode[domain];pixel[f'Group_{yr}']=group[domain];pixel[f'Purity_{yr}']=pur[domain];pixel[f'Valid_Fraction_{yr}']=vf[domain]
  for k in range(17):pixel[f'{yr}_IGBP_{k+1}_Fraction']=f[k][domain]
  np.savez_compressed(O/'cache/M8B'/f'LC_{yr}.npz',Mode=mode,Group=group,Purity=pur,Valid=ok,Fractions=f,Valid_Fraction=vf)
 common=np.logical_and.reduce(valids);exact=common&(all_mode[0]==all_mode[1])&(all_mode[0]==all_mode[2]);broad=common&(all_group[0]==all_group[1])&(all_group[0]==all_group[2]);np.savez_compressed(O/'cache/M8B/Landcover_Cohorts.npz',Main_Group=all_group[0],Main_Mode=all_mode[0],Main_Purity=all_purity[0],Exact_ThreePeriod=exact,Broad_ThreePeriod=broad,Common_Valid=common)
 pixel['ThreePeriod_Exact_Class_Consistent']=exact[domain];pixel['ThreePeriod_Broad_Group_Consistent']=broad[domain];pixel.to_parquet(O/'paired/M8B/Landcover_ThreePeriod_Fractions.parquet',index=False)
 change=[]
 for ya,yb,ia,ib in [(2001,2010,0,1),(2010,2020,1,2),(2001,2020,0,2)]:
  for ca in range(18):
   for cb in range(18):
    m=domain&(all_mode[ia]==ca)&(all_mode[ib]==cb)
    if m.any():change.append(dict(Record_Type='EXACT_1km_IGBP_TRANSITION',Year_From=ya,Year_To=yb,IGBP_From=ca,IGBP_To=cb,N_Pixels=int(m.sum()),Area_km2=float(area[m].sum()),Fraction_Of_Domain_Area=float(area[m].sum()/area[domain].sum()),Status='COMPUTED',Limit='observed classification transition, not independently verified land-use conversion'))
 for gid,name in GROUPS.items():
  base=domain&(all_group[0]==gid)
  for label,m in [('EXACT_CLASS',exact),('BROAD_GROUP',broad)]:change.append(dict(Record_Type='THREE_PERIOD_CONSISTENCY',Vegetation=name,Consistency_Definition=label,Year_From=2001,Year_To=2020,N_Main_Pixels=int(base.sum()),N_Pixels=int((base&m).sum()),Fraction_Of_Main_Type=float((base&m).sum()/base.sum()),Area_km2=float(area[base&m].sum()),Status='COMPUTED',Limit='three-period classification consistency, not full2001-2024 annual stability'))
 pd.DataFrame(change).to_csv(O/'M8_Landcover_Change_Sensitivity.csv',index=False,encoding='utf-8-sig');pd.DataFrame(qc).to_csv(O/'audit/M8B/Landcover_Validation.csv',index=False);pd.DataFrame(prov).to_csv(O/'M8_Landcover_Provenance.csv',index=False,encoding='utf-8-sig');(O/'cache/M8B/LANDCOVER_READY.json').write_text(json.dumps(dict(N_Years=3,N_Domain_Pixels=int(domain.sum()),N_Common_Valid=int((common&domain).sum()),N_Exact_ThreePeriod=int((exact&domain).sum()),N_Broad_ThreePeriod=int((broad&domain).sum()),Seconds=time.time()-start),indent=2),encoding='utf8');print(pd.DataFrame(qc).to_string(index=False),flush=True);print('M8B LANDCOVER READY',round(time.time()-start,1),flush=True)
if __name__=='__main__':main()
