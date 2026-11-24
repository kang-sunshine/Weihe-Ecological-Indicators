# -*- coding: utf-8 -*-
"""Figure 3 reviewer revision R2 | Word-native width (15.49 cm).
Frozen 6-layer NC is read-only. All spatial arrays, coordinates, NaNs and
statistics remain unchanged. The original 2-by-3 panel and palette are retained; tighter layout with larger native maps, pale map backgrounds and fine dashed graticules.

IMPORTANT: Actual a/b climatological averaging years are UNVERIFIED.
No years are added to those panel titles and no source data are recomputed.
"""
from pathlib import Path
import hashlib
import numpy as np
# Reader supports the standard netCDF4/xarray stack and h5py-only environments.
try:
 import xarray as xr
except ImportError:
 xr=None
try:
 import h5py
except ImportError:
 h5py=None
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
try:
 import shapefile
except ImportError:
 shapefile=None

ROOT=Path(__file__).resolve().parent
NC=ROOT/'data'/'Figure3_SpatialMetrics_FINAL_LOCKED.nc'
BOUND=ROOT/'map_assets'/'Export_Output_Weihe.shp'
OUT=ROOT/'outputs';OUT.mkdir(exist_ok=True)
STEM=OUT/'Figure3_R2_WordSize'

mpl.rcParams.update({'font.family':'serif',
 'font.serif':['Times New Roman','Liberation Serif','Nimbus Roman','DejaVu Serif'],
 'mathtext.fontset':'stix','font.weight':'bold','axes.titleweight':'bold',
 'axes.labelweight':'bold','font.size':8.2,'axes.titlesize':9.4,
 'axes.labelsize':8.2,'xtick.labelsize':7.3,'ytick.labelsize':7.3,
 'axes.linewidth':.75,'axes.edgecolor':'black','pdf.fonttype':42,'ps.fonttype':42})

DRY=LinearSegmentedColormap.from_list('dry',['#F5F2E9','#E8D39B','#D99A62','#BB5A47','#7D2F36'])
WET=LinearSegmentedColormap.from_list('wet',['#F3EBD4','#C9DCCB','#87BEB4','#4C92A6','#285D7E'])
EVT=LinearSegmentedColormap.from_list('evt',['#FFF4DA','#F1CE82','#DD9555','#BE5A46','#8A3442','#50243B'])
VARIABLES=['VPD_GS_mean','RootSM_GS_mean','CDH_frequency_yr','CDH_mean_duration_days',
           'CDH_mean_severity','CDH_burden_yr']
TITLES=['Atmospheric dryness','Root-zone soil moisture','CDH recurrence',
        'Mean event duration','Mean event severity','Cumulative CDH burden']
UNITS=['VPD (kPa)',r'RootSM (m$^3$ m$^{-3}$)',r'Events yr$^{-1}$',
       'Days','Severity (dimensionless d)',r'Burden (dimensionless d yr$^{-1}$)']


def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
 return h.hexdigest()


def read_boundary():
 if shapefile is not None:
  rdr=shapefile.Reader(str(BOUND))
  return [sh.points[begin:end] for sh in rdr.shapes() for begin,end in zip(list(sh.parts),list(sh.parts)[1:]+[len(sh.points)])]
 # Fallback: geopandas if the user's environment does not have pyshp.
 import geopandas as gpd
 g=gpd.read_file(BOUND).to_crs(4326)
 return [list(poly.exterior.coords) for geom in g.geometry for poly in ([geom] if geom.geom_type=='Polygon' else list(geom.geoms))]


def main():
 try:
  with xr.open_dataset(NC) as ds:
   assert set(ds.data_vars)==set(VARIABLES)
   x=ds.x.values.copy();y=ds.y.values.copy()
   zall=[ds[v].values.copy() for v in VARIABLES]
 except (ImportError,ValueError):
  if h5py is None:raise RuntimeError('Need netCDF4/h5netcdf or h5py for frozen NC')
  with h5py.File(NC,'r') as ds:
   x=ds['x'][:];y=ds['y'][:]
   zall=[ds[v][:] for v in VARIABLES]
 assert np.all(np.diff(x)>0) and np.all(np.diff(y)<0)
 assert all(a.shape==(414,703) for a in zall)
 ref=np.isfinite(zall[0]);assert all(np.array_equal(np.isfinite(a),ref) for a in zall)
 assert int(ref.sum())==167546
 parts=read_boundary()
 dx=np.median(np.diff(x));dy=abs(np.median(np.diff(y)))
 extent=(float(x.min()-dx/2),float(x.max()+dx/2),float(y.min()-dy/2),float(y.max()+dy/2))
 fig=plt.figure(figsize=(6.10,4.65),facecolor='white')
 # Preserve 2×3 map composition, but remove repeated coordinate ticks.
 cols=[.060,.376,.692];wide=.291
 map_y=[.669,.248];map_h=.244
 cb_y=[.626,.205];cb_h=.017
 title_y=[.940,.518]
 cms=[DRY,WET,EVT,EVT,EVT,EVT]
 for idx,(z,title,unit,cm) in enumerate(zip(zall,TITLES,UNITS,cms)):
  row=idx//3;col=idx%3
  cx=cols[col]
  fig.text(cx,title_y[row],f'{chr(97+idx)}  {title}',ha='left',va='bottom',
           fontsize=8.9,fontweight='bold',color='black')
  ax=fig.add_axes([cx,map_y[row],wide,map_h])
  ax.set_facecolor('#F1F5F3')
  cmap=cm.copy(); cmap.set_bad((1,1,1,0))
  vals=z[np.isfinite(z)];vmin,vmax=np.percentile(vals,[2,98])
  im=ax.imshow(np.ma.masked_invalid(z),extent=extent,origin='upper',
               cmap=cmap,vmin=vmin,vmax=vmax,interpolation='nearest',rasterized=True,
               aspect='equal',zorder=1)
  for part in parts:
   poly=np.asarray(part)
   if len(poly)>1:ax.plot(poly[:,0],poly[:,1],c='#343B3D',lw=.53,zorder=4)
  ax.set_xlim(extent[:2]);ax.set_ylim(extent[2:])
  ax.set_xticks([105,107,109]);ax.set_yticks([34,35,36,37])
  ax.set_axisbelow(False)
  ax.grid(color='#92A7A4',alpha=.35,lw=.36,ls=(0,(2.4,3.2)),zorder=3)
  ax.tick_params(length=2.2,pad=1.5,labelbottom=row==1,bottom=row==1,
                 labelleft=col==0,left=col==0)
  ax.set_xticklabels(['105°E','107°E','109°E'] if row==1 else [])
  ax.set_yticklabels(['34°N','35°N','36°N','37°N'] if col==0 else [])
  for lab in ax.get_xticklabels()+ax.get_yticklabels():lab.set_fontweight('bold')
  for sp in ax.spines.values():sp.set_color('#404040');sp.set_linewidth(.65)
  cax=fig.add_axes([cx,cb_y[row],wide,cb_h])
  cb=fig.colorbar(im,cax=cax,orientation='horizontal')
  cb.locator=MaxNLocator(nbins=4);cb.update_ticks()
  cb.ax.tick_params(labelsize=7.2,length=1.8,pad=1.3)
  for lab in cb.ax.get_xticklabels():lab.set_fontweight('bold')
  cb.outline.set_linewidth(.5)
  fig.text(cx+wide/2,cb_y[row]-.034,unit,ha='center',va='top',fontsize=7.6,fontweight='bold')

 fig.canvas.draw()
 for ext in ['png','pdf','svg']:
  fig.savefig(str(STEM)+'.'+ext,dpi=600 if ext=='png' else 300,facecolor='white')
 plt.close(fig)
 summary=['Figure 3 R2 Word-size | Six-layer frozen inputs (V8) retained',
          'Physical canvas 6.10 × 4.65 in, 2×3 original layout, enlarged map axes.',
          'Figure 3(a,b) actual source averaging years: UNVERIFIED; no invented dates.',
          'NC coordinates, finite masks, all six arrays untouched; display limits are P2/P98.',
          'Original palette retained; pale map background and fine graticules; no spatial smoothing, interpolation, new events or recalculation.',
          f'NC SHA256 {sha(NC)}',f'Boundary SHA256 {sha(BOUND)}']
 for name,z in zip(VARIABLES,zall):
  finite=z[np.isfinite(z)];summary.append(f'{name}: n={len(finite)}, mean={np.mean(finite):.9f}, P2/P98={np.percentile(finite,2):.9f}/{np.percentile(finite,98):.9f}')
 (OUT/'Figure3_R2_QC.txt').write_text('\n'.join(summary),encoding='utf-8')
 print('\n'.join(summary))

if __name__=='__main__':main()
