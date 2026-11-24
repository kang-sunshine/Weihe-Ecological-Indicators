#!/usr/bin/env python3
"""Figure 7 R4. Replots frozen, historical 20×20-pixel local legacy response.
No models are fitted; no values are interpolated or overwritten.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors, font_manager
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Patch

ROOT = Path(__file__).resolve().parent
HAZARDS = ('CDH','HEAT','HR')
NAMES = {'CDH':'Compound dry–hot','HEAT':'High temperature','HR':'Heavy rainfall'}
CATEGORIES = ('ACCELERATING','DECELERATING','NEAR_LINEAR','UNRESOLVED')
CAT_COLORS = {'ACCELERATING':'#C44B39','DECELERATING':'#37789C',
              'NEAR_LINEAR':'#BBAE91','UNRESOLVED':'#D9DEE3'}
BKG = '#F7F8FA'; GRID = '#D9DFE5'; FRAME = '#647281'; INK = '#000000'


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as fh:
        return list(csv.DictReader(fh))


def geo_polys(geometry):
    kind=geometry.get('type'); coords=geometry.get('coordinates',[])
    if kind=='Polygon':
        for ring in coords[:1]:
            if len(ring)>=3: yield np.asarray(ring, dtype=float)
    elif kind=='MultiPolygon':
        for p in coords:
            if p and len(p[0])>=3: yield np.asarray(p[0], dtype=float)
    else:
        raise ValueError(f'Unsupported GeoJSON type: {kind}')


def load_geometries(base):
    j=json.loads((base/'F7_Grid_Geometry.geojson').read_text(encoding='utf-8'))
    polygons={}
    for f in j['features']:
        key=int(f['properties']['Grid_ID'])
        polygons[key]=list(geo_polys(f['geometry']))
    bound=json.loads((base/'Basin_Boundary.geojson').read_text(encoding='utf-8'))
    boundary=[]
    for f in bound['features']:
        boundary.extend(list(geo_polys(f['geometry'])))
    if len(polygons)!=478 or not boundary:
        raise AssertionError('Expected 478 geometries and a nonempty basin outline')
    return polygons,boundary


def draw_base(ax, geometries, boundary, ext):
    ax.set_facecolor(BKG)
    base=[Polygon(poly,closed=True) for parts in geometries.values() for poly in parts]
    col=PatchCollection(base,facecolor=CAT_COLORS['UNRESOLVED'],edgecolor='white',
                        linewidths=.11,zorder=2)
    ax.add_collection(col)
    for p in boundary:
        ax.plot(p[:,0],p[:,1],color=FRAME,lw=.85,zorder=6)
    xmin,xmax,ymin,ymax=ext
    ax.set_xlim(xmin,xmax);ax.set_ylim(ymin,ymax)
    lat=(ymin+ymax)*.5
    ax.set_aspect(1/np.cos(np.deg2rad(lat)),adjustable='box')
    ax.set_xticks([104,106,108,110]); ax.set_yticks([34,35,36,37])
    ax.set_xticklabels([f'{v}°E' for v in [104,106,108,110]])
    ax.set_yticklabels([f'{v}°N' for v in [34,35,36,37]])
    ax.grid(True,ls=(0,(2,3)),lw=.45,color=GRID,zorder=0)
    ax.tick_params(axis='both',which='both',labelsize=12.8,length=0,pad=4,colors='#000000',labelcolor='#000000')
    for tick_label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        tick_label.set_color('#000000')
        tick_label.set_fontfamily(plt.rcParams['font.family'][0])
    for sp in ax.spines.values():sp.set_visible(False)


def draw_values(ax, rows, geometries, norm, cmap):
    patches=[]; vals=[]
    for r in rows:
        if r['Support_flag']!='True': continue
        key=int(r['Grid_ID']); v=float(r['Main_response_value'])
        for poly in geometries[key]:
            patches.append(Polygon(poly,closed=True)); vals.append(v)
    collection=PatchCollection(patches,cmap=cmap,norm=norm,edgecolor='white',
                                linewidths=.11,zorder=3)
    collection.set_array(np.array(vals))
    ax.add_collection(collection)
    return collection


def draw_classes(ax, rows, geometries):
    for cl in CATEGORIES[:3]:
        polygons=[]
        for r in rows:
            if r['Response_pattern_class']!=cl:continue
            if r['Support_flag']!='True':raise AssertionError('unsupported classified block')
            for poly in geometries[int(r['Grid_ID'])]:polygons.append(Polygon(poly,closed=True))
        if polygons:
            ax.add_collection(PatchCollection(polygons,facecolor=CAT_COLORS[cl],
                edgecolor='white',linewidths=.11,zorder=3))


def counts(rows):
    a=len(rows); eligible=sum(r['sampling_adequate']=='True' for r in rows)
    sup=[r for r in rows if r['Support_flag']=='True']
    types=Counter(r['Response_pattern_class'] for r in sup)
    return a,eligible,len(sup),types


def qc_check(data,geom):
    exp={'CDH':(421,131,103,28,0),
         'HEAT':(464,325,258,64,3),
         'HR':(463,240,118,121,1)}
    lines=[]
    for hz in HAZARDS:
        rows=[r for r in data if r['Hazard']==hz]
        ids=[int(r['Grid_ID']) for r in rows]
        if len(ids)!=478 or len(set(ids))!=478 or set(ids)!=set(geom):
            raise AssertionError(f'{hz} incorrect map-grid inventory')
        total,adequate,supported,c=counts(rows)
        result=(adequate,supported,c['ACCELERATING'],c['DECELERATING'],c['NEAR_LINEAR'])
        if result!=exp[hz]:raise AssertionError(f'{hz} QC mismatch: {result}')
        for r in rows:
            valid=r['Support_flag']=='True'
            if valid != (r['Main_response_value']!=''):
                raise AssertionError(f'{hz}: supported/response missing inconsistency')
            if valid:
                delta=float(r['delta_beta'])
                cat='ACCELERATING' if delta>.01 else 'DECELERATING' if delta<-.01 else 'NEAR_LINEAR'
                if r['Response_pattern_class']!=cat:
                    raise AssertionError(f'{hz}: response category inconsistency')
        lines.append(f'{hz}: candidate={total}; eligible={adequate}; strictly supported={supported}; '
                     f'accelerating={c["ACCELERATING"]}; decelerating={c["DECELERATING"]}; '
                     f'near-linear={c["NEAR_LINEAR"]}; unresolved={total-supported}; '
                     f'accelerating/supported={100*c["ACCELERATING"]/supported:.2f}%; '
                     f'accelerating/eligible={100*c["ACCELERATING"]/adequate:.2f}%; '
                     f'accelerating/all478={100*c["ACCELERATING"]/total:.2f}%')
    return lines


def plot(base,out, exact_font=False):
    data=read_csv(base/'F7_Map_Main.csv')
    geom,boundary=load_geometries(base)
    qc=qc_check(data,geom)
    fonts={f.name for f in font_manager.fontManager.ttflist}
    tnr='Times New Roman' in fonts
    if exact_font and not tnr:
        raise RuntimeError('Times New Roman not installed or not found; run on your Windows PC')
    font='Times New Roman' if tnr else 'Liberation Serif'
    plt.rcParams.update({'font.family':font,'pdf.fonttype':42,'ps.fonttype':42,
                         'font.size':12.7,'text.color':INK,'axes.labelcolor':INK,
                         'savefig.facecolor':'white','mathtext.fontset':'stix'})
    with (base/'F7_ColorScale_Metadata.csv').open(encoding='utf-8') as fh:
        meta=next(csv.DictReader(fh))
    rlim=float(meta['Display_Max'])
    norm=colors.TwoSlopeNorm(vmin=-rlim,vcenter=0,vmax=rlim)
    cmap=plt.get_cmap('RdBu_r')
    fig=plt.figure(figsize=(12.6,14.5),facecolor='white')
    # R4: retain R3 row separation, reduce horizontal gutter from .27 to .19.
    # Move bottom legend elements slightly up without changing the map axes.
    gs=fig.add_gridspec(4,2,height_ratios=[1,1,1,.12],left=.071,right=.964,
                        top=.940,bottom=.066,hspace=.46,wspace=.06)
    fig.canvas.draw()
    letters='abcdef'; all_rows={h:[r for r in data if r['Hazard']==h] for h in HAZARDS}
    lastcol=None
    for i,hz in enumerate(HAZARDS):
        a_left=fig.add_subplot(gs[i,0]);a_right=fig.add_subplot(gs[i,1])
        r=all_rows[hz]; n,elig,sup,c=counts(r)
        # Expand outer margin just enough to avoid clipped 104°E/110°E labels.
        extent=(103.82,110.52,33.46,37.54)
        for ax in (a_left,a_right):draw_base(ax,geom,boundary,extent)
        lastcol=draw_values(a_left,r,geom,norm,cmap)
        draw_classes(a_right,r,geom)
        a_left.set_title(f'({letters[i]}) {NAMES[hz]}  |  Post-reference slope',
                         loc='left',fontsize=15.5,fontweight='bold',y=1.155,pad=0)
        a_right.set_title(f'({letters[i+3]}) {NAMES[hz]}  |  Slope-change class',
                          loc='left',fontsize=15.5,fontweight='bold',y=1.155,pad=0)
        # R2: place summaries above their axes, never over the basin geometry.
        # Denominator meanings are unchanged (eligible vs strictly supported).
        a_left.text(.015,1.035,f'Supported: {sup}/{elig} eligible',
                    transform=a_left.transAxes,ha='left',va='bottom',
                    fontsize=12.3,color='#000000',fontweight='bold',
                    clip_on=False,zorder=12)
        a_right.text(.985,1.035,
            f'Accelerating: {c["ACCELERATING"]}/{sup} supported ({100*c["ACCELERATING"]/sup:.1f}%)',
            transform=a_right.transAxes,ha='right',va='bottom',
            fontsize=12.3,fontweight='bold',color='#000000',
            clip_on=False,zorder=12)
    # Separate legend rows prevent collisions with maps. Left is continuous; right is categorical.
    foot_l=fig.add_subplot(gs[3,0]);foot_r=fig.add_subplot(gs[3,1])
    for ax in (foot_l,foot_r):ax.set_axis_off()
    cb=fig.colorbar(lastcol,ax=foot_l,orientation='horizontal',location='bottom',
                    shrink=.84,fraction=.35,pad=.19,aspect=21,extend='both')
    cb.set_label('Post-reference legacy slope (anomaly-days / percentile point)',
                 fontsize=12.3,labelpad=5,color='black')
    cb.ax.tick_params(labelsize=11.7,length=2,pad=2,colors='black',labelcolor='black')
    # Raise the colorbar itself by 1.2% of figure height, preserving width/height.
    cpos=cb.ax.get_position()
    cb.ax.set_position([cpos.x0,cpos.y0+.012,cpos.width,cpos.height])
    legends=[Patch(facecolor=CAT_COLORS[cl],edgecolor='none',label=lb) for cl,lb in [
      ('ACCELERATING','Positive Δβ'),('DECELERATING','Negative Δβ'),
      ('NEAR_LINEAR','Small |Δβ|'),('UNRESOLVED','Unresolved')]]
    # Raise the entire four-key categorical legend by the same small amount.
    rpos=foot_r.get_position()
    foot_r.set_position([rpos.x0,rpos.y0+.012,rpos.width,rpos.height])
    foot_r.legend(handles=legends,loc='upper center',bbox_to_anchor=(.50,.95),
                   ncol=2,fontsize=12.5,frameon=False,handlelength=1.65,
                   columnspacing=1.6,borderpad=.02,labelspacing=.45)
    out.mkdir(parents=True,exist_ok=True)
    pdf=out/'Figure7_R4_SpatialLegacy.pdf';png=out/'Figure7_R4_SpatialLegacy.png'
    fig.savefig(pdf,facecolor='white')
    fig.savefig(png,dpi=500,facecolor='white')
    plt.close(fig)
    txt=['FIGURE 7 R4 PLOTTING QC',
         'R4-only changes: wspace .27 -> .19; black tick labels; colorbar/legend raised by .012 figure height',
         f'Status: {"PASS" if tnr else "DATA_PASS_FONT_PREVIEW_ONLY"}',
         f'Font requested: Times New Roman; exact font available: {tnr}',
         f'Actual rendering font family: {font}',
         'Spatial geometry: historical 20x20 analytical pixel units, not equal-area 20km blocks',
         'Primary mapped value: beta_post from original L5.5C; no refitting',
         'Category classification: delta_beta tolerance 0.01; not significance',
         'Basin references: historical sampled severity-percentile candidates; not Figure6 full ECDF',
         f'Continuous color rendering range: ±{rlim:.6f}; source values unchanged',
         'Unresolved geometry retained as grey; counts are blocks, not area percentage',
         'Source provenance and M8 contextual analysis NOT merged into Figure 7',*qc]
    (out/'Figure7_R4_PlotQC.txt').write_text('\n'.join(txt)+'\n',encoding='utf-8')
    caption=(
      'Figure 7. Spatial heterogeneity in post-event ecological legacy responses across the '
      'Weihe River Basin. (a–c) Local post-reference legacy slopes (βpost) for compound '
      'dry–hot (CDH), high-temperature (HEAT), and heavy-rainfall (HR) events. '
      '(d–f) Corresponding response classes based on the change in slope (Δβ) '
      'across the fixed basin-scale legacy candidate. Local analyses use historical '
      '20×20-pixel units and the historical sampled severity-percentile reference, '
      'which differs from the complete-event ECDF used for Figure 6. Only strictly '
      'model-supported units are colored; grey units were unsupported or unresolved '
      'and do not imply zero effect. Positive or negative slope change is determined '
      'by Δβ > 0.01 or Δβ < −0.01; |Δβ| ≤ 0.01 denotes a small numerical slope change, '
      'not a statistical non-significance test. The supported/eligible denominators '
      'were 131/421 (CDH), 325/464 (HEAT) and 240/463 (HR). Among supported units, '
      'accelerating classes comprised 103/131, 258/325 and 118/240; these correspond '
      'to 103/421, 258/464 and 118/463 of initially eligible units, respectively. '
      'These are fractions of analytical units, not geographic area percentages. '
      'The symmetric map color scale is display-limited; source βpost values are unchanged. '
      'The fixed basin references are model-dependent positions, not independently '
      'validated ecological tipping points. '
    )
    (out/'Figure7_R4_Caption.txt').write_text(caption+'\n',encoding='utf-8')
    print('OUTPUT:',pdf,png)
    print('FONT:',font,'EXACT:',tnr)
    print('\n'.join(qc))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Figure7 legacy spatial plots (no model training)')
    parser.add_argument('--data',type=Path,default=ROOT/'data')
    parser.add_argument('--out',type=Path,default=ROOT/'outputs')
    parser.add_argument('--exact-font',action='store_true',help='fail if Times New Roman is unavailable')
    a=parser.parse_args()
    plot(a.data,a.out,a.exact_font)
