# -*- coding: utf-8 -*-
"""Figure 9 R9.3 | Low-saturation diverging colour scale and wider column gap.

Uses the unchanged R8/R5 frozen Figure 9 inputs; no model refitting, cell
resampling, spatial smoothing, or reclassification. The raw signed B values
(including numerical zeros) are mapped with a LOW-SATURATION DIVERGING scale: 
negative = muted warm, zero = light neutral, positive = muted cool. The
colourbar has a clearly marked zero; the frozen sign-share counts are shown.

Expected neighbouring files (not included in the earlier R8 ZIP):
  data/F9_A_OOF.csv.gz, data/F9_BD_PixelMaps.csv.gz
Other supporting inputs are supplied in the R8 ZIP.

Run in EI_Figure9_R8_WordSize/ (alongside data/ and map_assets/):
  python Figure9_R9_3_SoftDiverging_Final.py --word-preview
  python Figure9_R9_3_SoftDiverging_Final.py --exact-font

R9.3 keeps the R9.2 layout and frozen scientific content; it changes only
the default map colour scale to a low-saturation diverging palette while
retaining the widened inter-column gap introduced in R9.2.
Output figure width = 14.40 cm, measured Word text block = 14.65 cm.
Set the picture width to 14.40 cm in Word (lock aspect ratio).
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import FuncNorm, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
# R8 ZIP has a single data/map-assets directory. Search only explicit nearby roots,
# never silently choose arbitrary files from an unrelated project.
ROOT_CHOICES = [HERE, HERE / 'EI_Figure9_R8_WordSize', HERE.parent / 'EI_Figure9_R8_WordSize']
ROOT = next((p for p in ROOT_CHOICES if (p/'data'/'F9_A_Metrics.csv').exists()
             and (p/'map_assets'/'F9_Grid_Reference.json').exists()), HERE)
DATA, MAP, OUTPUT = ROOT/'data', ROOT/'map_assets', ROOT/'outputs'
OUTPUT.mkdir(exist_ok=True, parents=True)
FILES = {
    'oof':DATA/'F9_A_OOF.csv.gz', 'metrics':DATA/'F9_A_Metrics.csv',
    'pixel':DATA/'F9_BD_PixelMaps.csv.gz', 'curve':DATA/'F9_C_HCL_Curve.csv',
    'peak':DATA/'F9_C_PeakContrast.csv', 'grid':MAP/'F9_Grid_Reference.json',
}
# Native-frozen Table S6A: common basin-wide P20–P80 interval, 50-km-block CI.
VEG = [
    ('Grassland /\nshrubland', 0.0277, 0.0167, 0.0413, 67002, True),
    ('Cropland',              0.0015, 0.0003, 0.0028, 51042, True),
    ('Forest',                0.0008,-0.0003, 0.0021, 16833, False),
]
INK = '#111111'
GRID = '#CBD4D5'
DARK_TEAL = '#26776D'
ORANGE = '#BA5948'
# All seven control colours belong to the same subdued hue family. Critically,
# the midpoint is a MEDIUM chromatic colour, not white or grey.
PALETTES = {
    'teal': ['#E0EAE2','#C5DDD0','#A4C9BA','#83B4A5','#5C988B','#38796F','#12584F'],
    'blue': ['#E0EAF0','#C5D8E6','#A9C7D9','#8CB4CC','#6795B5','#3F7199','#174B76'],
    'clay': ['#EADFD5','#DFC6B5','#D3AC95','#C28D70','#A97359','#89523F','#683D30'],
    'softdiv': ['#B98C78','#C9A796','#DCC6B9','#EEE4DD','#D8E2E1','#B7CFCD','#88B2AE','#548A87','#2F6662'],
}
FIG_W_CM = 14.40
FIG_H_CM = 11.80


def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for part in iter(lambda:f.read(2**20),b''):h.update(part)
    return h.hexdigest()


def configure(exact_font=False):
    fonts=['Times New Roman'] if exact_font else ['Times New Roman','Liberation Serif','Nimbus Roman','DejaVu Serif']
    if exact_font:
        from matplotlib.font_manager import FontProperties,findfont
        findfont(FontProperties(family='Times New Roman'), fallback_to_default=False)
    mpl.rcParams.update({
        'font.family':'serif','font.serif':fonts, 'mathtext.fontset':'stix',
        'font.size':9.4,'font.weight':'normal',
        'axes.titlesize':10.5,'axes.titleweight':'bold',
        'axes.labelsize':9.3,'axes.labelweight':'normal',
        'xtick.labelsize':8.8,'ytick.labelsize':8.8,'legend.fontsize':8.7,
        'axes.edgecolor':'#48565A','axes.linewidth':.78,
        'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
        'xtick.major.size':2.9,'xtick.major.width':.75,
        'ytick.major.size':2.9,'ytick.major.width':.75,
        'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white',
    })


def load():
    missing=[str(p) for p in FILES.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError('Figure 9 frozen input(s) missing:\n  '+'\n  '.join(missing)+
            '\nThe R8 ZIP does not contain F9_A_OOF.csv.gz or F9_BD_PixelMaps.csv.gz; '
            'restore the SAME frozen source CSVs used for R8. Do not synthesize replacements.')
    oof=pd.read_csv(FILES['oof']);metrics=pd.read_csv(FILES['metrics'])
    pix=pd.read_csv(FILES['pixel']);curve=pd.read_csv(FILES['curve'])
    peak=pd.read_csv(FILES['peak'])
    with open(FILES['grid'],encoding='utf-8') as f:grid=json.load(f)
    pooled=metrics.loc[metrics.Scope.eq('POOLED')].iloc[0]
    assert len(oof)==len(pix)==153932 and len(curve)==15
    assert np.isclose(float(pooled.R2),0.393027652877842,atol=1e-7)
    assert pix.Pixel_ID.is_unique and oof.Pixel_ID.is_unique
    assert np.array_equal(np.sort(pix.Pixel_ID.to_numpy()),np.sort(oof.Pixel_ID.to_numpy()))
    classes=pix.B_Sign_Class.value_counts().to_dict()
    assert classes=={'Positive':71781,'Numerical zero':64568,'Negative':17583},classes
    assert np.all(np.isfinite(pix.B))
    assert np.all((pix.loc[pix.B_Sign_Class.eq('Numerical zero'),'B'].abs()<=1e-12))
    assert np.all((pix.loc[pix.B_Sign_Class.eq('Positive'),'B']>1e-12))
    assert np.all((pix.loc[pix.B_Sign_Class.eq('Negative'),'B']<-1e-12))
    assert np.all((curve.Benefit>=curve.Bootstrap50_CI_lower) &
                  (curve.Benefit<=curve.Bootstrap50_CI_upper))
    return oof,pooled,pix,curve,peak,grid,classes


def map_grid(pix,grid):
    rows,cols=map(int,grid['Grid_shape'])
    b=np.full((rows,cols),np.nan)
    rr=pix['row'].to_numpy(int);cc=pix['col'].to_numpy(int)
    assert np.all((rr>=0)&(rr<rows)&(cc>=0)&(cc<cols))
    b[rr,cc]=pix.B.to_numpy(float)
    x=np.asarray(grid['X_centres'],float);y=np.asarray(grid['Y_centres'],float)
    dx=np.abs(np.median(np.diff(x)));dy=np.abs(np.median(np.diff(y)))
    extent=(float(x.min()-dx/2),float(x.max()+dx/2),
            float(y.min()-dy/2),float(y.max()+dy/2))
    return b,extent,x,y


def signed_norm(vmin,vmax,linear=.004):
    assert vmin<0<vmax
    neg=np.arcsinh(-vmin/linear);pos=np.arcsinh(vmax/linear)
    def f(values):
        a=np.asarray(values)
        return np.where(a<0,.5-.5*np.arcsinh(np.maximum(-a,0)/linear)/neg,
                        .5+.5*np.arcsinh(np.maximum(a,0)/linear)/pos)
    def inv(values):
        a=np.asarray(values)
        return np.where(a<.5,-linear*np.sinh((.5-a)*2*neg),
                        linear*np.sinh((a-.5)*2*pos))
    return FuncNorm((f,inv),vmin=vmin,vmax=vmax)


def panel_title(fig,ax,letter,title):
    p=ax.get_position(original=True)
    fig.text(p.x0,p.y1+.017,f'{letter}  {title}',ha='left',va='bottom',
             fontsize=10.15,fontweight='bold',color=INK)


def panel_a(ax,oof,pooled):
    xx=oof.Observed_Residual_Damage.to_numpy(float)
    yy=oof.OOF_Prediction.to_numpy(float)
    lo,hi=np.percentile(np.r_[xx,yy],[.2,99.8]);lo-=.025;hi+=.025
    cm=LinearSegmentedColormap.from_list('oof_density',
            ['#EFF4F4','#C7DDE0','#8DBCC3','#4B92A0','#225D71'])
    ax.hexbin(xx,yy,gridsize=53,extent=[lo,hi,lo,hi],mincnt=1,
              bins='log',cmap=cm,linewidths=0,rasterized=True,zorder=1)
    ax.plot([lo,hi],[lo,hi],color='#454F53',lw=1.0,ls=(0,(4,3)),zorder=3)
    slope,inter=np.polyfit(xx,yy,1)
    ax.plot([lo,hi],slope*np.array([lo,hi])+inter,color=ORANGE,lw=1.35,zorder=4)
    ax.set_xlim(lo,hi);ax.set_ylim(lo,hi)
    ax.set_xlabel('Observed residual damage',labelpad=2.5)
    ax.set_ylabel('OOF-predicted residual damage',labelpad=2.5)
    ax.text(.038,.965, f'R² = {pooled["R2"]:.3f}\nRMSE = {pooled["RMSE"]:.3f}\nn = {len(oof):,}',
            transform=ax.transAxes,ha='left',va='top',fontsize=8.45,color=INK,
            linespacing=1.25,zorder=5,
            bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=2.5))
    handles=[Line2D([0],[0],color='#454F53',lw=1,ls=(0,(4,3)),label='1:1'),
             Line2D([0],[0],color=ORANGE,lw=1.35,label='OLS')]
    ax.legend(handles=handles,frameon=False,loc='lower right',ncol=2,
              fontsize=8.2,handlelength=1.35,columnspacing=.6,borderpad=.1)
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(direction='out',pad=2)


def panel_b(ax,b,pix,ext,x,y,counts,palette):
    """Map the signed B field directly; 41.95% numerical zeros are NOT painted grey.

    Low-saturation diverging family: negative = muted warm, zero = light
    neutral, positive = muted cool. Values and exact class labels remain
    untouched.
    """
    cmap=LinearSegmentedColormap.from_list('B_'+palette,PALETTES[palette],N=256)
    cmap.set_bad((1,1,1,0))
    norm=signed_norm(float(pix.B.min()),float(pix.B.max()))
    img=ax.imshow(np.ma.masked_invalid(b),extent=ext,origin='upper',
                  interpolation='nearest',cmap=cmap,norm=norm,
                  rasterized=True,zorder=2)
    # Draw a basin boundary using the same native occupancy mask; do not alter cells.
    available=np.isfinite(b)
    try:
        from scipy.ndimage import binary_fill_holes,binary_closing
        boundary_mask=binary_fill_holes(binary_closing(available,iterations=2))
    except ImportError:
        boundary_mask=available
    ax.contour(np.asarray(x),np.asarray(y),boundary_mask.astype(float),
               levels=[.5],colors=['#2D3C40'],linewidths=.82,zorder=4)
    ax.set_xlim(ext[0]-.05,ext[1]+.05);ax.set_ylim(ext[2]-.05,ext[3]+.05)
    ax.set_aspect('equal',adjustable='box',anchor='C')
    ax.set_xticks([104,106,108,110]);ax.set_yticks([34,35,36,37])
    ax.set_xticklabels(['104°E','106°E','108°E','110°E'])
    ax.set_yticklabels(['34°N','35°N','36°N','37°N'])
    ax.grid(color='#879B9D',lw=.42,ls=(0,(2.2,3.2)),alpha=.40,zorder=3)
    ax.tick_params(labelsize=8.7,pad=1.5)
    for sp in ax.spines.values():sp.set_linewidth(.65);sp.set_color('#5E686C')
    # No statistics or labels are drawn ON the map. Counts are displayed below
    # the panel, outside the axes, by draw(). This preserves all map area.
    return img,norm


def panel_c(ax,curve):
    cc=curve.sort_values('HCL')
    xx=cc.HCL.to_numpy(float);yy=cc.Benefit.to_numpy(float)
    low=cc.Bootstrap50_CI_lower.to_numpy(float)
    high=cc.Bootstrap50_CI_upper.to_numpy(float)
    ax.axhline(0,color='#58696A',lw=.85,ls=(0,(3,2)),zorder=1)
    ax.fill_between(xx,low,high,color=DARK_TEAL,alpha=.17,zorder=2,lw=0)
    ax.plot(xx,yy,color=DARK_TEAL,lw=1.65,zorder=3)
    ax.scatter(xx,yy,s=24,facecolor='white',edgecolor=DARK_TEAL,lw=1.15,zorder=4)
    k=int(np.argmax(yy))
    ax.scatter([xx[k]],[yy[k]],s=47,facecolor=ORANGE,edgecolor='white',lw=.9,zorder=5)
    ax.annotate('HCL ≈ 0.59',xy=(xx[k],yy[k]),xytext=(.76,.82),
                textcoords='axes fraction',ha='center',fontsize=8.5,color=INK,
                arrowprops={'arrowstyle':'-','lw':.55,'color':'#555555'})
    ax.set_xlim(-4.6,5.1);ax.set_ylim(-.018,.064)
    ax.set_xticks([-4,-2,0,2,4]);ax.set_yticks([-.02,0,.02,.04,.06])
    ax.set_xlabel('Hydroclimatic limitation (HCL)',labelpad=2.5)
    ax.set_ylabel('Modelled contrast (B)',labelpad=2.5)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:.2f}'))
    ax.grid(axis='y',ls=(0,(2,3)),lw=.55,color=GRID,alpha=.8)
    ax.spines[['top','right']].set_visible(False)
    ax.text(.03,.97,'15 HCL bins · 50-km block 95% CI',transform=ax.transAxes,
            ha='left',va='top',fontsize=8.3,color=INK)


def panel_d(ax):
    yy=[2,1,0]
    ax.axvline(0,color='#58696A',lw=.9,ls=(0,(3,2)),zorder=1)
    for y,(label,mean,lo,hi,n,sig) in zip(yy,VEG):
        color=ORANGE if mean>.01 else (DARK_TEAL if sig else '#6F8186')
        ax.hlines(y,lo,hi,colors=color,lw=2.35,zorder=3)
        ax.scatter([mean],[y],s=65,facecolor=color if sig else 'white',
                   edgecolor=color,lw=1.5,zorder=4)
        ax.text(hi+.0016,y+.09,f'{mean:.4f}',fontsize=8.9,ha='left',va='center',
                color=INK,fontweight='bold',zorder=5)
    # Keep vegetation labels INSIDE this panel so they cannot collide with c.
    ax.set_yticks([])
    ax.set_xlim(-.023,.052);ax.set_ylim(-.50,2.53)
    for y,(name,*_) in zip(yy,VEG):
        ax.text(-.022,y,name,ha='left',va='center',fontsize=8.35,color=INK)
    ax.set_xticks([0,.01,.02,.03,.04])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:.2f}'))
    ax.set_xlabel('Mean modelled contrast (B)',labelpad=2.5)
    ax.grid(axis='x',color=GRID,ls=(0,(2,3)),lw=.55,alpha=.8)
    ax.spines[['top','right']].set_visible(False)
    ax.text(.96,.055,'Filled: 95% CI excludes 0',transform=ax.transAxes,
            fontsize=8.3,ha='right',va='bottom',color=INK)
    ax.text(.96,.97,'50-km 95% CI',transform=ax.transAxes,
            fontsize=8.3,ha='right',va='top',color=INK)


def draw(oof,pooled,pix,curve,grid,classes,palette='softdiv'):
    b,ext,x,y=map_grid(pix,grid)
    fig=plt.figure(figsize=(FIG_W_CM/2.54,FIG_H_CM/2.54),facecolor='white')
    # Four equal physical panel boxes. Reclaim the space formerly occupied
    # by the two explanatory footer sentences for taller panel axes.
    W,H=.385,.303
    # Keep all 4 axes equal-size and vertically aligned by column.
    # R9.1 right-column x = .540; R9.2 x = .557 (+2.45 mm in Word).
    A=[.102,.605,W,H];B=[.557,.605,W,H]
    C=[.102,.122,W,H];D=[.557,.122,W,H]
    axa=fig.add_axes(A);axb=fig.add_axes(B)
    axc=fig.add_axes(C);axd=fig.add_axes(D)
    panel_a(axa,oof,pooled)
    img,norm=panel_b(axb,b,pix,ext,x,y,classes,palette)
    panel_c(axc,curve);panel_d(axd)
    for ax,lbl,ttl in [(axa,'a','Historical OOF prediction'),
                       (axb,'b','Spatial greening–damage contrast'),
                       (axc,'c','Hydroclimatic dependence'),
                       (axd,'d','Land-cover contrasts')]:
        panel_title(fig,ax,lbl,ttl)

    # Share label sits in the WHITE BAND BELOW b, NEVER inside the map.
    # These are descriptive proportions; zero classification is defined in
    # Section 3.4 and Figure 9 caption of the accompanying manuscript.
    shares={k:100*classes[k]/len(pix) for k in ('Negative','Numerical zero','Positive')}
    fig.text(.562,.553,
             f'Neg {shares["Negative"]:.1f}%   ·   Zero {shares["Numerical zero"]:.1f}%'
             f'   ·   Pos {shares["Positive"]:.1f}%',
             ha='left',va='center',fontsize=8.2,fontweight='normal',color=INK)

    # Dedicated horizontal colourbar is kept below the share line.
    cax=fig.add_axes([.613,.504,.285,.014])
    cb=fig.colorbar(img,cax=cax,orientation='horizontal')
    cb.set_ticks([-.04,-.01,0,.04,.14])
    cb.set_ticklabels(['−0.04','−0.01','0','0.04','0.14'])
    cb.ax.tick_params(labelsize=8.0,pad=1.1,length=1.9)
    cb.outline.set_linewidth(.55)
    # Mark the exact numerical zero WITHOUT making it a white map colour.
    cax.axvline(0.,color='#181818',lw=.95,ymin=0,ymax=1)
    fig.text(.578,.511,'B',ha='right',va='center',fontsize=9.15,
             fontweight='bold',color=INK)
    return fig,b


def make_word_preview(image_path,out_dir):
    """A4 page preview: original manuscript width 21cm, margins 3.175cm."""
    from PIL import Image,ImageDraw
    dpi=240
    px_per_cm=dpi/2.54
    page=Image.new('RGB',(round(21*px_per_cm),round(29.7*px_per_cm)),'white')
    image=Image.open(image_path).convert('RGB')
    w=round(FIG_W_CM*px_per_cm); h=round(image.height*w/image.width)
    image=image.resize((w,h),Image.Resampling.LANCZOS)
    x=round(3.175*px_per_cm); y=round(4.0*px_per_cm)
    page.paste(image,(x,y))
    page.save(out_dir/'Figure9_R9_3_WordPagePreview.png')
    page.save(out_dir/'Figure9_R9_3_WordPagePreview.pdf',resolution=dpi)


def main():
    ap=argparse.ArgumentParser(description='Figure 9 R9.3: low-saturation diverging map and increased column spacing; true-size Word figure.')
    ap.add_argument('--palette',choices=sorted(PALETTES),default='softdiv')
    ap.add_argument('--exact-font',action='store_true',help='Require installed Times New Roman (Windows).')
    ap.add_argument('--word-preview',action='store_true',help='Export a true-scale A4 page illustration.')
    ap.add_argument('--data-dir',type=Path,help='Use existing frozen Figure 9 CSV directory without copying it.')
    ap.add_argument('--map-dir',type=Path,help='Directory containing F9_Grid_Reference.json.')
    ap.add_argument('--output-dir',type=Path,help='Write R9 files to a separate output folder.')
    args=ap.parse_args()
    global DATA,MAP,OUTPUT
    if args.data_dir is not None:
        DATA=args.data_dir.expanduser()
        for k,p in list(FILES.items()):
            if k!='grid':FILES[k]=DATA/p.name
    if args.map_dir is not None:
        MAP=args.map_dir.expanduser()
        FILES['grid']=MAP/'F9_Grid_Reference.json'
    if args.output_dir is not None:
        OUTPUT=args.output_dir.expanduser();OUTPUT.mkdir(exist_ok=True,parents=True)
    configure(args.exact_font)
    oof,pooled,pix,curve,peak,grid,counts=load()
    fig,b=draw(oof,pooled,pix,curve,grid,counts,args.palette)
    target=OUTPUT/'Figure9_R9_3_SoftDiverging_Final'
    for ext in ('png','pdf','svg'):
        fig.savefig(target.with_suffix('.'+ext),
                    dpi=600 if ext=='png' else 300,facecolor='white')
    plt.close(fig)
    if args.word_preview:
        make_word_preview(target.with_suffix('.png'),OUTPUT)
    qc=[
        'Figure 9 R9.3 | physical-size and data-preservation QC',
        f'Export width: {FIG_W_CM:.2f} cm; height: {FIG_H_CM:.2f} cm.',
        'Source Word text block: 21.00 - 3.175 - 3.175 = 14.650 cm.',
        'All 4 panel axes: same (W=.385, H=.303) figure coordinates.',
        'Right column shifted from x=.540 to x=.557 (+2.45 mm at 14.4 cm); all panel sizes unchanged.',
        f'OOF n={len(oof)}, pooled R2={float(pooled.R2):.12f}, RMSE={float(pooled.RMSE):.9f}',
        f'Sign counts: {counts}, total={sum(counts.values())}.',
        f'B actual min/max = {pix.B.min():.9f} / {pix.B.max():.9f}',
        f'Continuous palette: {args.palette}. Zero is included in SAME colour scale, not overpainted.',
        'Map has no overlaid percentage label; percentages shown below the map above its colorbar.',
        'Removed both bottom explanatory sentences; original B definition is retained in Methods/caption.',
        'No fitting, re-estimation, smoothing, grid interpolation, reclassification, or fabricated cells.',
        'CI for c and d is the original frozen 50-km spatial-block CI.',
    ]
    for k,p in FILES.items():qc.append(f'SHA256 {k}: {sha256(p)} {p.name}')
    (OUTPUT/'Figure9_R9_3_QC.txt').write_text('\n'.join(qc)+'\n',encoding='utf-8')
    print('PASS: Figure 9 R9.3 / '+args.palette)
    print('Figure width in Word = 14.40 cm. Do not enlarge or shrink after inserting.')
    print('Outputs in:',OUTPUT)

if __name__=='__main__':
    main()
