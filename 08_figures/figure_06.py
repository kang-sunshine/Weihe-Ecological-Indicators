# -*- coding: utf-8 -*-
"""Figure 6 R2 V2 | typography and panel-area refinement | reproducible plotting only.

Requirements: numpy pandas matplotlib
Example (Windows PowerShell):
  python Figure6_R2_V2_TimesNewRoman_LargerPanels.py --legacy "./EI_Figure6_Redraw_DataPack.zip" --recovery ".\\EI_Figure6_R2_ContinuousRecovery\\data" --out ".\\EI_Figure6_R2_ContinuousRecovery\\outputs"

No data/model recalculation. Legacy panels source M1; continuous recovery M5.
Structural empty bins remain NaN, and are never interpolated or extrapolated.
"""
from __future__ import annotations
import argparse
import io
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.font_manager import fontManager
import warnings
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
LEGACY_FILE = 'EI_Figure6_Redraw_DataPack.zip'
E_FILE = 'F6_R2_E_Continuous_P120.csv'
F_FILE = 'F6_R2_F_Continuous_RMPT120.csv'
EVENTS = ('CDH', 'HEAT', 'HR')
C = {'CDH': '#C65A3D', 'HEAT': '#9A5291', 'HR': '#277FA3'}
NAMES = {'CDH': 'Compound dry–hot', 'HEAT': 'High temperature', 'HR': 'Heavy rainfall'}
CORE_FILES = ['F6_ABC_Legacy_Bins.csv','F6_ABC_Legacy_Fits.csv',
              'F6_ABC_Breakpoint_Metadata.csv','F6_D_Stage_Evidence.csv',
              'F6_EF_Recovery_Contrasts.csv']

CAPTION = (
    'Figure 6. Severity-dependent legacy candidates and recovery dynamics under three '
    'hydroclimatic extremes. (a–c) Historical within-bin median post-event legacy '
    '(open circles), original fitted segmented relationships (solid lines) and '
    'within-bin interquartile ranges (light shading), indexed to the full hazard-specific '
    'severity distribution. Vertical dashed lines indicate upper-search-edge legacy '
    'candidates (*), not independently validated ecological thresholds. '
    '(d) Response-stage candidates for damage initiation (circles), magnitude '
    '(triangles) and legacy (diamonds); the HR damage-magnitude candidate is unsupported '
    '(gray cross), while HEAT magnitude and legacy candidates nearly coincide. '
    '(e,f) Kaplan–Meier 120-day recovery probability and restricted mean persistence '
    'time (RMPT120), respectively, across fixed 5-percentile severity bins. '
    'Lines connect adjacent estimable bin centres only; light bands show '
    '50-km whole-spatial-block bootstrap 95% confidence intervals (250 resamples). '
    'Structural empty bins produced by tied severities and the full-population upper-rank '
    'ECDF remain blank. CDH Q95–100 has 237 valid bootstrap resamples; other nonempty '
    'bins have 250. The shaded Q90–100 area marks the Extreme class. '
    'Recovery uses the original 120-day follow-up and subsequent-event censoring rules. '
    'These binned recovery curves are descriptive rather than smoothed monotonic fits; '
    'the within-bin IQR in (a–c) is not a confidence interval, and their y-axis scales differ.'
)

def load_zip_or_dir(source: Path, filename: str) -> pd.DataFrame:
    if source.is_dir():
        return pd.read_csv(source / filename)
    with zipfile.ZipFile(source, 'r') as z:
        return pd.read_csv(io.BytesIO(z.read(filename)))


def find_legacy(custom: str | None) -> Path:
    if custom:
        p=Path(custom)
        if not p.exists(): raise FileNotFoundError(f'Legacy data not found: {p}')
        return p
    for p in (ROOT/LEGACY_FILE, ROOT/'data'/LEGACY_FILE,
              ROOT/'EI_Figure6_Redraw_DataPack',
              Path.cwd()/LEGACY_FILE,
              Path(r'.\analysis_project')/LEGACY_FILE):
        if p.exists(): return p
    raise FileNotFoundError('Could not find legacy CSV ZIP. Specify --legacy FULL_PATH_TO_ZIP.')


def find_rec(custom: str | None) -> Path:
    if custom:
        d=Path(custom)
        if (d/E_FILE).exists() and (d/F_FILE).exists(): return d
        if (d/'data'/E_FILE).exists() and (d/'data'/F_FILE).exists(): return d/'data'
        raise FileNotFoundError(f'Recovery CSVs missing from: {d}')
    for d in (ROOT, ROOT/'data',
              Path(r'.\EI_Figure6_R2_ContinuousRecovery\data')):
        if (d/E_FILE).exists() and (d/F_FILE).exists(): return d
    raise FileNotFoundError('Could not find continuous recovery CSVs. Specify --recovery DIRECTORY.')


def verify(bins, fits, meta, stage, old, pe, pf):
    expected_ecdf='FULL_HAZARD_ECDF_FLOAT64_RIGHT_UPPER_RANK'
    for name,d in [('bin',bins),('fit',fits),('meta',meta),('stage',stage),('old',old)]:
        if set(d.Event)!=set(EVENTS): raise ValueError(f'Event set mismatch: {name}')
        if 'ECDF_Reference' in d and set(d.ECDF_Reference.dropna())!={expected_ecdf}:
            raise ValueError(f'Full-hazard ECDF mismatch: {name}')
    if [len(meta),len(stage),len(old)] != [3,9,6]:
        raise ValueError('Historical source rows changed')
    if not (meta.Edge_Hit.astype(str)=='UPPER').all():
        raise ValueError('Expected three search-edge constrained candidates')
    if set(old.Analysis)!={'PRIMARY_POSTPEAK_CATALOG_INCLUSIVE'}:
        raise ValueError('Historical recovery analysis not the M5 primary endpoint')
    for d in [pe,pf]:
        if len(d)!=60 or set(d.event_type)!=set(EVENTS):
            raise ValueError('Continuous recovery requires 3 hazards × 20 bins')
        for ev in EVENTS:
            a=d[d.event_type==ev].sort_values('bin_left')
            if a.bin_left.tolist()!=list(range(0,100,5)) or a.bin_right.tolist()!=list(range(5,105,5)):
                raise ValueError(f'Unexpected 5-percentile bins: {ev}')
        # Two archived registry labels describe the same all-finite, full-hazard upper-rank ECDF.
        if set(d.ECDF_reference) != {'ALL_FINITE_ORIGINAL_L53_HAZARD_FLOAT64_RIGHT_UPPER_RANK'}:
            raise ValueError('Unexpected recovery percentile definition')
        expected_pop={'CDH':4796211,'HEAT':17055393,'HR':14395376}
        for ev,pop in expected_pop.items():
            if set(d.loc[d.event_type==ev,'n_ECDF_reference'])!={pop}:
                raise ValueError('Full-hazard reference population mismatch: '+ev)
        if not (d.bootstrap_block_km==50).all(): raise ValueError('Bootstrap blocks not 50 km')
        if not (d.bootstrap_n==250).all(): raise ValueError('Bootstrap replicates not 250')
        if not (d.Version=='PRIMARY_POSTPEAK_CATALOG_INCLUSIVE').all():
            raise ValueError('Recovery endpoint version mismatch')
    k=['event_type','bin_left','bin_right']
    a=pe.sort_values(k).reset_index(drop=True)
    b=pf.sort_values(k).reset_index(drop=True)
    for col in ['n_events','n_risk_120','bootstrap_valid_n','estimate_status']:
        if not (a[col].values==b[col].values).all():
            raise ValueError('P120/RMPT support not aligned: '+col)
    for ev, n, count, blank in [('CDH',1640995,19,1),('HEAT',7018595,17,3),('HR',4121986,14,6)]:
        s=a[a.event_type==ev]
        if s.n_events.sum()!=n or (s.estimate_status=='ESTIMABLE').sum()!=count or (s.estimate_status=='EMPTY_BIN').sum()!=blank:
            raise ValueError('Unexpected event/empty-bin accounting: '+ev)
    for d, value in [(a,'P120'),(b,'RMPT120')]:
        g=d[d.estimate_status=='ESTIMABLE']
        if not g[value].between(0,1 if value=='P120' else 120).all():
            raise ValueError('Endpoint outside feasible range')
        if not ((g.CI_lower <= g[value]) & (g[value] <= g.CI_upper)).all():
            raise ValueError('Endpoint estimate outside its CI')
        if not (g.bootstrap_valid_n>=200).all():
            raise ValueError('Insufficient valid bootstrap support')
    for ev, q in [('CDH',89.2472),('HEAT',90.330794),('HR',93.605530)]:
        m=meta[meta.Event==ev].iloc[0]
        if abs(m.Q_full-q)>0.015: raise ValueError('Legacy candidate location changed: '+ev)
    return a,b


def style():
    """Use actual Times New Roman when present (normally on Windows).

    Missing Times New Roman is never silently reported as compliant: a local
    non-Windows preview can use a metric-compatible surrogate without bundling fonts.
    The authoritative manuscript PDF should be exported on a machine with TNR.
    """
    fonts={f.name for f in fontManager.ttflist}
    if 'Times New Roman' in fonts:
        font='Times New Roman'
        exact=True
    else:
        font=next((f for f in ('Liberation Serif','Nimbus Roman') if f in fonts), 'DejaVu Serif')
        exact=False
        warnings.warn('Times New Roman is not installed here. Preview uses '
                      +font+'. Run on a Windows PC with Times New Roman for the final TNR PDF.',
                      RuntimeWarning,stacklevel=2)
    plt.rcParams.update({
        'font.family':'serif', 'font.serif':[font], 'font.size':9.2,
        'font.weight':'normal', 'axes.linewidth':.85,
        'axes.labelweight':'bold','axes.titleweight':'bold',
        'xtick.labelsize':8.7,'ytick.labelsize':8.7,
        'xtick.major.size':3.1,'xtick.major.width':.8,
        'pdf.fonttype':42,'ps.fonttype':42,
        'savefig.facecolor':'white',
        'mathtext.fontset':'custom',
        'mathtext.rm':font,
        'mathtext.it':font+':italic',
        'mathtext.bf':font+':bold'
    })
    return font,exact


def decorate(ax,title,xlabel,ylabel=None,xlim=(0,100),shade=True):
    ax.set_title(title,loc='left',fontsize=10.4,fontweight='bold',pad=9)
    ax.set_xlabel(xlabel,fontsize=9.0,labelpad=5)
    if ylabel: ax.set_ylabel(ylabel,fontsize=9.2,labelpad=6)
    ax.set_xlim(*xlim)
    ax.set_facecolor('#F8F9FB')
    if shade: ax.axvspan(90,100,color='#D7DDE3',alpha=.44,zorder=0)
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    ax.grid(axis='y',linestyle=(0,(2,3)),linewidth=.55,color='#CAD3D9',alpha=.9,zorder=0)
    ax.tick_params(axis='both',direction='out',pad=3)
    ax.set_axisbelow(True)


def legacy_panel(ax,event,letter,bins,fits,meta):
    color=C[event];d=bins[bins.Event==event].sort_values('Q_full');f=fits[fits.Event==event].sort_values('Q_full');m=meta[meta.Event==event].iloc[0]
    x=d.Q_full.to_numpy(float);y=d.Y.to_numpy(float)
    lo=d.Y_IQR_Lower.to_numpy(float);hi=d.Y_IQR_Upper.to_numpy(float)
    decorate(ax,f'({letter}) {NAMES[event]}','Severity percentile',
             'Post-event legacy\n(anomaly-days)' if letter in ('a','c') else None)
    ax.fill_between(x,lo,hi,color=color,alpha=.085,linewidth=0,zorder=1)
    ax.plot(f.Q_full,f.Y,color=color,lw=2.15,zorder=4,solid_capstyle='round')
    ax.scatter(x,y,facecolors='white',edgecolors=color,s=16,lw=.75,zorder=5)
    q=float(m.Q_full)
    ax.axvline(q,color='#606A73',lw=1.1,ls=(0,(3,3)),zorder=2)
    ymax=max(np.max(hi),float(f.Y.max()),float(d.Y.max()))
    ax.set_ylim(0,max(8,np.ceil(ymax*1.11/5)*5))
    ax.set_xticks([0,20,40,60,80,100])
    ax.text(.96,.94,f'Q{q:.1f}*',ha='right',va='top',transform=ax.transAxes,
            fontsize=9.3,fontweight='bold',color='#303943',
            bbox={'facecolor':'#F8F9FB','edgecolor':'none','alpha':.85,'pad':.8},zorder=6)


def stages_panel(ax,df):
    decorate(ax,'(d) Response-stage candidates','Severity percentile',xlim=(26,102))
    ax.set_ylim(-.64,2.62);ax.set_xticks([30,40,50,60,70,80,90,100]);ax.set_yticks([2,1,0]);ax.set_yticklabels(EVENTS)
    ymap={'CDH':2,'HEAT':1,'HR':0}
    for ev in EVENTS:
        color=C[ev];y=ymap[ev];d=df[df.Event==ev]
        onset=d[d.Stage.str.contains('initiation')].iloc[0]
        mag=d[d.Stage.str.contains('magnitude')].iloc[0]
        legacy=d[d.Stage.str.contains('legacy')].iloc[0]
        q0,q1,q2=[float(x.Q_full) for x in (onset,mag,legacy)]
        ax.hlines(y,28,100,color='#D3DADE',lw=.85,zorder=1)
        ax.scatter(q0,y,marker='o',s=51,facecolors='white',edgecolors=color,lw=1.6,zorder=4)
        ax.annotate(f'{q0:.1f}',(q0,y),xytext=(0,-12 if ev=='HR' else 10),textcoords='offset points',
                    ha='center',va='top' if ev=='HR' else 'bottom',fontsize=8.1,color='#2F3740')
        if ev=='HEAT':
            ax.scatter(q1,y-.16,marker='^',s=54,facecolors=color,edgecolors=color,lw=1.15,zorder=5)
            ax.scatter(q2,y+.16,marker='D',s=47,facecolors=color,edgecolors=color,lw=1,zorder=5)
            ax.annotate(f'{q2:.1f} ×2',(q2,y+.16),xytext=(-8,13),textcoords='offset points',
                        fontsize=8.3,ha='right',va='bottom',color=color,fontweight='bold')
        elif ev=='HR':
            ax.scatter(q1,y,marker='x',s=72,color='#8D969E',lw=2,zorder=5)
            ax.annotate('40.4 n.s.',(q1,y),xytext=(0,-13),textcoords='offset points',
                        fontsize=8.1,ha='center',va='top',color='#777F87')
            ax.scatter(q2,y,marker='D',s=49,facecolors=color,edgecolors=color,zorder=5)
            ax.annotate(f'{q2:.1f}*',(q2,y),xytext=(-2,12),textcoords='offset points',
                        fontsize=8.3,ha='right',color=color,fontweight='bold')
        else:
            ax.scatter(q1,y,marker='^',s=55,facecolors='white',edgecolors=color,lw=1.6,zorder=5)
            ax.annotate(f'{q1:.1f}',(q1,y),xytext=(0,-13),textcoords='offset points',
                        fontsize=8.3,ha='center',va='top',color='#2F3740')
            ax.scatter(q2,y,marker='D',s=49,facecolors=color,edgecolors=color,zorder=5)
            ax.annotate(f'{q2:.1f}*',(q2,y),xytext=(-1,10),textcoords='offset points',
                        fontsize=8.3,ha='right',va='bottom',color=color,fontweight='bold')


def recovery_panel(ax,df,letter,metric):
    is_p=metric=='P120'
    decorate(ax, '(e) Recovery probability' if is_p else '(f) Persistence time',
             'Severity percentile',
             'KM recovery by day 120 (%)' if is_p else 'RMPT120 (days)')
    ax.set_xticks([0,20,40,60,80,100]);ax.set_xlim(0,100)
    if is_p:
        ax.set_ylim(30,102);ax.set_yticks([40,50,60,70,80,90,100])
    else:
        ax.set_ylim(45,94);ax.set_yticks([50,60,70,80,90])
    scale=100 if is_p else 1
    for ev in EVENTS:
        a=df[df.event_type==ev].sort_values('bin_left').copy()
        # Retain all 20 rows in the array, including NaNs: matplotlib breaks the lines across empty bins.
        xx=a.bin_midpoint.to_numpy(float)
        yy=a[metric].to_numpy(float)*scale
        ll=a.CI_lower.to_numpy(float)*scale;uu=a.CI_upper.to_numpy(float)*scale
        ax.fill_between(xx,ll,uu,color=C[ev],alpha=.105,lw=0,zorder=1)
        ax.plot(xx,yy,color=C[ev],lw=1.8,marker='o',ms=3.4,
                mec='white',mew=.65,zorder=4,solid_capstyle='round',label=ev)
    # Visual gaps are data, not missing labels, and are called out explicitly in the caption.


def supplement(old,outdir):
    fig,axes=plt.subplots(1,2,figsize=(6.5,2.95))
    fig.subplots_adjust(left=.11,right=.98,bottom=.24,top=.86,wspace=.37)
    for ax,metric,title,units in zip(axes,['KM_P120','KM_RMPT120'],
                                  ['(a) Recovery probability','(b) Persistence time'],
                                  ['Extreme − High (pp)','Extreme − High (days)']):
        decorate(ax,title,units,shade=False)
        ax.axvline(0,color='#707983',ls=(0,(3,3)),lw=1.0)
        for i,ev in enumerate(EVENTS):
            row=old[(old.Event==ev)&(old.Metric==metric)].iloc[0]
            y=2-i;c=C[ev];low=float(row.Difference_CI_50km_Lower);high=float(row.Difference_CI_50km_Upper);v=float(row.Difference)
            ax.plot([low,high],[y,y],color=c,lw=2.3,solid_capstyle='round')
            ax.plot([low,low],[y-.055,y+.055],color=c,lw=1.3)
            ax.plot([high,high],[y-.055,y+.055],color=c,lw=1.3)
            ax.scatter([v],[y],s=48,c=c,edgecolors='white',linewidth=.8,zorder=4)
        ax.set_yticks([2,1,0]);ax.set_yticklabels(EVENTS)
        ax.set_ylim(-.5,2.5);ax.grid(axis='x',lw=.55,ls=(0,(2,3)),color='#D1D9DF');ax.grid(axis='y',visible=False)
        ax.set_xlim((-24,7) if metric=='KM_P120' else (-7,18))
    for ext in ('pdf','png'):
        fig.savefig(outdir/f'Figure6_Supp_R2_V2_TimesNewRoman.{ext}',dpi=600,facecolor='white')
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--legacy',help='Prior EI_Figure6_Redraw_DataPack.zip or extracted folder')
    ap.add_argument('--recovery',help='Folder containing F6_R2_E/F_Continuous CSVs')
    ap.add_argument('--out',help='Output folder, default: ./Figure6_R2_OUTPUT')
    args=ap.parse_args();old_src=find_legacy(args.legacy);rec_dir=find_rec(args.recovery)
    out=Path(args.out) if args.out else ROOT/'Figure6_R2_OUTPUT';out.mkdir(parents=True,exist_ok=True)
    bins,fits,meta,stage,old=[load_zip_or_dir(old_src,f) for f in CORE_FILES]
    pe=pd.read_csv(rec_dir/E_FILE);pf=pd.read_csv(rec_dir/F_FILE)
    pe,pf=verify(bins,fits,meta,stage,old,pe,pf)
    font,font_exact=style()
    # Keep 16.5 cm manuscript width, but reclaim whitespace for larger panels:
    # panel width +~7%, height +~15% vs R2 V1 at identical insertion width.
    fig,axes=plt.subplots(3,2,figsize=(16.5/2.54,21.2/2.54))
    fig.subplots_adjust(left=.110,right=.975,top=.955,bottom=.147,wspace=.225,hspace=.43)
    for ax,ev,letter in [(axes[0,0],'CDH','a'),(axes[0,1],'HEAT','b'),(axes[1,0],'HR','c')]:
        legacy_panel(ax,ev,letter,bins,fits,meta)
    stages_panel(axes[1,1],stage)
    recovery_panel(axes[2,0],pe,'e','P120')
    recovery_panel(axes[2,1],pf,'f','RMPT120')
    legend=[Line2D([0],[0],color=C[ev],lw=2,marker='o',markersize=4,label=ev) for ev in EVENTS]
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.56,.037),ncol=3,
               frameon=False,fontsize=9.1,handlelength=1.6,columnspacing=1.8)
    fig.text(.110,.086,'* upper-search-edge legacy candidate; n.s. unsupported',
             ha='left',va='center',fontsize=7.2,color='#565E65')
    for ext in ('pdf','png'):
        fig.savefig(out/f'Figure6_R2_V2_TimesNewRoman_LargerPanels.{ext}',dpi=600,facecolor='white')
    plt.close(fig)
    supplement(old,out)
    (out/'Figure6_R2_V2_Caption.txt').write_text(CAPTION,encoding='utf-8')
    lines=['Figure 6 R2 V2 plot QC','Source legacy: '+str(old_src),'Source recovery: '+str(rec_dir),
           f'Font: {font}', f'Exact Times New Roman available: {font_exact}','No interpolation / no refitting / no event recataloguing']
    for ev in EVENTS:
        p=pe[pe.event_type==ev];f=pf[pf.event_type==ev]
        minimum=int(p.loc[p.estimate_status=='ESTIMABLE','bootstrap_valid_n'].min())
        lines.append(f'{ev}: events={p.n_events.sum():,}; estimable={sum(p.estimate_status=="ESTIMABLE")}/20; empty={sum(p.estimate_status=="EMPTY_BIN")}; bootstrap_min={minimum}')
    (out/'Figure6_R2_V2_PlotQC.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines));print('Outputs:',out)
    if not font_exact: print('FINAL FONT QC: run on a Windows machine with Times New Roman to embed true TNR.')

if __name__=='__main__':main()
