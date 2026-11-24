# -*- coding: utf-8 -*-
"""Figure 10 R5 — Word-native layout for the unchanged frozen 2026 review data.

Changes relative to R4 are purely presentational: enlarge upper-row a/f panels
and narrow the gap between the two columns without changing frozen data. Five frozen
CSV files are loaded directly; models / ALE / bootstrap are NOT rerun.

Run: python Figure10_R5_Plot.py [--exact-font] [--dpi 600]
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

HOME = Path(__file__).resolve().parent
DATA = HOME / 'data'
OUTPUT = HOME / 'outputs'
PREFIX = 'Figure10_Nonlinear_Hydroclimatic_Regulation_FINAL_V1_'
INPUT = {
    'importance': PREFIX+'SpatialPermutationImportance.csv',
    'ale': PREFIX+'MainALE_30bins.csv',
    'thresholds': PREFIX+'ThresholdStability.csv',
    'bootstrap': PREFIX+'BlockBootstrapThresholds.csv',
    'interaction': PREFIX+'NIRv_RootSM_SecondOrderALE.csv',
}
COL = {'RootSM_base':'#28748E','VPD_base':'#B65545',
       'Precip_base':'#3C8A76','RAD_base':'#B48B44'}


def setup(exact_font):
    if exact_font:
        from matplotlib.font_manager import FontProperties, findfont
        findfont(FontProperties(family='Times New Roman'),fallback_to_default=False)
    mpl.rcParams.update({
        'font.family':'serif',
        'font.serif':(['Times New Roman'] if exact_font else ['Times New Roman','Liberation Serif','Nimbus Roman','DejaVu Serif']),
        'mathtext.fontset':'stix','font.weight':'bold','axes.titleweight':'bold',
        'axes.labelweight':'bold','font.size':9.0,'axes.labelsize':9.2,
        'xtick.labelsize':8.0,'ytick.labelsize':8.0,'axes.titlesize':10.5,
        'axes.linewidth':.82,'axes.edgecolor':'#2F373B',
        'xtick.major.width':.75,'ytick.major.width':.75,
        'xtick.major.size':3,'ytick.major.size':3,
        'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white',
    })


def read_frozen():
    for p in INPUT.values():
        if not (DATA/p).is_file():
            raise FileNotFoundError(DATA/p)
    d={k:pd.read_csv(DATA/v) for k,v in INPUT.items()}
    assert len(d['importance'])==8
    assert len(d['ale'])==120 and len(d['thresholds'])==4
    assert len(d['bootstrap'])==120 and len(d['interaction'])==144
    assert set(d['ale'].feature)==set(d['thresholds'].feature)
    for _,g in d['ale'].groupby('feature'):
        assert len(g)==30 and int(g['count'].sum())==153932
    assert np.isclose(d['importance']['relative_importance_pct'].sum(),100,atol=1e-5)
    assert np.isclose(d['importance'].loc[d['importance'].feature.eq('RAD_base'),'relative_importance_pct'].iloc[0],40.27494205682875,atol=1e-7)
    assert len(d['interaction'].NIRv_trend.unique())==len(d['interaction'].RootSM.unique())==12
    return d


def plotframe(ax):
    ax.set_facecolor('#FAFBFC')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both',pad=2.3,width=.75)
    ax.grid(axis='y',lw=.52,ls=(0,(2,3)),color='#CCD5D9',zorder=0)
    ax.set_axisbelow(True)
    for t in ax.get_xticklabels()+ax.get_yticklabels(): t.set_fontweight('bold')


def label(fig,x,y,panel,title,fontsize=10.4):
    fig.text(x,y,f'{panel}  {title}',fontsize=fontsize,color='#141719',fontweight='bold',ha='left',va='bottom')


def importance(fig,ax,imp):
    imp=imp.sort_values('relative_importance_pct',ascending=False).reset_index(drop=True)
    ys=np.arange(len(imp))
    colors=['#B85C48','#BA8F53']+['#7893A1']*5+['#536F80']
    ax.barh(ys,imp.relative_importance_pct,height=.72,color=colors,edgecolor='none',zorder=2)
    ax.invert_yaxis()
    names={'Root-zone soil moisture':'Root-zone moisture',
           'Deep soil moisture':'Deep soil moisture',
           'Maximum temperature':'Max. temperature',
           'Mean temperature':'Mean temperature',
           'Greening trend':'NIRv greening trend'}
    ax.set_yticks(ys)
    ax.set_yticklabels([names.get(n,n) for n in imp.display_name],fontsize=8.2)
    ax.tick_params(axis='y',length=0,pad=4)
    ax.set_xlim(0,49);ax.set_xticks([0,20,40]);ax.grid(axis='y',visible=False)
    ax.grid(axis='x',lw=.5,ls=(0,(2,3)),color='#D4DBDE',zorder=0)
    ax.spines['left'].set_visible(False)
    for y,v in enumerate(imp.relative_importance_pct):
        ax.text(v+.52,y,f'{v:.1f}%',ha='left',va='center',fontsize=8.35,fontweight='bold')
    ax.set_xlabel('Relative importance (%)',labelpad=3.0,fontsize=8.9)


def edges(v):
    v=np.sort(np.unique(np.asarray(v,float)))
    e=np.empty(len(v)+1,float)
    e[1:-1]=(v[1:]+v[:-1])/2
    e[0]=v[0]-(v[1]-v[0])/2
    e[-1]=v[-1]+(v[-1]-v[-2])/2
    return e


def interaction(fig,ax,cax,d):
    xs=np.sort(d.NIRv_trend.unique())
    ys=np.sort(d.RootSM.unique())
    mat=d.pivot(index='RootSM',columns='NIRv_trend',values='ALE_interaction').reindex(index=ys,columns=xs)
    vals=np.ma.masked_invalid(mat.to_numpy(float))
    vmax=float(np.nanmax(np.abs(vals)))
    cm=LinearSegmentedColormap.from_list('F10_R5_cw',['#286585','#83B1C7','#F1EFEA','#E7A889','#B54440'],256)
    cm.set_bad('#FFFFFF')
    mesh=ax.pcolormesh(edges(xs),edges(ys),vals,shading='flat',rasterized=True,
                       cmap=cm,norm=TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax))
    ax.set_xlim(edges(xs)[0],edges(xs)[-1]);ax.set_ylim(edges(ys)[0],edges(ys)[-1])
    ax.set_xticks([0,.02,.04]);ax.set_yticks([.20,.28,.36])
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xlabel('NIRv trend (decade$^{-1}$)',labelpad=3,fontsize=8.9)
    ax.set_ylabel('RootSM (m$^3$ m$^{-3}$)',labelpad=2,fontsize=8.9)
    ax.tick_params(pad=2.2)
    for t in ax.get_xticklabels()+ax.get_yticklabels():t.set_fontweight('bold')
    cb=fig.colorbar(mesh,cax=cax,orientation='vertical',ticks=[-.006,0,.006])
    cb.ax.tick_params(labelsize=7.7,width=.7,length=2.3,pad=2)
    cb.ax.set_yticklabels(['−.006','0','+.006'])
    for t in cb.ax.get_yticklabels():t.set_fontweight('bold')
    cb.outline.set_linewidth(.7)


def ale(ax,df,row,key,xlabel):
    g=df.loc[df.feature.eq(key)].sort_values('x')
    x=g.x.to_numpy(float);y=g.ALE.to_numpy(float)
    lo=float(row.bootstrap_value_CI_low);hi=float(row.bootstrap_value_CI_high)
    median=float(row.bootstrap_value_median)
    hue=COL[key]
    if row.stability=='uncertain':
        ax.axvspan(lo,hi,facecolor='#AEB8C1',alpha=.19,zorder=0)
        ax.axvline(median,color='#68757B',lw=1.05,ls=(0,(4,3)),zorder=2)
    else:
        ax.axvspan(lo,hi,facecolor=hue,alpha=.11,zorder=0)
        ax.axvline(median,color=hue,lw=1.05,ls=(0,(4,3)),zorder=2)
    ax.plot(x,y,color=hue,lw=1.7,marker='o',ms=2.7,
            markerfacecolor='white',markeredgecolor=hue,markeredgewidth=.8,zorder=3)
    ax.axhline(0,color='#555C60',lw=.75,zorder=1)
    span=x.max()-x.min()
    ax.set_xlim(x.min()-.025*span,x.max()+.025*span)
    dy=y.max()-y.min()
    ax.set_ylim(y.min()-.14*dy,y.max()+.18*dy)
    ax.set_xlabel(xlabel,labelpad=4,fontsize=8.75)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    if key=='RootSM_base':
        # Keep diagnostic but out of the data area: the shaded band already shows uncertainty.
        ax.text(.98,.96,'Uncertain location',fontsize=7.3,fontweight='bold',
                transform=ax.transAxes,va='top',ha='right',color='#343C40',
                bbox=dict(facecolor='#F6F8F8',edgecolor='none',alpha=.85,pad=1.2))


def draw(d):
    # A4-compatible figure width of 15.5 cm; the larger HEIGHT eliminates R3 collisions.
    fig=plt.figure(figsize=(6.10,6.60),facecolor='white')
    # Top panels: reserve an entire left margin for long importance labels.
    # R5: enlarge the top pair and reduce the horizontal gap between ALE columns.
    # The left importance axis retains a margin for long feature names.
    a=fig.add_axes([.205,.728,.315,.203])
    f=fig.add_axes([.625,.731,.285,.200])
    cb=fig.add_axes([.934,.731,.014,.200])
    # Balanced, wider ALE panels; horizontal gutter reduced from R4.
    b=fig.add_axes([.095,.427,.390,.196])
    c=fig.add_axes([.580,.427,.390,.196])
    d_ax=fig.add_axes([.095,.118,.390,.196])
    e=fig.add_axes([.580,.118,.390,.196])
    for ax in (a,b,c,d_ax,e):plotframe(ax)
    importance(fig,a,d['importance'])
    interaction(fig,f,cb,d['interaction'])
    th=d['thresholds'].set_index('feature')
    ale(b,d['ale'],th.loc['RootSM_base'],'RootSM_base','RootSM (m$^3$ m$^{-3}$)')
    ale(c,d['ale'],th.loc['VPD_base'],'VPD_base','VPD (kPa)')
    ale(d_ax,d['ale'],th.loc['Precip_base'],'Precip_base','Precipitation (mm / 8 days)')
    ale(e,d['ale'],th.loc['RAD_base'],'RAD_base','Radiation (MJ m$^{-2}$ / 8 days)')
    b.set_ylabel('First-order ALE',labelpad=3.2,fontsize=9.0)
    d_ax.set_ylabel('First-order ALE',labelpad=3.2,fontsize=9.0)
    # Single shared annotation above f instead of long colour-bar title.
    label(fig,.067,.953,'a','Predictor importance',fontsize=10.5)
    label(fig,.595,.953,'f','Greening × RootSM interaction',fontsize=10.1)
    label(fig,.095,.658,'b','Root-zone soil moisture')
    label(fig,.580,.658,'c','Vapor pressure deficit')
    label(fig,.095,.347,'d','Precipitation')
    label(fig,.580,.347,'e','Solar radiation')
    return fig


def qc_txt(d,hashes):
    imp=d['importance'].set_index('feature')
    vals={k:round(float(imp.loc[k,'relative_importance_pct']),4) for k in imp.index}
    lines=['FIGURE 10 R5 WORD-ADAPTED DRAWING QC',
           'Scientific source: frozen L6.0 FINAL V2 / Figure 10 V1 CSV files',
           'Plotting only: no refitting, no new ALE/CI/bootstraps.',
           'Physical size: 6.10 x 6.60 in (~15.49 x 16.76 cm).',
           'Main panels: a importance; b RootSM ALE; c VPD ALE; d Precip ALE; e Radiation ALE; f 2D interaction.',
           'Dashed lines: bootstrap-median turning positions.',
           'Shaded ranges: bootstrap 95% value intervals (30 block resamples), NOT ALE effect CIs.',
           'RootSM turning position: uncertain.',
           'R5: larger a/f panels; narrower two-column gap; no long footer inside figure.',
           'SOURCE SHA256:',
           *[f'{k}: {v}' for k,v in hashes.items()],
           'IMPORTANCE PERCENTAGES:',str(vals),'TURNING POINTS:']
    for _,r in d['thresholds'].iterrows():
        lines.append(f"{r.feature}: main={r.main_threshold_value:.8f}; median={r.bootstrap_value_median:.8f}; CI=[{r.bootstrap_value_CI_low:.8f},{r.bootstrap_value_CI_high:.8f}]; {r.stability}")
    lines.append('Source pixels for all first-order ALE variables: 153,932 per feature.')
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--exact-font',action='store_true')
    p.add_argument('--dpi',type=int,default=600)
    args=p.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    setup(args.exact_font)
    data=read_frozen()
    hashes={k:hashlib.sha256((DATA/v).read_bytes()).hexdigest() for k,v in INPUT.items()}
    fig=draw(data)
    stem=OUTPUT/'Figure10_R5_WordSize'
    for suffix in ('.png','.pdf','.svg'):
        fig.savefig(str(stem)+suffix,facecolor='white',dpi=args.dpi if suffix=='.png' else 300)
    plt.close(fig)
    (OUTPUT/'Figure10_R5_QC.txt').write_text(qc_txt(data,hashes),encoding='utf-8')
    print('COMPLETE:',stem)

if __name__=='__main__':main()
