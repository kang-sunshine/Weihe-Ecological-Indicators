# -*- coding: utf-8 -*-
"""Figure 4 reviewer revision R2 | more unified composite-damage trajectory and frozen scientific results.

All plotted values are read from the archived FINAL V3 source tables.
No recomputation of confidence intervals, no redefinition of A_comp as Delta A_comp.
Panel d landmarks are DESCRIPTIVE onset-aligned summaries, not formal M5 recovery.
"""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'; OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
SOURCE=DATA/'Figure4_V1_Ensemble_SourceData.csv'
TIMING=DATA/'Figure4_V2_ResponseTiming_Corrected.csv'
STEM=OUT/'Figure4_R2_WordSize'

mpl.rcParams.update({'font.family':'serif',
    'font.serif':['Times New Roman','Liberation Serif','Nimbus Roman','DejaVu Serif'],
    'mathtext.fontset':'stix','font.weight':'bold','axes.titleweight':'bold',
    'axes.labelweight':'bold','font.size':8.2,'axes.titlesize':10.3,
    'axes.labelsize':8.6,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
    'legend.fontsize':7.7,'axes.edgecolor':'#292929','axes.linewidth':.75,
    'pdf.fonttype':42,'ps.fonttype':42,'text.color':'black','axes.labelcolor':'black'})

COLOR={'Tmax':'#B54B3A','VPD':'#D18422','RootSM':'#3E75A5',
       'ZSync':'#65508F','ZuWUE':'#278B8E','ZNIRv':'#5A9656','A_comp':'#C5535D'}


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for ch in iter(lambda:f.read(2**20),b''):h.update(ch)
    return h.hexdigest()


def get(df,name):
    q=df.loc[df.variable.eq(name)].sort_values('relative_days')
    assert len(q)==13 and q.relative_days.is_unique
    return q


def label_ticks(ax):
    for t in ax.get_xticklabels()+ax.get_yticklabels():t.set_fontweight('bold')


def draw_strip(fig,rect,q,color,label,invert=False,show_x=False):
    ax=fig.add_axes(rect)
    x=q.relative_days.to_numpy(float)
    y=q['mean'].to_numpy(float)
    low=q.ci95_low.to_numpy(float)
    high=q.ci95_high.to_numpy(float)
    if invert:y,low,high=-y,-high,-low
    ax.fill_between(x,low,high,color=color,alpha=.15,lw=0)
    ax.plot(x,y,color=color,lw=1.55,solid_capstyle='round')
    ax.axvline(0,c='#4C4C4C',lw=.75,ls=(0,(4,3)))
    ax.axhline(0,c='#BBBBBB',lw=.65)
    valid=np.r_[low,high]
    span=np.nanmax(valid)-np.nanmin(valid)
    ax.set_ylim(np.nanmin(valid)-.12*span,np.nanmax(valid)+.12*span)
    ax.set_xlim(-40,56)
    ax.set_xticks([-40,-24,-8,0,16,32,48,56]); ax.tick_params(direction='out',length=2.1,pad=1.6,width=.7)
    if not show_x:ax.tick_params(axis='x',bottom=False,labelbottom=False)
    else:ax.set_xlabel('Days from CDH onset',labelpad=3)
    ax.spines[['top','right']].set_visible(False)
    ax.text(.015,.88,label,color='black',fontsize=7.75,fontweight='bold',ha='left',va='top',transform=ax.transAxes,
            bbox={'facecolor':'white','edgecolor':'none','alpha':.82,'pad':.3})
    label_ticks(ax)
    return ax


def main():
    src=pd.read_csv(SOURCE); timing=pd.read_csv(TIMING)
    vars={'Tmax','VPD','RootSM','ZSync','ZuWUE','ZNIRv','A_comp'}
    assert set(src.variable)==vars and len(src)==91
    assert np.array_equal(np.sort(src.relative_days.unique()),np.arange(-40,57,8))
    assert set(timing.variable)=={'ZSync','ZNIRv','ZuWUE','A_comp'}
    assert np.allclose(src.ci95_low,src['mean']-1.96*src.cluster_robust_se,atol=1e-12)
    assert np.allclose(src.ci95_high,src['mean']+1.96*src.cluster_robust_se,atol=1e-12)

    fig=plt.figure(figsize=(6.10,5.52),facecolor='white')
    # top strip panels: enough physical height for each phase response
    top_y=[.830,.718,.606]; strip_h=.091
    left=.090;right=.557;colw=.370
    fig.text(left,.977,'a  Hydroclimatic forcing',ha='left',va='top',fontsize=10.4,fontweight='bold')
    fig.text(right,.977,'b  Functional anomalies',ha='left',va='top',fontsize=10.4,fontweight='bold')
    aa=[];bb=[]
    for y,name,label,inv in zip(top_y,['Tmax','VPD','RootSM'],['Tmax','VPD','RootSM deficit'],[False,False,True]):
        aa.append(draw_strip(fig,[left,y,colw,strip_h],get(src,name),COLOR[name],label,inv,y==top_y[-1]))
    for y,name,label in zip(top_y,['ZSync','ZuWUE','ZNIRv'],['Synchrony','uWUE','NIRv']):
        bb.append(draw_strip(fig,[right,y,colw,strip_h],get(src,name),COLOR[name],label,False,y==top_y[-1]))
    fig.text(.036,.760,'Standardized climate response',rotation=90,ha='center',va='center',fontsize=8.7,fontweight='bold')
    fig.text(.501,.760,'Standardized functional state (Z)',rotation=90,ha='center',va='center',fontsize=8.7,fontweight='bold')

    # panel c: preserve ALL 13 raw A_comp phase means and frozen CI values.
    # Cohesive shape: one translucent continuous underlay, thin 95% CI band,
    # and a single piecewise-linear curve. No smoothing or interpolation.
    c=fig.add_axes([left,.125,colw,.322])
    q=get(src,'A_comp');x=q.relative_days.to_numpy(float)
    y=q['mean'].to_numpy(float);lo=q.ci95_low.to_numpy(float);hi=q.ci95_high.to_numpy(float)
    c.set_facecolor('#FCFCFC')
    c.axvspan(0,56,color='#FBEFEE',alpha=.65,zorder=0)
    c.fill_between(x,0.416,y,color=COLOR['A_comp'],alpha=.075,linewidth=0,zorder=1)
    c.fill_between(x,lo,hi,color=COLOR['A_comp'],alpha=.21,linewidth=0,zorder=2)
    c.plot(x,y,c=COLOR['A_comp'],lw=2.0,solid_joinstyle='round',
           solid_capstyle='round',zorder=3)
    # Highlight only onset and the descriptive ensemble maximum to avoid
    # a row of distracting circular markers on every 8-day phase.
    idx0=np.where(x==0)[0][0]; idxp=int(np.argmax(y))
    c.scatter([x[idx0],x[idxp]],[y[idx0],y[idxp]],s=22,c='white',
              edgecolors=COLOR['A_comp'],linewidths=1.2,zorder=4)
    c.axvline(0,c='#4C4C4C',lw=.85,ls=(0,(4,3)),zorder=4)
    c.axhline(.46,color='#E7E5E5',lw=.6,zorder=0)
    c.axhline(.50,color='#E7E5E5',lw=.6,zorder=0)
    c.axhline(.54,color='#E7E5E5',lw=.6,zorder=0)
    c.set_xlim(-40,56)
    c.set_xticks([-40,-24,-8,0,16,32,48,56]);c.set_ylim(.416,.550)
    c.set_xlabel('Days from CDH onset',labelpad=3)
    c.set_ylabel(r'Composite damage ($A_{comp}$)',labelpad=3)
    c.tick_params(direction='out',length=2.6,pad=2)
    c.spines[['top','right']].set_visible(False);label_ticks(c)
    fig.text(left,.474,'c  Composite functional damage',fontsize=10.4,fontweight='bold')

    # panel d: descriptive onset/peak/recovery landmarks from frozen table
    d=fig.add_axes([right,.153,colw,.284])
    order=['ZSync','ZNIRv','ZuWUE','A_comp']
    yy=[3,2,1,0]
    d.axvline(0,c='#444444',lw=.85,ls=(0,(4,3)))
    d.axvspan(-3,3,color='#F1F1F1',zorder=0)
    for y0,name in zip(yy,order):
        r=timing.loc[timing.variable.eq(name)].iloc[0]
        if name=='ZSync':
            d.plot(0,y0,marker='o',ms=6.3,mfc='white',mec=COLOR[name],mew=1.4)
            d.text(3,y0+.10,'centred Sync; lag not inferred',fontsize=7.0,color='#565656',va='center')
            continue
        onset=float(r.onset_lag_days) if pd.notna(r.onset_lag_days) else np.nan
        peak=float(r.peak_lag_days) if pd.notna(r.peak_lag_days) else np.nan
        rec=float(r.recovery_lag_days) if pd.notna(r.recovery_lag_days) else np.nan
        cens=str(r.recovery_censored).strip().lower() in {'true','1','yes'}
        if np.isfinite(onset): d.plot(onset,y0,marker='D',ms=4.7,mfc='white',mec=COLOR[name],mew=1.3,zorder=4)
        if np.isfinite(peak):
            d.plot([onset if np.isfinite(onset) else 0,peak],[y0,y0],c=COLOR[name],lw=1.8,alpha=.75)
            d.plot(peak,y0,marker='o',ms=5.0,mfc=COLOR[name],mec='white',mew=.45,zorder=5)
        if np.isfinite(rec):
            d.plot([peak,rec],[y0,y0],c=COLOR[name],lw=1.8,alpha=.70)
            d.plot(rec,y0,marker='|',ms=11,mew=1.7,c=COLOR[name])
        elif cens:
            d.annotate('',xy=(56,y0),xytext=(peak if np.isfinite(peak) else 0,y0),
                       arrowprops={'arrowstyle':'->','color':COLOR[name],'lw':1.7})
            d.text(55,y0+.22,'>56 d',fontsize=7.0,color='#4A4A4A',ha='right')
    d.set_xlim(-4,60);d.set_ylim(-.5,3.5);d.set_yticks(yy)
    d.set_yticklabels(['Sync','NIRv','uWUE',r'$A_{comp}$'])
    d.set_xticks([0,8,16,24,32,40,48,56]);d.set_xlabel('Days from CDH onset',labelpad=3)
    d.grid(axis='x',color='#E3E7E7',linewidth=.5)
    d.spines[['top','left','right']].set_visible(False)
    d.tick_params(direction='out',length=2.6,pad=2); label_ticks(d)
    fig.text(right,.474,'d  Descriptive response landmarks',fontsize=10.4,fontweight='bold')
    handles=[Line2D([],[],marker='D',color='#555555',mfc='white',lw=0,ms=4.7,label='Onset'),
             Line2D([],[],marker='o',color='#555555',lw=0,ms=5,label='Peak'),
             Line2D([],[],marker='|',color='#555555',lw=0,ms=9,label='Within-window recovery')]
    d.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,-.26),frameon=False,ncol=3,
             fontsize=7.25,handletextpad=.2,columnspacing=.7,borderaxespad=0)

    fig.canvas.draw()
    for ext in ['png','pdf','svg']:
        fig.savefig(str(STEM)+'.'+ext,dpi=600 if ext=='png' else 300,facecolor='white')
    plt.close(fig)

    qc=[f'Figure 4 R2 Word-size; frozen V3 input. Physical canvas: 6.10 × 5.52 in.',
        f'Source rows: {len(src)}; N nominal sampled catalogue events = 400,000 (archived report).',
        'All phase-wise means, CI bounds and event counts read unchanged from frozen CSV.',
        'Panel a: RootSM sign reversed for display as deficit (including bounds).',
        'Panel b: signed Z trajectories, no inferred Sync lag.',
        'Panel c: 13 original phase means retained; no smoothing; redesigned fill + fewer emphasis markers; Delta A_comp label UNVERIFIED, not used.',
        'Panel d: descriptive onset-aligned landmarks; NOT 120-day M5 KM/AJ recovery.',
        'Source interval=cluster-robust SE×1.96, calculation code UNVERIFIED.',
        f'SHA256 SRC {sha(SOURCE)}',f'SHA256 TIM {sha(TIMING)}',
        'No event recomputation or model retraining.']
    (OUT/'Figure4_R2_QC.txt').write_text('\n'.join(qc),encoding='utf-8')
    print('\n'.join(qc))

if __name__=='__main__':main()
