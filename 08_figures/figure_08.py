#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ecological Indicators Figure 8 R1 | Locked Figure8 V7 / L6.0 FINAL V2.
Draw-only: original CSV values, sample support, confidence limits and color scale
are preserved.  Run: python Figure8_R1_Plot.py [--exact-font]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.font_manager import fontManager

BASE = Path(__file__).resolve().parent
DATA, OUT = BASE / 'data', BASE / 'outputs'
FILES = ('F8_A_RootSM_Greening.csv', 'F8_B_RootSM_Contrast.csv', 'F8_C_ResponseMatrix.csv')
# Original V7 colorbar: P98(abs(matrix)), symmetrical about zero. Reproduce exactly.
VMAX = 0.083110184122127326
NODE_COLORS = ['#2E5872','#7FA7B4','#D8E4E3','#F5F2E9',
               '#E6BD8F','#CC7856','#903F37']
REGIME = {'Dry': '#2E7286', 'Intermediate': '#A68445', 'Wet': '#BE5C4B'}
DARK = '#1E2930'


def check_data(a, b, c):
    assert len(a) == 30 and len(b) == 10 and len(c) == 100
    assert set(a.moisture_regime) == set(REGIME)
    assert all(set(a.loc[a.moisture_regime == r, 'greening_decile']) == set(range(1,11)) for r in REGIME)
    assert set(b.RootSM_decile) == set(range(1, 11))
    assert len(c[['RootSM_decile','greening_decile']].drop_duplicates()) == 100
    assert set(c.RootSM_decile) == set(range(1,11)) and set(c.greening_decile) == set(range(1,11))
    assert a.damage_n.sum() == c.n.sum() == 153932
    for d, center, se in [(a,'damage_residual_mean','damage_residual_SE'),
                          (b,'greening_resilience_benefit','greening_resilience_benefit_SE')]:
        np.testing.assert_allclose(d.CI_lower, d[center] - 1.96 * d[se], atol=2e-14, rtol=0)
        np.testing.assert_allclose(d.CI_upper, d[center] + 1.96 * d[se], atol=2e-14, rtol=0)
        assert ((d.CI_lower <= d[center]) & (d.CI_upper >= d[center])).all()
    np.testing.assert_allclose(b.greening_resilience_benefit,
                               b.low_greening_damage - b.high_greening_damage,
                               atol=2e-14, rtol=0)
    vals = c.damage_residual.dropna().to_numpy(dtype=float)
    assert vals.size == 100
    np.testing.assert_allclose(np.percentile(np.abs(vals), 98), VMAX, atol=2e-14, rtol=0)
    assert (b.greening_resilience_benefit > 0).sum() == 5
    assert (b.greening_resilience_benefit < 0).sum() == 5
    supported = b.loc[(b.CI_lower > 0) | (b.CI_upper < 0), 'RootSM_decile'].tolist()
    assert supported == [4,5,8,9], f'Unexpected CI pattern: {supported}'
    return supported


def set_font(exact=False):
    fonts = {f.name for f in fontManager.ttflist}
    if exact and 'Times New Roman' not in fonts:
        raise RuntimeError('Times New Roman not available. Install/use it on Windows, or run without --exact-font for a preview.')
    family = 'Times New Roman' if 'Times New Roman' in fonts else 'Liberation Serif'
    mpl.rcParams.update({
        'font.family': family, 'font.size': 9.0, 'font.weight': 'bold',
        'axes.labelsize': 9.1, 'axes.labelweight': 'bold',
        'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
        'axes.edgecolor': DARK, 'axes.linewidth': .8,
        'text.color': 'black', 'axes.labelcolor': 'black',
        'xtick.color': 'black', 'ytick.color': 'black',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white',
    })
    return family


def format_ax(ax, grid_axis='y'):
    ax.set_facecolor('#FAFBFC')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(length=3.1, width=.78, direction='out', pad=4)
    if grid_axis:
        ax.grid(axis=grid_axis, color='#CDD4D9', linewidth=.48,
                linestyle=(0, (2.3,3.3)), alpha=.62, zorder=0)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight('bold')


def figure(a, b, c, family):
    # 16.5-cm two-column journal figure. Fixed axes boxes avoid title overlap.
    fig = plt.figure(figsize=(16.5/2.54, 14.2/2.54), facecolor='white')
    ax_a = fig.add_axes([.125, .654, .835, .230])
    ax_b = fig.add_axes([.125, .148, .372, .330])
    ax_c = fig.add_axes([.606, .148, .294, .330])
    cb_ax= fig.add_axes([.917, .174, .016, .280])

    # (a) Original pixels-weighted means and +/-1.96x original 20x20-block SE.
    format_ax(ax_a)
    for group in REGIME:
        d = a.loc[a.moisture_regime == group].sort_values('greening_decile')
        x=d.greening_decile.to_numpy(float)
        mean=d.damage_residual_mean.to_numpy(float)
        lower=d.CI_lower.to_numpy(float)
        upper=d.CI_upper.to_numpy(float)
        color=REGIME[group]
        ax_a.fill_between(x, lower, upper, color=color, alpha=.13, linewidth=0, zorder=1)
        ax_a.plot(x, mean, '-o', color=color, linewidth=1.85, markersize=4.4,
                  markeredgewidth=1.0, markeredgecolor=color, markerfacecolor='white',
                  solid_capstyle='round', zorder=4, label=group)
    ax_a.axhline(0,color='#3A4146',lw=.95,zorder=2)
    ax_a.set_xlim(.72,10.28);ax_a.set_ylim(-.105,.090)
    ax_a.set_xticks(range(1,11)); ax_a.yaxis.set_major_locator(MultipleLocator(.05))
    ax_a.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax_a.set_xlabel('Greening decile',labelpad=4)
    ax_a.set_ylabel('Severity-adjusted residual damage',labelpad=8)
    fig.text(.125,.943,'(a) Greening–damage responses across moisture regimes',
             ha='left',va='center',fontsize=10.2,fontweight='bold')
    fig.text(.955,.943,'CDH  |  n = 153,932 pixels',
             ha='right',va='center',fontsize=8.35,fontweight='bold',color='black')
    fig.legend(handles=[Line2D([0],[0],color=REGIME[g],lw=1.9,marker='o',markersize=4,
                               markerfacecolor='white',label=g) for g in REGIME],
               loc='center',bbox_to_anchor=(.78,.910),bbox_transform=fig.transFigure,
               ncol=3,frameon=False,handlelength=2.0,columnspacing=1.35,
               handletextpad=.45,fontsize=8.5)

    # (b) Original low-greening minus high-greening *descriptive* contrasts.
    format_ax(ax_b)
    ax_b.axhline(0,color='#3A4146',lw=1.05,zorder=2)
    ax_b.set_xlim(.62,10.38);ax_b.set_ylim(-.118,.120)
    ax_b.set_xticks(range(1,11));ax_b.yaxis.set_major_locator(MultipleLocator(.05))
    ax_b.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    for _,r in b.sort_values('RootSM_decile').iterrows():
        x=float(r.RootSM_decile); y=float(r.greening_resilience_benefit)
        col = '#2E7286' if y>=0 else '#BE5C4B'
        solid = (r.CI_lower > 0) or (r.CI_upper < 0)
        ax_b.errorbar(x,y,yerr=np.array([[y-r.CI_lower],[r.CI_upper-y]]),
                      fmt='none',elinewidth=1.17,ecolor='#56636B',capsize=2.8,
                      capthick=1.0,zorder=3)
        ax_b.scatter([x],[y],s=31 if solid else 29,facecolor=col if solid else 'white',
                     edgecolor=col,linewidth=1.2,zorder=5)
    ax_b.set_xlabel('Root-zone soil moisture decile',labelpad=5)
    ax_b.set_ylabel('Δ residual damage (low − high)',labelpad=7)
    fig.text(.125,.534,'(b) Empirical greening contrast',
             ha='left',va='center',fontsize=10.2,fontweight='bold')
    # Note lives between title and plot, clear of error bars and graph limits.
    fig.text(.497,.505,'Filled markers: 95% CI excludes zero',
             ha='right',va='center',fontsize=7.35,color='black',fontweight='bold')

    # (c) Original 100 values; exact V7 7-stop diverging palette and saturation.
    matrix=c.pivot(index='RootSM_decile',columns='greening_decile',values='damage_residual')
    matrix=matrix.reindex(index=range(1,11),columns=range(1,11))
    cmap=LinearSegmentedColormap.from_list('Figure8_V7_seven_nodes', NODE_COLORS, N=256)
    cmap.set_bad('#ECEFF0')
    norm=TwoSlopeNorm(vmin=-VMAX,vcenter=0.0,vmax=VMAX)
    image=ax_c.imshow(matrix.values,origin='lower',interpolation='nearest',
                      cmap=cmap,norm=norm,aspect='equal',extent=(.5,10.5,.5,10.5))
    ax_c.set_facecolor('white')
    ax_c.set_xlim(.5,10.5);ax_c.set_ylim(.5,10.5)
    ax_c.set_xticks(range(1,11));ax_c.set_yticks(range(1,11))
    ax_c.set_xticks(np.arange(1.5,10.5,1),minor=True)
    ax_c.set_yticks(np.arange(1.5,10.5,1),minor=True)
    ax_c.grid(which='minor',linewidth=.28,color='white',alpha=.33)
    ax_c.tick_params(which='minor',bottom=False,left=False)
    ax_c.tick_params(which='major',length=3.1,width=.8,pad=3)
    ax_c.spines['top'].set_visible(False);ax_c.spines['right'].set_visible(False)
    ax_c.set_xlabel('Greening decile',labelpad=5)
    ax_c.set_ylabel('Root-zone soil moisture decile',labelpad=7)
    for tick in ax_c.get_xticklabels()+ax_c.get_yticklabels():tick.set_fontweight('bold')
    fig.text(.606,.534,'(c) Joint moisture–greening response',
             ha='left',va='center',fontsize=10.2,fontweight='bold')
    cbar=fig.colorbar(image,cax=cb_ax,ticks=[-.08,-.04,0,.04,.08])
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    cbar.ax.tick_params(length=2.8,width=.8,labelsize=8.0,pad=3)
    for tick in cbar.ax.get_yticklabels():tick.set_fontweight('bold')
    cbar.outline.set_linewidth(.75)
    fig.text(.927,.479,'Residual',ha='center',va='center',
             fontsize=7.35,fontweight='bold',color='black')
    # The common response is dimensionless; bottom note stays outside panels.
    fig.text(.125,.047,'Positive residual = greater damage  |  Negative residual = lower damage',
             ha='left',va='center',fontsize=7.7,color='black',fontweight='bold')
    return fig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exact-font',action='store_true',help='Require Times New Roman for final Windows export')
    args=parser.parse_args()
    a,b,c=[pd.read_csv(DATA/f,encoding='utf-8-sig') for f in FILES]
    supported=check_data(a,b,c)
    font=set_font(args.exact_font)
    OUT.mkdir(parents=True,exist_ok=True)
    fig=figure(a,b,c,font)
    for ext,params in [('pdf',{}),('png',{'dpi':600})]:
        dest=OUT/f'Figure8_R1.{ext}'
        fig.savefig(dest,format=ext,facecolor='white',**params)
        print('SAVED:',dest)
    plt.close(fig)
    qc='\n'.join([
        'Figure 8 R1 | PLOT-ONLY QC',
        'Science source: Figure8 FINAL V7 / L6.0 FINAL V2',
        'A: 30 groups, B: 10 groups, C: 100 finite bins',
        'Pixels in A and C: 153,932',
        'A/B: original mean +/- 1.96*original spatial-block SE (not bootstrap)',
        f'B original 95% CI excludes zero at RootSM deciles: {supported}',
        'B point sign: 5 positive, 5 negative; neither marker color proves causality',
        f'C display bounds: {-VMAX:.17f} to {VMAX:.17f}; raw values not clipped',
        'No recalculation, smoothing, interpolation, inference, model fitting or original data writes',
        f'FONT: {font}' + (' (exact)' if args.exact_font else (' (local preview fallback)' if font!='Times New Roman' else '')),
        'STATUS: PLOT PASS (scientific source preserved)',
    ])+'\n'
    (OUT/'Figure8_R1_QC.txt').write_text(qc,encoding='utf-8')
    print(qc)

if __name__=='__main__':main()
