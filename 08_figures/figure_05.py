# -*- coding: utf-8 -*-
"""Figure 5 | Word-width edition R3.
Micro-tuning relative to R2: enlarge panels a and d slightly while preserving
all frozen data, four-panel structure, and Word-native insertion width.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa

ROOT=Path(__file__).resolve().parent
ENS=ROOT/'data'/'Figure5_CDH_State_Ensemble_V1_1.csv'
TRN=ROOT/'data'/'Figure5_CDH_Transition_Summary_V1_1.csv'
OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
STEM=OUT/'Figure5_WordSize_R3'

mpl.rcParams.update({
    'font.family':'serif',
    'font.serif':['Times New Roman','Liberation Serif','Nimbus Roman','DejaVu Serif'],
    'font.weight':'bold','axes.titleweight':'bold','axes.labelweight':'bold',
    'axes.edgecolor':'#262626','axes.linewidth':.85,
    'font.size':8.8,'axes.titlesize':10.4,'axes.labelsize':9.0,
    'xtick.labelsize':8.0,'ytick.labelsize':8.0,'legend.fontsize':8.2,
    'pdf.fonttype':42,'ps.fonttype':42,'mathtext.fontset':'stix',
    'savefig.facecolor':'white',
})
C={'Sync':'#65508F','uWUE':'#278B8E','NIRv':'#5A9656','Acomp':'#C5535D',
   'E1':'#4D8F78','E2':'#D6A13D','E3':'#D47749','E4':'#8F3E53'}
STAGES=['E1','E2','E3','E4']
TICKS=['Pre-event','Onset','Termination','Peak damage']
q=pd.read_csv(ENS).set_index('stage').loc[STAGES].reset_index()
_ = pd.read_csv(TRN)
assert q.A_comp_median.iloc[-1] > q.A_comp_median.iloc[0]

fig=plt.figure(figsize=(6.10,5.02),dpi=200,facecolor='white')

def panel_title(ax,letter,title):
    ax.set_title(letter+'  '+title,loc='left',pad=5.0,fontweight='bold',fontsize=10.5,color='black')

def boldticks(ax):
    for t in ax.get_xticklabels()+ax.get_yticklabels(): t.set_fontweight('bold')

# Manual layout: enlarge a and d slightly relative to b and c.
pos_a=[0.070,0.565,0.395,0.355]
pos_b=[0.575,0.565,0.390,0.345]
pos_c=[0.070,0.095,0.395,0.325]
pos_d=[0.575,0.090,0.390,0.350]

# a
x=q.ZSync_median.to_numpy(); y=q.ZuWUE_median.to_numpy(); z=q.ZNIRv_median.to_numpy()
a=fig.add_axes(pos_a,projection='3d')
a.plot(x,y,z,color='.35',lw=1.8)
for i,s in enumerate(STAGES):
    a.scatter(x[i],y[i],z[i],color=C[s],s=42,edgecolor='white',linewidth=.5,depthshade=False)
a.scatter([0],[0],[0],marker='x',c='black',s=30,linewidths=1.0)
xyz_off=[(.00,.013,.052),(.024,.000,.023),(-.045,-.055,.065),(-.025,-.024,.03)]
for i,s in enumerate(STAGES):
    dx,dy,dz=xyz_off[i]
    a.text(x[i]+dx,y[i]+dy,z[i]+dz,s,fontsize=8.7,color='black',fontweight='bold')
a.view_init(elev=22,azim=-55)
a.set_box_aspect((1.16,1.00,.90),zoom=1.08)
a.set_xlabel(r'$Z_{\mathrm{Sync}}$',labelpad=-1.2,fontsize=8.7)
a.set_ylabel(r'$Z_{\mathrm{uWUE}}$',labelpad=-1.8,fontsize=8.7)
a.set_zlabel(r'$Z_{\mathrm{NIRv}}$',labelpad=-1.6,fontsize=8.7)
a.xaxis.pane.fill=a.yaxis.pane.fill=a.zaxis.pane.fill=False
a.grid(True,alpha=.14)
a.set_xticks([-1.0,-0.5,0.0,0.25]); a.set_yticks([-0.6,-0.2,0.2]); a.set_zticks([-0.4,-0.2,0.0])
a.tick_params(labelsize=7.3,pad=0)
for axis in (a.xaxis,a.yaxis,a.zaxis):
    for lab in axis.get_ticklabels(): lab.set_fontweight('bold')
panel_title(a,'a','Functional-state migration')

# b
b=fig.add_axes(pos_b); xx=np.arange(4)
for col,key,label in [('ZSync_median','Sync','Synchrony'),('ZuWUE_median','uWUE','uWUE'),('ZNIRv_median','NIRv','NIRv')]:
    b.plot(xx,q[col],color=C[key],marker='o',markersize=4.2,lw=1.6,label=label)
b.axhline(0,color='.62',lw=.65); b.axhline(-.5,color='.60',lw=.65,ls='--')
b.set_xticks(xx,TICKS); b.set_xlim(-.12,3.12); b.set_ylim(-1.02,.40)
b.set_ylabel('Functional state (Z)',labelpad=3)
panel_title(b,'b','Functional dimensions')
b.legend(loc='lower left',bbox_to_anchor=(.00,.01),ncol=3,frameon=False,borderaxespad=0,fontsize=7.7,handlelength=1.0,columnspacing=.75,labelspacing=.2,handletextpad=.35)
b.spines[['top','right']].set_visible(False); b.tick_params(length=2.5,width=.8,pad=2); boldticks(b)

# c
c=fig.add_axes(pos_c)
m=q.A_comp_median.to_numpy(); lo=m-q.A_comp_p2_5.to_numpy(); hi=q.A_comp_p97_5.to_numpy()-m
c.errorbar(xx,m,yerr=np.vstack([lo,hi]),fmt='o-',color=C['Acomp'],lw=1.7,ms=4.5,capsize=2.1,elinewidth=.9)
for i,v in enumerate(m):
    c.annotate(f'{v:.2f}',(i,v),xytext=(0,4),textcoords='offset points',fontsize=8.2,fontweight='bold',ha='center',va='bottom')
c.set_xlim(-.12,3.12); c.set_ylim(-.12,2.56)
c.set_xticks(xx,TICKS); c.set_ylabel(r'Composite damage ($A_{comp}$)',labelpad=3)
panel_title(c,'c','Composite damage')
c.text(.02,.975,'Whiskers: event P2.5–P97.5',transform=c.transAxes,fontsize=7.5,fontweight='bold',va='top',ha='left',color='#363636')
c.spines[['top','right']].set_visible(False); c.tick_params(length=2.5,width=.8,pad=2); boldticks(c)

# d
d=fig.add_axes(pos_d); d.set_xlim(0,1); d.set_ylim(0,1); d.axis('off')
panel_title(d,'d','Observation windows')
def label(x,y,t,fs=8.4,ha='center'):
    return d.text(x,y,t,transform=d.transAxes,fontsize=fs,fontweight='bold',ha=ha,va='center',color='black')
label(.01,.91,'Onset = day 0',fs=9.3,ha='left')
d.add_patch(Rectangle((.055,.715),.34,.108,transform=d.transAxes,facecolor='#E8ECEA',edgecolor='none'))
d.add_patch(Rectangle((.395,.715),.55,.108,transform=d.transAxes,facecolor='#F2D9D2',edgecolor='none'))
label(.225,.767,'Baseline',fs=8.9); label(.665,.767,'Peak search',fs=8.9)
for x0 in [.055,.395,.945]: d.plot([x0,x0],[.705,.835],transform=d.transAxes,color='.37',lw=.72)
label(.055,.67,'−40 d',fs=7.9,ha='left'); label(.395,.67,'0',fs=8.1); label(.945,.67,'+56 d',fs=7.9,ha='right')

d.plot([.02,.98],[.586,.586],transform=d.transAxes,color='#D4D7D7',lw=.65)
label(.01,.535,'Termination = day 0',fs=9.3,ha='left')
d.add_patch(Rectangle((.055,.347),.890,.105,transform=d.transAxes,facecolor='#E7E1EC',edgecolor='none'))
label(.50,.399,'Formal recovery follow-up',fs=9.0)
for x0 in [.055,.945]: d.plot([x0,x0],[.335,.463],transform=d.transAxes,color='.37',lw=.72)
label(.055,.291,'0',fs=7.9,ha='left'); label(.945,.291,'+120 d',fs=7.9,ha='right')
label(.50,.205,'Recovery: 2 adjacent valid post-peak phases',fs=8.2)
label(.50,.130,'New extreme: KM censor / AJ competing',fs=8.0)
label(.50,.055,'Legacy: separate onset-based stopping rule',fs=8.0)

fig.canvas.draw(); render=fig.canvas.get_renderer(); coll=[]
for i,t in enumerate(d.texts):
    b1=t.get_window_extent(render).expanded(1.00,1.02)
    for tt in d.texts[i+1:]:
        if b1.overlaps(tt.get_window_extent(render).expanded(1.00,1.02)):
            coll.append((t.get_text(),tt.get_text()))
for ext in ['png','pdf','svg']:
    fig.savefig(str(STEM)+'.'+ext,dpi=600 if ext=='png' else 300,bbox_inches='tight',pad_inches=.035,facecolor='white')
plt.close(fig)
q.to_csv(str(STEM)+'_SourceData.csv',index=False)
with open(str(STEM)+'_QC.txt','w',encoding='utf-8') as f:
    f.write('FIGURE 5 WORD-WIDTH R3 | QC\n'+'='*50+'\n')
    f.write('Nominal canvas: 6.10in × 5.02in, intended insertion width 15–15.5cm.\n')
    f.write('Micro-tuning relative to R2: panel a and d enlarged slightly using manual axes positions.\n')
    f.write(f'E1–E4 Acomp medians: {list(m)}\n')
    f.write('a–c use frozen input unchanged; panel c whiskers are event P2.5–P97.5.\n')
    f.write('d: peak window onset+0..56 days; recovery window termination+0..120 days.\n')
    f.write(f'Panel-d text overlap flags: {len(coll)}\n')
    for x in coll: f.write('OVERLAP '+str(x)+'\n')
print('saved', STEM, 'overlaps', coll)
