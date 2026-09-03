"""
Three candidate redesigns of conceptual Figure 1, all in the lancet theme
(serif, matched palette, lowercase labels) so they sit with Figs 2-6.

  V1  fig1_v1_panels.png  - three clean panels (a) bistability, (b) mechanism, (c) surveillance
  V2  fig1_v2_flow.png    - single integrated left-to-right narrative
  V3  fig1_v3_data.png    - (a) REAL effective potential from data + (b) mechanism/surveillance schematic

Renders to fig1_previews/ for selection. Run from project root.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from scipy.stats import gaussian_kde
import pandas as pd

from config import DATA_FILE

OUT = Path('fig1_previews'); OUT.mkdir(exist_ok=True)

# ── lancet palette (matches regenerate_panels_v2.set_lancet_style) ──
CRIMSON='#B71C1C'; TEAL='#00695C'; AMBER='#E65100'; STEEL='#1565C0'
SLATE='#546E7A'; DARK='#212121'; GRAIN='#8D6E63'
CR_L='#FBE9E7'; ST_L='#E3F2FD'; AM_L='#FFF3E0'

plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'dejavuserif',
    'axes.edgecolor': '#444', 'axes.linewidth': 0.8,
    'savefig.facecolor': 'white',
})


def double_well(ax):
    """Stylised double-well potential at three PO4 regimes (shared by V1/V2)."""
    x = np.linspace(0, 3, 400)
    def V(x, tilt):
        v = 1.4*((x-0.5)**2)*((x-2.1)**2) - tilt*np.exp(-((x-2.1)/0.45)**2)
        return v - v.min()
    for label, tilt, c in [('Low PO$_4$ (<1 mg/L)',0.0,STEEL),
                           ('Mid PO$_4$ (1.5–2)',0.5,AMBER),
                           ('High PO$_4$ (>2 mg/L)',1.1,CRIMSON)]:
        v = V(x, tilt); v = v/v.max()
        ax.plot(x, v, color=c, lw=2.6, label=label)
    ax.text(0.5,0.05,'low-As\nmode',ha='center',va='bottom',fontsize=10,color=DARK,style='italic')
    ax.text(2.1,0.05,'high-As\nmode',ha='center',va='bottom',fontsize=10,color=DARK,style='italic')
    ax.axvline(1.0,color='#666',ls=':',lw=1.0)
    ax.text(1.0,1.03,'WHO 10 µg/L',ha='center',va='bottom',fontsize=9,color=DARK,fontweight='bold')
    ax.set_xlabel(r'log$_{10}$ As (µg/L)',fontsize=12)
    ax.set_ylabel('Conceptual stability landscape (a.u.)',fontsize=12)
    ax.set_xlim(0,3); ax.set_ylim(0,1.15); ax.tick_params(labelsize=10)
    ax.legend(loc='upper center',fontsize=9.5,frameon=False,bbox_to_anchor=(0.5,0.42))


def real_potential(ax):
    """REAL effective potential V=-log p(log10 As | PO4 tertile) from the data."""
    df = pd.read_csv(DATA_FILE)
    d = df[['As','PO43-']].dropna()
    d = d[d['As']>0]
    d['logAs'] = np.log10(d['As'])
    q = d['PO43-'].quantile([1/3,2/3]).values
    grid = np.linspace(d['logAs'].quantile(0.01), d['logAs'].quantile(0.99), 300)
    groups = [('Low PO$_4$',  d[d['PO43-']<=q[0]],            STEEL),
              ('Mid PO$_4$',  d[(d['PO43-']>q[0])&(d['PO43-']<=q[1])], AMBER),
              ('High PO$_4$', d[d['PO43-']>q[1]],             CRIMSON)]
    for label, g, c in groups:
        kde = gaussian_kde(g['logAs'])
        V = -np.log(kde(grid)); V = V - V.min()
        ax.plot(grid, V, color=c, lw=2.6, label=f'{label} (n={len(g)})')
    ax.axvline(1.0,color='#666',ls=':',lw=1.0)
    ymid = ax.get_ylim()[1]*0.55
    ax.text(0.92,ymid,'WHO 10 µg/L',rotation=90,ha='right',va='center',fontsize=9,color=DARK,fontweight='bold')
    ax.set_xlabel(r'log$_{10}$ As (µg/L)',fontsize=12)
    ax.set_ylabel(r'Effective potential  $-\log p$',fontsize=12)
    ax.tick_params(labelsize=10)
    ax.legend(loc='upper center',fontsize=9.5,frameon=False)


def mechanism(ax, title=True, aspect='equal'):
    ax.set_xlim(0,10); ax.set_ylim(0,10); ax.set_aspect(aspect); ax.axis('off')
    if title:
        ax.text(5.0,9.0,'Phosphate displaces arsenate from Fe-oxyhydroxide',
                ha='center',va='top',fontsize=11,color=DARK,style='italic')
    gy=5.2
    ax.add_patch(Circle((5.0,gy),1.6,facecolor=GRAIN,edgecolor='#4E342E',lw=1.5,zorder=2))
    ax.text(5.0,gy,'Fe-(oxy)\nhydroxide',ha='center',va='center',fontsize=11,fontweight='bold',color='white',zorder=3)
    ax.add_patch(Circle((1.5,gy),0.5,facecolor=STEEL,edgecolor=DARK,lw=0.6,zorder=3))
    ax.text(1.5,gy,r'PO$_4$',fontsize=10,color='white',ha='center',va='center',zorder=4,fontweight='bold')
    ax.annotate('',xy=(3.2,gy),xytext=(2.1,gy),arrowprops=dict(arrowstyle='-|>',color=STEEL,lw=2.4))
    ax.text(2.65,gy+0.6,'binds',ha='center',fontsize=10,color=STEEL,style='italic')
    ax.annotate('',xy=(8.2,gy),xytext=(6.7,gy),arrowprops=dict(arrowstyle='-|>',color=CRIMSON,lw=2.4))
    ax.add_patch(Circle((8.7,gy),0.5,facecolor=CRIMSON,edgecolor=DARK,lw=0.6,zorder=3))
    ax.text(8.7,gy,'As',fontsize=10,color='white',ha='center',va='center',zorder=4,fontweight='bold')
    ax.text(7.4,gy+0.6,'released',ha='center',fontsize=10,color=CRIMSON,style='italic')
    ax.text(5.0,2.0,r'$\equiv$Fe–OAs $+$ H$_2$PO$_4^-$ $\rightarrow$ $\equiv$Fe–OPO$_3$H$^-$ $+$ As$_{(aq)}$',
            ha='center',va='center',fontsize=11,color=DARK,
            bbox=dict(boxstyle='round,pad=0.45',facecolor='#FAFAFA',edgecolor='#BBB',lw=0.8))


def surveillance(ax, title=True, aspect='equal'):
    ax.set_xlim(0,10); ax.set_ylim(0,10); ax.set_aspect(aspect); ax.axis('off')
    ax.text(2.3,9.0,'Arsenic surveillance',ha='center',fontsize=11,fontweight='bold',color=CRIMSON)
    ax.add_patch(FancyBboxPatch((0.3,3.6),4.0,4.7,boxstyle="round,pad=0.1",facecolor=CR_L,edgecolor=CRIMSON,lw=1.5))
    for i,l in enumerate(['Lab AAS / ICP-MS','Field kit \\$5–15/test','Sample transport','Days–weeks']):
        ax.text(0.65,7.7-i*0.62,'•',fontsize=11,color=CRIMSON,va='center')
        ax.text(0.95,7.7-i*0.62,l,fontsize=9.5,va='center')
    ax.text(2.3,4.6,r'$\sim$\$5–15 / test',ha='center',fontsize=11,fontweight='bold',color=CRIMSON)
    ax.text(2.3,4.0,'low rural coverage',ha='center',fontsize=9,style='italic',color='#666')
    ax.annotate('',xy=(5.55,5.9),xytext=(4.45,5.9),arrowprops=dict(arrowstyle='-|>',color=DARK,lw=1.8))
    ax.text(5.0,6.45,'proxy',ha='center',fontsize=10,style='italic',color=DARK)
    ax.text(7.7,9.0,'Phosphate proxy',ha='center',fontsize=11,fontweight='bold',color=STEEL)
    ax.add_patch(FancyBboxPatch((5.7,3.6),4.0,4.7,boxstyle="round,pad=0.1",facecolor=ST_L,edgecolor=STEEL,lw=1.5))
    for i,l in enumerate(['Colorimetric strip','Field kit \\$0.30–1/test','In situ, no transport','Minutes']):
        ax.text(6.05,7.7-i*0.62,'•',fontsize=11,color=STEEL,va='center')
        ax.text(6.35,7.7-i*0.62,l,fontsize=9.5,va='center')
    ax.text(7.7,4.6,r'$\sim$10$\times$ cheaper',ha='center',fontsize=11,fontweight='bold',color=STEEL)
    ax.text(7.7,4.0,'full WASH network',ha='center',fontsize=9,style='italic',color='#666')
    # R1.3: phosphate is a PRIORITISING filter, not a replacement test. The old
    # banner read 'well likely high-As', which overstates a screen with sensitivity
    # 0.57 and invites exactly the false reassurance the reviewer warned about.
    ax.text(5.0,1.9,r'PO$_4$ $>$ 1.5–2.0 mg/L $\Rightarrow$ prioritise for arsenic testing',
            ha='center',fontsize=10,fontweight='bold',color=DARK,
            bbox=dict(boxstyle='round,pad=0.4',facecolor=AM_L,edgecolor=AMBER,lw=1.2))
    ax.text(5.0,1.15,'sensitivity 0.57: a negative result does not clear a well',
            ha='center',fontsize=9,style='italic',color='#666')


def lab(ax, s, x=0.0, y=1.04, fontsize=13):
    ax.text(x,y,s,transform=ax.transAxes,ha='left',va='bottom',fontsize=fontsize,fontweight='bold',color=DARK)


def save(fig, name):
    fig.savefig(OUT/name, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig); print('  OK', name)


# ── V1: three clean panels ──
def v1():
    fig = plt.figure(figsize=(16,5.2))
    gs = fig.add_gridspec(1,3,width_ratios=[1.05,1.0,1.05],wspace=0.28)
    a=fig.add_subplot(gs[0,0]); double_well(a); lab(a,'(a) Phosphate-controlled bistability')
    b=fig.add_subplot(gs[0,1]); mechanism(b); lab(b,'(b) Desorption mechanism', x=0.12)
    c=fig.add_subplot(gs[0,2]); surveillance(c); lab(c,'(c) Surveillance reframing', x=0.05)
    save(fig,'fig1_v1_panels.png')


# ── V2: single integrated flow ──
def v2():
    fig, ax = plt.subplots(figsize=(16,5.0)); ax.set_xlim(0,30); ax.set_ylim(0,10); ax.axis('off')
    def stage_arrow(x): ax.annotate('',xy=(x+0.9,5),xytext=(x,5),arrowprops=dict(arrowstyle='-|>',color=SLATE,lw=2.5))
    # 1 inputs
    ax.add_patch(FancyBboxPatch((0.2,3.4),3.0,3.2,boxstyle="round,pad=0.1",facecolor=ST_L,edgecolor=STEEL,lw=1.4))
    ax.text(1.7,5.0,'Monsoon recharge\n+ fertiliser\n$\\Rightarrow$ PO$_4$ rises',ha='center',va='center',fontsize=10.5,color=DARK)
    stage_arrow(3.4)
    # 2 mechanism (mini grain)
    gy=5
    ax.add_patch(Circle((6.6,gy),1.25,facecolor=GRAIN,edgecolor='#4E342E',lw=1.4,zorder=2))
    ax.text(6.6,gy,'Fe-ox',ha='center',va='center',fontsize=9.5,color='white',fontweight='bold',zorder=3)
    ax.add_patch(Circle((5.1,gy),0.4,facecolor=STEEL,edgecolor=DARK,lw=0.5,zorder=3)); ax.text(5.1,gy,r'PO$_4$',fontsize=8,color='white',ha='center',va='center',zorder=4,fontweight='bold')
    ax.annotate('',xy=(8.6,gy+0.2),xytext=(7.7,gy+0.2),arrowprops=dict(arrowstyle='-|>',color=CRIMSON,lw=2.0))
    ax.add_patch(Circle((9.0,gy+0.2),0.4,facecolor=CRIMSON,edgecolor=DARK,lw=0.5,zorder=3)); ax.text(9.0,gy+0.2,'As',fontsize=8,color='white',ha='center',va='center',zorder=4,fontweight='bold')
    ax.text(6.8,2.7,'PO$_4$ desorbs As\nfrom Fe-oxyhydroxide',ha='center',va='top',fontsize=10,style='italic',color=DARK)
    stage_arrow(10.0)
    # 3 bistability mini double well
    axins = ax.inset_axes([11.2/30,0.30,5.8/30,0.5])
    double_well(axins); axins.legend().remove(); axins.set_xlabel('log$_{10}$ As',fontsize=9); axins.set_ylabel('$V$',fontsize=9); axins.tick_params(labelsize=7)
    axins.set_title('bistable regimes',fontsize=10,color=DARK)
    stage_arrow(17.4)
    # 4 threshold
    ax.add_patch(FancyBboxPatch((18.5,3.4),4.3,3.2,boxstyle="round,pad=0.1",facecolor=AM_L,edgecolor=AMBER,lw=1.4))
    ax.text(20.65,5.0,'PO$_4$ $>$ 1.5–2.0\nmg/L flags\nhigh-As mode',ha='center',va='center',fontsize=10.5,color=DARK,fontweight='bold')
    stage_arrow(23.0)
    # 5 surveillance payoff
    ax.add_patch(FancyBboxPatch((23.7,3.4),5.8,3.2,boxstyle="round,pad=0.1",facecolor=ST_L,edgecolor=STEEL,lw=1.4))
    ax.text(26.6,5.0,'Cheap PO$_4$ field strip\n($\\sim$10$\\times$ cheaper than As lab)\n$\\Rightarrow$ full WASH coverage',ha='center',va='center',fontsize=10.5,color=DARK)
    ax.text(0.0,9.4,'From phosphate loading to low-cost arsenic surveillance',
            ha='left',va='top',fontsize=14,fontweight='bold',color=DARK)
    save(fig,'fig1_v2_flow.png')


# ── V3: single-panel surveillance figure ──
def v3():
    # Figure 1 for the JHMA revision (2026-09-01). The stylised double-well
    # schematic that was panel (a) has been REMOVED on reviewer R1.5's request:
    # it was not computed from data and duplicated the empirical effective
    # potential in Fig 2d, which now carries that argument alone. Under the
    # de-escalation of the bistability framing (R1.1/R2.2) a stability-landscape
    # cartoon no longer has a claim to make. Figure 1 is now the surveillance
    # reframing only.
    fig = plt.figure(figsize=(9.2,5.6))
    ax = fig.add_subplot(1,1,1)
    surveillance(ax,title=False,aspect='auto')
    # FINAL: V3 is the chosen Figure 1 -> export at 300 dpi to the package
    for d in [Path('Draft/STOTENSubmission/submission'), Path('Draft/STOTENSubmission/source')]:
        fig.savefig(d/'figure1_conceptual.png', dpi=300, bbox_inches='tight', facecolor='white')
    save(fig,'fig1_v3_data.png')


if __name__=='__main__':
    print('Building Figure 1 variants ->', OUT)
    v1(); v2(); v3()
    print('done.')
