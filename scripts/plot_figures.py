"""Static PNG/PDF figures of the benchmark summaries. Reads the report CSVs; no benchmark is rerun."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, PercentFormatter, NullFormatter
from matplotlib.transforms import offset_copy

ROOT=Path(__file__).resolve().parents[1]

# Chart tokens. Classical methods are gray context; the three differentiable families
# take the three hues that stay distinct for all pairs, including under colour-vision deficiency.
SURFACE,INK,INK2,MUTED,GRID,ROWGRID,AXIS,DARK='#fcfcfb','#0b0b0b','#52514e','#898781','#e1e0d9','#efeee9','#c3c2b7','#4a4946'
BLUE,ORANGE,AQUA='#2a78d6','#eb6834','#1baf7a'
PLUM='#a33ea1'  # fourth hue: passes all-pairs CVD and normal-vision checks against the three above
FAMILY={'moment':('Moment matching',BLUE,'^'),'soft':('Soft / Gumbel relaxation',ORANGE,'D'),
        'ot':('Optimal transport',AQUA,'s'),'variational':('Variational / deterministic',PLUM,'X'),'classical':('Classical',MUTED,'o')}
FAMILY_ORDER=list(FAMILY)
PROVIDER={'custom':'Local','stonesoup':'Stone Soup','particles':'particles','julia':'Resamplings.jl','pfjax':'PFJAX',
          'cuthbert':'Cuthbert','pydpf':'PyDPF','torchfilter':'torchfilter','filterflow':'FilterFlow','variational':'Variational-Resampling'}
METHOD={'systematic_stop':'systematic + stop-grad','corrected_ot':'corrected OT','ot':'OT','ssp':'SSP','mvn':'MVN',
        'gaussian':'Gaussian','gumbel':'Gumbel-softmax','kl':'KL greedy','tv':'TV rounding'}
# Representative series for line charts: (colour, marker). Two classical baselines share gray.
REPS={('particles','multinomial'):(MUTED,'o'),('stonesoup','systematic'):(MUTED,'v'),('pydpf','soft'):(ORANGE,'D'),
      ('custom','gumbel'):(ORANGE,'P'),('pfjax','ot'):(AQUA,'s'),('custom','eigen'):(BLUE,'^'),('custom','gaussian'):(BLUE,'h'),
      ('variational','kl'):(PLUM,'X')}
GRAD={('custom','eigen'):(BLUE,'^'),('custom','gaussian'):(BLUE,'h'),('pydpf','soft'):(ORANGE,'D'),('custom','gumbel'):(ORANGE,'P'),
      ('pydpf','ot'):(AQUA,'s'),('pydpf','systematic'):(MUTED,'v'),('pydpf','systematic_stop'):(DARK,'>')}
NS=[32,128,512,2048]


def family(method):
    if method in ('soft','gumbel'):return 'soft'
    if method in ('ot','corrected_ot'):return 'ot'
    if method in ('eigen','gaussian','mvn'):return 'moment'
    if method in ('kl','tv'):return 'variational'
    return 'classical'


def colour(method):return DARK if method=='systematic_stop' else FAMILY[family(method)][1]
def marker(method):return FAMILY[family(method)][2]
def label(provider,method):return f'{PROVIDER.get(provider,provider)} · {METHOD.get(method,method)}'


def ordered(df):
    key=df.method.map(family).map(FAMILY_ORDER.index)
    return df.assign(_f=key).sort_values(['_f','method','provider']).drop(columns='_f').reset_index(drop=True)


def style():
    plt.rcParams.update({'font.family':['Noto Sans','DejaVu Sans'],'font.size':9,'figure.facecolor':SURFACE,'axes.facecolor':SURFACE,
        'savefig.facecolor':SURFACE,'axes.edgecolor':AXIS,'axes.linewidth':.8,'axes.spines.top':False,'axes.spines.right':False,
        'axes.grid':True,'axes.axisbelow':True,'grid.color':GRID,'grid.linewidth':.6,'grid.linestyle':'-',
        'xtick.color':AXIS,'ytick.color':AXIS,'xtick.labelcolor':INK2,'ytick.labelcolor':INK2,'xtick.major.size':3,'ytick.major.size':0,
        'xtick.minor.size':0,'ytick.minor.size':0,'axes.labelcolor':INK2,'axes.labelsize':8.5,'text.color':INK,'axes.titlesize':9.5,
        'axes.titleweight':'bold','axes.titlelocation':'left','axes.titlecolor':INK,'legend.frameon':False,'legend.fontsize':8.5,
        'lines.linewidth':1.6,'lines.solid_capstyle':'round','lines.solid_joinstyle':'round','pdf.fonttype':42,'svg.fonttype':'none'})


def header(fig,title,subtitle):fig._header=(title,subtitle)


def place_header(fig):
    """Stack subtitle and title above the top legend, measured after layout."""
    fig.canvas.draw();inv=fig.transFigure.inverted()
    top=max([1]+[inv.transform(l.get_window_extent())[1,1] for l in fig.legends])
    title,subtitle=fig._header;lines=subtitle.count('\n')+1
    fig.text(0,top,subtitle,transform=offset_copy(fig.transFigure,fig=fig,y=10,units='points'),va='bottom',color=INK2,fontsize=9,linespacing=1.4)
    fig.text(0,top,title,transform=offset_copy(fig.transFigure,fig=fig,y=14+15*lines,units='points'),va='bottom',fontsize=13,weight='bold')


def top_legend(fig,handles,labels=None,ncol=5):
    labels=labels if labels is not None else [h.get_label() for h in handles]
    fig.legend(handles,labels,loc='lower left',bbox_to_anchor=(0,1),bbox_transform=offset_copy(fig.transFigure,fig=fig,y=8,units='points'),
               ncol=ncol,borderaxespad=0,borderpad=0,handletextpad=.3,columnspacing=1.6)


def footer(fig,text):
    fig.text(0,0,text,transform=offset_copy(fig.transFigure,fig=fig,y=-10,units='points'),va='top',color=MUTED,fontsize=7.5,linespacing=1.4)


def save(fig,out,name):
    place_header(fig)
    for ext in ('png','pdf'):fig.savefig(out/f'{name}.{ext}',dpi=200,bbox_inches='tight',pad_inches=.3)
    plt.close(fig);print(f'wrote {name}')


def family_legend(fig,methods,ncol=5,extra=()):
    handles=[Line2D([],[],ls='none',marker=FAMILY[f][2],ms=6.5,color=FAMILY[f][1],mec=SURFACE,label=FAMILY[f][0])
             for f in FAMILY_ORDER if any(family(m)==f for m in methods)]
    if 'systematic_stop' in set(methods):
        handles.append(Line2D([],[],ls='none',marker='o',ms=6.5,color=DARK,mec=SURFACE,label='Classical + stop-gradient'))
    top_legend(fig,handles+list(extra),ncol=ncol)


def ms_format(v,_):return f'{v/1000:.3g} s' if v>=1000 else f'{v:.3g}'


def log_format(v,_):return '' if v<=0 else f'{v:g}' if 1e-3<=v<1e5 else f'1e{round(np.log10(v))}'


def plain_log(axis):axis.set_major_formatter(FuncFormatter(log_format));axis.set_minor_formatter(NullFormatter())


def n_axis(ax):
    ax.set_xscale('log',base=2);ax.set_xticks(NS,[str(n) for n in NS]);ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(24,2700)


def dots(ax,df,col,half=None,log=False,floor=None,pct=False):
    """One row per method: CI whisker plus dot. Values below `floor` are drawn at the floor and labelled."""
    for y,(_,r) in enumerate(df.iterrows()):
        v=r.get(col,np.nan)
        if not np.isfinite(v):continue
        c,m=colour(r.method),marker(r.method)
        clipped=floor is not None and v<floor
        x=floor if clipped else v
        h=r.get(half,np.nan) if half else np.nan
        if np.isfinite(h) and h>0 and not clipped:
            lo,hi=v-h,v+h
            if log:lo=max(lo,v/20)
            if pct:lo,hi=max(lo,0),min(hi,1)
            ax.plot([lo,hi],[y,y],color=c,lw=1.4,alpha=.45,zorder=2)
        ax.plot(x,y,ls='none',marker='<' if clipped else m,ms=6,color=c,mec=SURFACE,mew=1,zorder=3)
        if clipped:ax.annotate(f'{v:.0e}',(x,y),xytext=(7,0),textcoords='offset points',va='center',fontsize=7.5,color=INK2)
    if log:ax.set_xscale('log');plain_log(ax.xaxis)
    if pct:ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    ax.set_ylim(len(df)-.5,-.5);ax.grid(axis='y',color=ROWGRID)


def row_labels(ax,df):ax.set_yticks(range(len(df)),[label(p,m) for p,m in zip(df.provider,df.method)],fontsize=8)


def family_breaks(axes,df):
    fam=df.method.map(family).tolist()
    for i in range(1,len(fam)):
        if fam[i]!=fam[i-1]:
            for ax in axes:ax.axhline(i-.5,color=AXIS,lw=.8,zorder=1)


def rep_lines(ax,df,x,y,half=None):
    for k,((p,m),(c,mk)) in enumerate(REPS.items()):
        part=df[(df.provider==p)&(df.method==m)].sort_values(x)
        if part.empty or part[y].isna().all():continue
        xs=part[x]*2**((k-len(REPS)//2)*.05)  # small horizontal dodge keeps overlapping whiskers apart
        ax.plot(xs,part[y],color=c,marker=mk,ms=6,mec=SURFACE,mew=1,label=label(p,m),zorder=3)
        if half:ax.errorbar(xs,part[y],yerr=part[half],fmt='none',ecolor=c,elinewidth=1.2,alpha=.45,capsize=0,zorder=2)


# ---------------------------------------------------------------- static resampling

def fig_mixture(s,out):
    df=s.query("cloud=='mixture' and n==128 and dim==2").sort_values(['sliced_w1_mean','provider']).reset_index(drop=True)
    fig,axes=plt.subplots(1,4,figsize=(12.5,9.6),sharey=True,layout='constrained')
    dots(axes[0],df,'sliced_w1_mean','sliced_w1_ci95_half');axes[0].set(title='Sliced W1',xlabel='lower is better')
    dots(axes[1],df,'covariance_error_mean','covariance_error_ci95_half');axes[1].set(title='Covariance error (Frobenius)',xlabel='lower is better')
    for y,v in enumerate(df.covariance_error_mean):
        if v<1e-6:axes[1].annotate(f'{v:.0e}',(v,y),xytext=(7,0),textcoords='offset points',va='center',fontsize=7.5,color=INK2)
    dots(axes[2],df,'mode_coverage_mean','mode_coverage_ci95_half',pct=True);axes[2].set(title='Trials keeping both modes',xlabel='higher is better')
    dots(axes[3],df,'forward_ms_mean','forward_ms_ci95_half',log=True);axes[3].set(title='Forward time, ms (log)',xlabel='lower is better')
    axes[3].xaxis.set_major_formatter(FuncFormatter(ms_format));row_labels(axes[0],df)
    family_legend(fig,df.method)
    header(fig,'Gaussian-mixture resampling: distribution error, moments and cost',
           'Fixed bimodal weighted cloud · N = 128 · d = 2 · rows sorted by sliced W1 · whiskers are 95% intervals over resampling seeds (conditional on the cloud)')
    footer(fig,'Source: static_summary.csv. Fast methods 100 trials, OT/Gumbel 10. Forward time from the serial timing pass; includes adapter conversion, excludes JIT warm-up.\n'
               'Mode kept = output mass > 1% within radius 1.35 of each mode centre (±3). Unsorted systematic implementations lose a mode in some trials; Stone Soup sorts weights first.')
    save(fig,out,'01_mixture_distribution')


def fig_tradeoff(s,out):
    df=s.query('n==128 and dim==2')
    fig,axes=plt.subplots(1,4,figsize=(13,4.3),sharex=True,layout='constrained')
    for ax,cloud in zip(axes,['fixed','gaussian','mixture','skewed']):
        part=df[df.cloud==cloud]
        for fam in ['classical','ot','soft','moment','variational']:
            sub=part[part.method.map(family)==fam]
            ax.scatter(sub.forward_ms_mean,sub.sliced_w1_mean,s=40,marker=FAMILY[fam][2],c=[colour(m) for m in sub.method],
                       edgecolors=SURFACE,linewidths=1,zorder=3)
        for _,r in pd.concat([part[part.method=='eigen'],part.nsmallest(1,'sliced_w1_mean')]).iterrows():
            ax.annotate(label(r.provider,r.method),(r.forward_ms_mean,r.sliced_w1_mean),xytext=(6,4),textcoords='offset points',fontsize=7.5,color=INK2)
        ax.set_xscale('log');ax.xaxis.set_major_formatter(FuncFormatter(ms_format));ax.set_ylim(bottom=0);ax.set_title(f'{cloud.capitalize()} cloud')
    axes[0].set_ylabel('Sliced W1 (lower is better)');fig.supxlabel('Forward time per call, ms (log scale)',fontsize=8.5,color=INK2)
    family_legend(fig,df.method)
    header(fig,'Distribution error versus forward time','Each point is one provider/method · N = 128 · d = 2 · static weighted clouds')
    footer(fig,'Source: static_summary.csv. Lower-left is better. Timing from the serial pass; wrapper timings include conversions (Julia includes IPC) and are not isolated kernel speeds.')
    save(fig,out,'02_accuracy_vs_runtime')


def fig_static_scaling(s,out):
    df=s.query('dim==2')
    fig,axes=plt.subplots(1,4,figsize=(13,4.4),sharex=True,layout='constrained')
    for ax,cloud in zip(axes,['fixed','gaussian','mixture','skewed']):
        rep_lines(ax,df[df.cloud==cloud],'n','sliced_w1_mean','sliced_w1_ci95_half')
        n_axis(ax);ax.set_yscale('log');plain_log(ax.yaxis);ax.set_title(f'{cloud.capitalize()} cloud')
    axes[0].set_ylabel('Sliced W1 (log, lower is better)');fig.supxlabel('Particles N (log scale)',fontsize=8.5,color=INK2)
    top_legend(fig,*axes[0].get_legend_handles_labels(),ncol=7)
    header(fig,'Distribution error as the particle count grows','Representative method from each family · d = 2 · whiskers are 95% intervals over resampling seeds')
    footer(fig,'Source: static_summary.csv. Expensive OT/Gumbel cells above N = 128 use 3 repeats. Moment-matching and OT outputs move particles, so their error need not shrink with N\n'
               f'on non-Gaussian clouds. All {len(df.groupby(["provider","method"]))} provider/method series are in the CSV.')
    save(fig,out,'03_distribution_error_vs_n')


def fig_runtime(s,out):
    df=s.query("cloud=='gaussian' and dim==2")
    fig,ax=plt.subplots(figsize=(9.5,5.6),layout='constrained')
    for (p,m),part in sorted(df.groupby(['provider','method']),key=lambda kv:family(kv[0][1])!='classical',reverse=True):
        part=part.sort_values('n').dropna(subset=['forward_ms_mean'])
        if family(m)=='classical':ax.plot(part.n,part.forward_ms_mean,color=colour(m),lw=1,alpha=.7,zorder=2)
        else:ax.plot(part.n,part.forward_ms_mean,color=colour(m),marker=marker(m),ms=5.5,mec=SURFACE,mew=1,zorder=3)
        last=part.iloc[-1]
        if last.forward_ms_mean>5:
            inner=last.n<2048
            ax.annotate(f'{label(p,m)}  {ms_format(last.forward_ms_mean,0)}{"" if last.forward_ms_mean>=1000 else " ms"}',(last.n,last.forward_ms_mean),
                        xytext=(-6,9) if inner else (8,0),textcoords='offset points',ha='right' if inner else 'left',va='bottom' if inner else 'center',fontsize=8,color=INK2)
    n_axis(ax);ax.set_xlim(24,9000);ax.set_yscale('log');ax.yaxis.set_major_formatter(FuncFormatter(ms_format))
    ax.set(xlabel='Particles N (log scale)',ylabel='Forward time per call, ms (log scale)')
    ax.annotate('Classical, soft and moment-matching methods: about 0.2–3 ms at every N',(2048,0.25),xytext=(-6,-14),textcoords='offset points',
                ha='right',fontsize=8,color=INK2)
    family_legend(fig,df.method)
    header(fig,'CPU forward-time scaling','Gaussian cloud · d = 2 · serial timing pass, one fresh interpreter per method · compilation excluded')
    footer(fig,'Source: static_summary.csv (timing columns). Dense OT builds an N × N cost matrix; FilterFlow corrected OT stops at N = 128 because of its cubic correction solve.\n'
               'Wrapper timings include adapter conversion and seed handling; Julia includes Python-to-Julia IPC. CPU only.')
    save(fig,out,'04_forward_time_scaling')


def fig_memory(s,out):
    df=s.query("cloud=='gaussian' and dim==2").pivot_table(index=['provider','method'],columns='n',values='total_peak_rss_mb_bound_mean').reset_index()
    df['top']=df[[2048,128]].max(axis=1);df=df.sort_values('top',ascending=False).reset_index(drop=True)
    fig,ax=plt.subplots(figsize=(8.5,9),layout='constrained')
    for y,r in df.iterrows():
        c=colour(r.method)
        if np.isfinite(r[2048]):ax.plot([r[128],r[2048]],[y,y],color=c,lw=1.4,alpha=.45,zorder=2)
        ax.plot(r[128],y,ls='none',marker='o',ms=6,mfc=SURFACE,mec=c,mew=1.4,zorder=3)
        if np.isfinite(r[2048]):ax.plot(r[2048],y,ls='none',marker='o',ms=6.5,color=c,mec=SURFACE,mew=1,zorder=4)
    ax.set_ylim(len(df)-.5,-.5);ax.grid(axis='y',color=ROWGRID);row_labels(ax,df);ax.set_xlim(left=0)
    ax.set(xlabel='Whole-process peak RSS bound, MiB (lower is better)')
    extra=[Line2D([],[],ls='none',marker='o',ms=6,mfc=SURFACE,mec=INK2,mew=1.4,label='N = 128'),
           Line2D([],[],ls='none',marker='o',ms=6.5,color=INK2,mec=SURFACE,label='N = 2048')]
    family_legend(fig,df.method,ncol=3,extra=extra)
    header(fig,'Process peak memory','Gaussian cloud · d = 2 · serial timing pass · sorted by the larger of the two values')
    footer(fig,'Source: static_summary.csv (total_peak_rss_mb_bound). Includes framework imports and JIT caches, so it mostly reflects the provider stack,\n'
               'not per-call allocation. Julia adds its worker high-water mark (an upper bound). FilterFlow corrected OT has no N = 2048 cell.')
    save(fig,out,'05_peak_memory')


# ---------------------------------------------------------------- filtering

MODELS=[('linear','Linear Gaussian'),('ungm','UNGM'),('volatility','Stochastic volatility'),('bearing','Bearing-only tracking')]


def filter_rows(f):
    return ordered(f.query('n==128')[['provider','method']].drop_duplicates())


def fig_filter_metric(f,out,col,name,title,xlabel,pct=False,ref=None,note='',extra=''):
    rows=filter_rows(f)
    fig,axes=plt.subplots(1,4,figsize=(13,9.4),sharey=True,layout='constrained')
    for ax,(model,heading) in zip(axes,MODELS):
        part=rows.merge(f.query('n==128 and model==@model'),on=['provider','method'],how='left')
        dots(ax,part,col+'_mean',col+'_ci95_half',pct=pct);ax.set(title=heading,xlabel=xlabel)
        if ref is not None:ax.axvline(ref,color=INK2,lw=.9,zorder=1)
    row_labels(axes[0],rows);family_breaks(axes,rows);family_legend(fig,rows.method)
    header(fig,title,'N = 128 · 50 time steps · resample when ESS < N/2 · whiskers are 95% intervals across 10 observation sequences'+extra)
    footer(fig,'Source: filter_summary.csv. Fast methods: 10 sequences × 10 PF seeds; OT/Gumbel: 10 sequences × 1 seed. Means condition on successful trials\n'
               '(PFJAX MVN on UNGM: 7% numerical failures).'+note)
    save(fig,out,name)


def fig_filter_scaling(f,out):
    fig,axes=plt.subplots(2,2,figsize=(11,7.6),sharex=True,layout='constrained')
    for ax,(model,heading) in zip(axes.flat,MODELS):
        rep_lines(ax,f[f.model==model],'n','rmse_mean','rmse_ci95_half');n_axis(ax);ax.set(title=heading)
    for ax in axes[:,0]:ax.set_ylabel('RMSE vs latent truth')
    for ax in axes[1]:ax.set_xlabel('Particles N (log scale)')
    top_legend(fig,*axes[0,0].get_legend_handles_labels(),ncol=4)
    header(fig,'Filtering RMSE as the particle count grows','Representative method from each family · whiskers are 95% intervals across 10 observation sequences')
    footer(fig,'Source: filter_summary.csv. Expensive OT/Gumbel filters stop at N = 128. Bearing RMSE uses position only. Linear-model RMSE floor is set by the exact posterior,\n'
               'so differences there are small by construction.')
    save(fig,out,'08_filter_rmse_vs_n')


def fig_bearing(f,out):
    df=f.query("n==128 and model=='bearing'").sort_values(['recovered_mean','post_occlusion_rmse_mean'],ascending=[False,True]).reset_index(drop=True)
    fig,axes=plt.subplots(1,3,figsize=(12,9.2),sharey=True,layout='constrained')
    dots(axes[0],df,'recovered_mean','recovered_ci95_half',pct=True);axes[0].set(title='Runs that recover',xlabel='higher is better')
    dots(axes[1],df,'recovery_steps_mean','recovery_steps_ci95_half');axes[1].set(title='Steps to recover (capped at 20)',xlabel='lower is better')
    dots(axes[2],df,'post_occlusion_rmse_mean','post_occlusion_rmse_ci95_half');axes[2].set(title='Post-gap position error',xlabel='lower is better')
    row_labels(axes[0],df);family_legend(fig,df.method)
    header(fig,'Bearing-only tracking: recovery after a ten-step observation gap',
           'N = 128 · observations 20–29 hidden · rows sorted by recovery rate · whiskers are 95% intervals across 10 observation sequences')
    footer(fig,'Source: filter_summary.csv. Recovery = position error below 2 units for three consecutive observed steps; unrecovered runs count as 20 steps.\n'
               'Post-gap error is Euclidean position error.')
    save(fig,out,'09_bearing_recovery')


# ---------------------------------------------------------------- gradients and learning

def fig_local_gradients(g,out):
    df=g.query("cloud=='mixture' and n==128")
    wide=df.pivot_table(index=['provider','method'],columns='function',values='squared_error_mean').reset_index()
    order=wide.sort_values(['sin','x2','provider'])[['provider','method']]
    fig,axes=plt.subplots(1,2,figsize=(10,6.4),sharey=True,layout='constrained')
    for ax,(fn,heading) in zip(axes,[('sin','E[sin x]'),('x2','E[x²]')]):
        part=order.merge(df[df.function==fn],on=['provider','method'],how='left')
        dots(ax,part,'squared_error_mean','squared_error_ci95_half',log=True,floor=1e-6);ax.set(title=f'Gradient of {heading}',xlabel='gradient MSE vs exact (log, lower is better)')
    row_labels(axes[0],order);family_legend(fig,order.method)
    header(fig,'Local expectation gradients through one resampling step','Gaussian-mixture cloud · N = 128 · reference is the exact derivative of the weighted discrete distribution')
    footer(fig,'Source: gradient_summary.csv. Values below 1e-6 are drawn at the left edge with their value; eigen matches x² gradients by construction (moment matching).\n'
               'Classical resamplers without stop-gradient, and OT, return the same gradient for every seed here (zero variance), so their error is pure bias.')
    save(fig,out,'10_local_gradient_error')


def fig_likelihood_gradients(tn,cn,out):
    metrics=[('gradient_bias','Bias  ‖mean(g) − g*‖'),('gradient_variance','Variance  tr Cov(g)'),('gradient_mse','MSE  mean ‖g − g*‖²')]
    fig,axes=plt.subplots(2,3,figsize=(13,7.8),sharex=True,sharey='col',layout='constrained')
    tn=tn[tn.provider!='torchfilter']  # torchfilter soft is numerically identical to PyDPF soft
    for ax,(col,heading) in zip(axes[0],metrics):
        for (p,m),(c,mk) in GRAD.items():
            part=tn[(tn.provider==p)&(tn.method==m)].sort_values('n')
            ax.plot(part.n,part[col],color=c,marker=mk,ms=6,mec=SURFACE,mew=1,label=label(p,m))
        ax.set_title(f'PyTorch filter · {heading}')
    for ax,(col,heading) in zip(axes[1],metrics):
        for stop,(c,mk,name) in {False:(MUTED,'v','Cuthbert · systematic'),True:(DARK,'>','Cuthbert · systematic + stop-grad')}.items():
            part=cn[cn.stop_gradient==stop].sort_values('n')
            ax.plot(part.n,part[col],color=c,marker=mk,ms=6,mec=SURFACE,mew=1,label=name)
        ax.set_title(f'Cuthbert native (JAX) · {heading}')
    for ax in axes.flat:n_axis(ax);ax.set_yscale('log');plain_log(ax.yaxis)
    for ax in axes[1]:ax.set_xlabel('Particles N (log scale)')
    top_legend(fig,*axes[0,0].get_legend_handles_labels(),ncol=4)
    axes[1,0].legend(loc='lower left',fontsize=8)
    header(fig,'Particle-filter log-likelihood gradients against the exact Kalman gradient',
           'Cuthbert tutorial random-walk model and saved dataset (T = 50) · 100 seeds per N (Gumbel and PyDPF OT: 10, N ≤ 128) · log scales')
    footer(fig,'Source: torch_native_summary.csv, cuthbert_native_summary.csv. Gradient w.r.t. (log q, log r). PyDPF soft and torchfilter soft give identical values; only PyDPF is drawn.\n'
               'Resample threshold 0.7N, as in the Cuthbert tutorial. No confidence intervals are computed for these summaries.')
    save(fig,out,'11_likelihood_gradient_vs_n')


def fig_profiles(p,out):
    names={'logq':'log q (process noise)','logr':'log r (observation noise)'}
    fig,axes=plt.subplots(2,2,figsize=(11,7.4),sharex='col',layout='constrained')
    for j,(param,heading) in enumerate(names.items()):
        part=p[p.parameter==param]
        exact=part.drop_duplicates('value').sort_values('value')
        for i,(col,ref,ylabel) in enumerate([('loglik','exact_loglik','Log likelihood'),('gradient','exact_gradient','d log-likelihood / d parameter')]):
            ax=axes[i,j]
            ax.plot(exact.value,exact[ref],color=INK,lw=1.6,label='Kalman exact',zorder=2)
            for stop,(c,mk,name) in {False:(MUTED,'v','PF systematic'),True:(DARK,'>','PF systematic + stop-grad')}.items():
                sub=part[part.stop_gradient==stop].sort_values('value')
                if col=='loglik' and stop:continue  # the forward pass is identical with and without stop-gradient
                ax.plot(sub.value,sub[col],ls='none',marker=mk,ms=6,color=c,mec=SURFACE,mew=1,label=name if col=='gradient' or not stop else None,zorder=3)
            if col=='gradient':ax.axhline(0,color=AXIS,lw=.9,zorder=1)
            ax.set(title=f'{ylabel.split(" /")[0] if col=="loglik" else "Gradient"} · {heading}',ylabel=ylabel)
        axes[1,j].set_xlabel(f'{param} (21-point slice)')
    top_legend(fig,*axes[1,0].get_legend_handles_labels(),ncol=3)
    header(fig,'Cuthbert native likelihood and gradient profiles','One slice per parameter, other parameter fixed · N = 3000 particles · PF likelihood is identical with and without stop-gradient')
    footer(fig,'Source: results/standard/cuthbert_native/profiles.csv. Gradient zero crossings mark the slice maximiser.')
    save(fig,out,'12_cuthbert_gradient_profiles')


def fig_learning(l,out):
    df=l.sort_values(['score_rmse','provider']).reset_index(drop=True)
    fig,axes=plt.subplots(1,3,figsize=(11.5,4.8),sharey=True,layout='constrained')
    dots(axes[0],df,'parameter_absolute_error_mean','parameter_absolute_error_ci95_half');axes[0].set(title='|ŝ − 1|, true scale',xlabel='lower is better')
    dots(axes[1],df,'mle_reference_error_mean','mle_reference_error_ci95_half');axes[1].set(title='|ŝ − ŝ_ref|, grid-filter MLE',xlabel='lower is better')
    dots(axes[2],df,'score_rmse');axes[2].set(title='Likelihood-score RMSE',xlabel='lower is better')
    for ax in axes[:2]:ax.set_xlim(left=0)
    row_labels(axes[0],df);family_legend(fig,df.method)
    header(fig,'Stochastic-volatility observation-scale learning','N = 128 · 10 datasets · scale chosen on the grid {0.6, 0.8, 1.0, 1.2, 1.4} · rows sorted by score RMSE · whiskers are 95% intervals')
    footer(fig,'Source: learning_summary.csv. Coarse-grid estimates, not optimiser convergence. Score = central difference in log s (step 0.01) against a 1001-cell numerical\n'
               'grid-filter reference, which is itself approximate.')
    save(fig,out,'13_parameter_learning')


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',default='results/standard');a=p.parse_args()
    root=ROOT/a.input;report=root/'report';out=report/'figures';out.mkdir(exist_ok=True)
    style()
    s=pd.read_csv(report/'static_summary.csv');f=pd.read_csv(report/'filter_summary.csv')
    fig_mixture(s,out);fig_tradeoff(s,out);fig_static_scaling(s,out);fig_runtime(s,out);fig_memory(s,out)
    fig_filter_metric(f,out,'rmse','06_filter_rmse','Filtering RMSE across the four state-space models','RMSE vs latent truth (lower is better)',
                      note=' Bearing RMSE uses position only.')
    fig_filter_metric(f,out,'coverage95','07_filter_coverage','Filter calibration: 95% Gaussian-ellipsoid coverage','share of steps covered (closer to 95% is better)',
                      pct=True,ref=.95,extra='\nDark vertical line = nominal 95%. Each panel has its own x range; bearing coverage spans 0–100%.',note=' Coverage is a Gaussian-ellipsoid diagnostic, not exact credible-set coverage for multimodal posteriors.')
    fig_filter_scaling(f,out);fig_bearing(f,out)
    fig_local_gradients(pd.read_csv(report/'gradient_summary.csv'),out)
    fig_likelihood_gradients(pd.read_csv(report/'torch_native_summary.csv'),pd.read_csv(report/'cuthbert_native_summary.csv'),out)
    fig_profiles(pd.read_csv(root/'cuthbert_native/profiles.csv'),out)
    fig_learning(pd.read_csv(report/'learning_summary.csv'),out)


if __name__=='__main__':main()
