"""Create paper-style figures and their source tables from completed measurements.

Run with the project's Python environment from any directory. No benchmark is rerun.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, LogLocator, NullLocator
from summary_io import read_summary

ROOT=Path(__file__).resolve().parents[1]
DIST=ROOT/'website/dist'
OUT=DIST/'figures'
OUT.mkdir(exist_ok=True)
(DIST/'data').mkdir(exist_ok=True)
REPORT=ROOT/'results/standard/report'
PROVIDERS={'custom':'Local','stonesoup':'Stone Soup','particles':'particles','julia':'Julia','pfjax':'PFJAX','cuthbert':'Cuthbert','pydpf':'PyDPF','torchfilter':'Torchfilter','filterflow':'FilterFlow'}
METHODS={'systematic_stop':'systematic + stop-gradient','gaussian':'Gaussian draw','mvn':'Gaussian draw','corrected_ot':'corrected OT','gumbel':'Gumbel relaxation'}
STATIC={'n':128,'dim':2,'cloud':'mixture'}
CONFIGS=[
 dict(id='covariance',title='Eigen matches covariance to numerical precision',question='Does the eigen map preserve the promised spread?',table='static_summary',metric='covariance_error_mean',case=STATIC,axis='Covariance Frobenius error',log=True,unit='state units²',
 meaning='Covariance describes spread in each direction and how coordinates vary together. This error is the distance between the output covariance and the weighted input covariance. Zero means an exact match; lower is better.',formula='Square each entry of the covariance difference, add them, then take the square root.',
 benchmark='One fixed 2D Gaussian-mixture cloud; N=128. Reference: its weighted population covariance.',
 finding='The affine construction achieves its stated moment property on this full-rank cloud. The next figure checks a different property: probability shape.',
 limit='This is one full-rank cloud, not evidence that every distribution or nonlinear expectation is preserved.'),
 dict(id='shape',title='Matching covariance does not preserve mixture shape',question='Does the output put probability in the right places?',table='static_summary',metric='sliced_w1_mean',case=STATIC,axis='Sliced Wasserstein-1 distance',unit='state units',
 meaning='Sliced W1 measures how far probability mass must move to align the two clouds, averaged over 16 projected directions. Zero means every tested projection matches; lower is better.',formula='Project onto 16 fixed lines → integrate the gap between the two 1D CDFs → average the distances.',
 benchmark='The same fixed 2D mixture and N=128 as Figure 1. Reference: the weighted empirical input, not an ideal analytic mixture.',
 finding='Eigen preserves covariance but changes probability shape. In this case a classical copying method has a much smaller shape error.',
 limit='The comparison is conditional on this cloud. Systematic resampling depends on input order and implementation.'),
 dict(id='linear',title='Linear tracking errors are close across methods',question='How far is the estimated state from the simulated truth?',table='filter_summary',metric='rmse_mean',case={'n':128,'model':'linear'},axis='State tracking RMSE',unit='state units',
 meaning='RMSE is a typical tracking-error size. We square state errors across time, average them, and take a square root. Lower is better. This is error against the realized true state, not posterior error against Kalman.',formula='RMSE = √mean((estimated state − true state)²).',
 benchmark='Shared linear Gaussian model; N=128; 50 steps; 10 independent observation datasets.',
 finding='The smallest observed means are very close. Their broad, overlapping uncertainty intervals do not establish a clear winner. Tracking error must also be read with uncertainty coverage.',
 limit='These are successful-run means. Fast methods use 10 PF seeds per dataset; costly methods use one. No paired significance test is reported.'),
 dict(id='ungm',title='Nonlinear ambiguity needs a separate tracking test',question='Can the filter track a state whose sign is ambiguous?',table='filter_summary',metric='rmse_mean',case={'n':128,'model':'ungm'},axis='Nonlinear state tracking RMSE',unit='state units',
 meaning='This RMSE measures typical state error in the nonlinear growth model. Its squared observation can make positive and negative states look alike. Lower is better.',formula='RMSE = √mean((estimated state − true state)²).',
 benchmark='Shared UNGM; N=128; 50 steps; 10 independent datasets. Reference: simulated true states; there is no exact Kalman posterior for this task.',
 finding='A good moment match is not a guarantee of the smallest nonlinear tracking error. The point estimates favor a classical method here, but intervals overlap.',
 limit='PFJAX Gaussian-resampler failures remain visible in the full table. All accuracy means condition on successful trials.'),
 dict(id='score',title='Eigen has the lowest observed score MSE in this linear test',question='How close is the likelihood gradient to the exact gradient?',table='native_scores',metric='gradient_mse',case={'n':128},axis='Likelihood-gradient MSE',unit='squared score units',
 meaning='A score is the gradient of the log likelihood. Score MSE is the average squared distance from the exact Kalman gradient on the same data. It includes both systematic error and random variation; lower is better.',formula='Score MSE = mean(‖estimated gradient − exact Kalman gradient‖²).',
 benchmark='Native linear Gaussian score comparison; N=128. Cuthbert and PyTorch methods use the same saved observations and exact Kalman reference.',
 finding='Eigen provides the smallest recorded score MSE among the evaluated methods at this setting. This is direct evidence about the intended learning signal, not merely a finite-difference check.',
 limit='This summary has no confidence intervals. Most methods have 100 repeats; Gumbel and PyDPF OT have 10. One linear dataset does not establish nonlinear learning performance.'),
 dict(id='learning',title='Volatility scale fitting is a distinct learning test',question='Does the method recover the observation-scale parameter?',table='learning_summary',metric='parameter_absolute_error_mean',case={'n':128},axis='Mean absolute scale error',unit='scale units',
 meaning='Each run chooses an observation-scale value. Absolute scale error is the distance from the true scale, s=1. The plot averages that distance across datasets. Lower is better.',formula='Absolute scale error = |selected scale − 1|; then average over datasets.',
 benchmark='Shared stochastic-volatility learning; N=128; 10 datasets; scale grid {0.6,0.8,1,1.2,1.4}.',
 finding='The smallest point estimate belongs to soft resampling in Torchfilter. Overlapping intervals and the coarse scale grid limit claims of superiority.',
 limit='This is discrete grid selection, not continuous optimizer convergence. The separate grid-filter likelihood reference is approximate.'),
 dict(id='recovery',title='Recovery after missing bearings is often difficult',question='How often does tracking recover after the observation gap?',table='filter_summary',metric='recovered_mean',case={'n':128,'model':'bearing'},axis='Recovered trials',unit='fraction',direction='higher',percent=True,
 meaning='Recovery fraction is the share of trials that satisfy the declared recovery rule after observations return. Higher is better. A value of 32% means 32 recovered trials per 100 attempts when all trials complete.',formula='Recovery fraction = number of recovered trials ÷ number of successful trials.',
 benchmark='Shared bearing-only tracking; N=128; 50 steps; zero-based observations 20–29 omitted; 10 independent datasets.',
 finding='The best observed recovery fraction is still low. Most runs do not satisfy the recovery rule, and the intervals are wide. Recovery needs more than correct static moments.',
 limit='Recovery requires position distance below 2 units for three consecutive steps after the gap. Student-t intervals can extend outside [0,1]. This is not the native particles bearing model or Stone Soup EKF example.'),
 dict(id='runtime',title='An explicit map is not automatically the fastest',question='What does one forward resampling call cost on this CPU?',table='static_summary',metric='forward_ms_mean',case=STATIC,axis='Serial forward integration time (ms)',unit='ms',
 meaning='This is elapsed time for a forward resampling call, including the adapter work used to invoke each implementation. Lower is faster at these inputs.',formula='Measure elapsed milliseconds in the separate serial timing pass; average the timed calls.',
 benchmark='The fixed 2D mixture with N=128; CPU only. Timings are from the separate serial pass, not concurrent accuracy workers.',
 finding='Some classical implementations are faster than the eigen implementation at this setting. An explicit formula avoids a transport optimization loop but does not guarantee the lowest wall-clock time.',
 limit='Framework, conversions, seed handling, and adapter overhead are included; Julia includes IPC. This is not an isolated kernel or GPU speed ranking.')
]

def label(row):
    if row['provider']=='custom' and row['method']=='eigen':return 'Eigen (local)'
    return f"{PROVIDERS[row['provider']]} · {METHODS.get(row['method'],row['method'])}"

def readable(value,percent=False):
    if percent:return f'{value*100:.1f}%'
    return f'{value:.3g}'


def plot_figure(spec,shown,best,compact=False):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelsize':11,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.edgecolor':'#bcc7cf','text.color':'#253746','axes.labelcolor':'#425563','xtick.color':'#657787','ytick.color':'#253746','savefig.facecolor':'white'})
    fig,ax=plt.subplots(figsize=(5,6.4) if compact else (9.2,4.6),layout=None)
    fig.subplots_adjust(left=.43 if compact else .315,right=.82 if compact else .865,top=.82 if compact else .84,bottom=.23 if compact else .25)
    values=[r[spec['metric']] for r in shown]
    errors=[r.get(spec.get('ci_key','')) or 0 for r in shown]
    if spec.get('log'):
        ax.set_xscale('log');ax.set_xlim(min(values)*.12,max(1,max(v+e for v,e in zip(values,errors))*2))
        ax.xaxis.set_major_locator(LogLocator(base=10,numticks=5));ax.xaxis.set_minor_locator(NullLocator())
        if compact:ax.set_xticks([1e-15,1e-8,1])
    else:
        upper=max(v+e for v,e in zip(values,errors))*1.07
        lower=min(0,min(v-e for v,e in zip(values,errors)))
        ax.set_xlim(lower,upper if upper>0 else 1)
    if spec.get('percent'):ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis='x',color='#e6ebee',linewidth=.8);ax.set_axisbelow(True)
    winner_ids={(r['provider'],r['method']) for r in best}
    for i,row in enumerate(shown):
        value=row[spec['metric']];err=row.get(spec.get('ci_key','')) or 0
        eigen=row['provider']=='custom' and row['method']=='eigen'
        winner=(row['provider'],row['method']) in winner_ids
        color='#147d70' if eigen else '#305c94' if winner else '#9ba9b5'
        ax.errorbar(value,i,xerr=err,fmt='o',color=color,ecolor=color,markersize=7.5 if eigen or winner else 6,capsize=3,elinewidth=1.5,zorder=3)
        ax.text(1.025,i,readable(value,spec.get('percent')),transform=ax.get_yaxis_transform(),va='center',fontsize=11,color=color if eigen or winner else '#627789',fontweight='bold' if eigen or winner else 'normal',clip_on=False)
    plot_labels=[label(r).replace(' + ','\n+ ') if len(label(r))>29 else label(r) for r in shown]
    if compact:plot_labels=[textwrap.fill(label(r).replace(' · ','\n'),width=20,replace_whitespace=False) for r in shown]
    ax.set_yticks(range(len(shown)),plot_labels,fontsize=10 if compact else 11);ax.tick_params(axis='y',length=0,pad=12)
    ax.set_ylim(len(shown)-.45,-.6)
    axis_label=spec['axis']+(' · log scale' if spec.get('log') else '')
    ax.set_xlabel(textwrap.fill(axis_label,27) if compact else axis_label,labelpad=12,fontsize=10 if compact else 11)
    fig.text(.045,.955,f"FIGURE {spec['number']:02d}  /  {'HIGHER IS BETTER' if spec.get('direction')=='higher' else 'LOWER IS BETTER'}",fontsize=8 if compact else 10,color='#627787',weight='bold')
    fig.text(.045,.895,spec['axis'],fontsize=12 if compact else 15,weight='bold')
    fig.text(.045,.04 if compact else .045,'Teal: eigen baseline\nBlue: best observed comparator' if compact else 'Teal: eigen baseline    Blue: best observed comparator    Grey: other displayed methods',fontsize=9,color='#657787')
    for ext in (['svg','png'] if compact else ['svg','pdf','png']):
        suffix='-mobile' if compact else ''
        fig.savefig(OUT/f"{spec['id']}{suffix}.{ext}",dpi=180,metadata={'Creator':'Resampler Lab'} if ext=='pdf' else None)
    plt.close(fig)


def generate():
    tables={name:read_summary(REPORT/f'{name}.csv') for name in ['static_summary','filter_summary','learning_summary','torch_native_summary','cuthbert_native_summary']}
    tables['native_scores']=tables['torch_native_summary']+[{**r,'provider':'cuthbert','method':'systematic_stop' if r['stop_gradient'] else 'systematic'} for r in tables['cuthbert_native_summary']]
    figures=[]
    for number,config in enumerate(CONFIGS,1):
        spec=dict(config,number=number)
        if spec['metric'].endswith('_mean'):spec['ci_key']=spec['metric'][:-5]+'_ci95_half'
        rows=[dict(r) for r in tables[spec['table']] if all(r.get(k)==v for k,v in spec['case'].items())]
        valid=[r for r in rows if isinstance(r.get(spec['metric']),(int,float)) and math.isfinite(r[spec['metric']])]
        valid.sort(key=lambda r:r[spec['metric']],reverse=spec.get('direction')=='higher')
        best=[r for r in valid if math.isclose(r[spec['metric']],valid[0][spec['metric']],rel_tol=1e-12,abs_tol=0)]
        shown=valid[:5]
        eigen=next(r for r in valid if r['provider']=='custom' and r['method']=='eigen')
        if eigen not in shown:shown.append(eigen)
        for row in rows:
            row['display_name']=label(row)
            row['rank']=next((i+1 for i,r in enumerate(valid) if r['provider']==row['provider'] and r['method']==row['method']),None)
        shown=[next(r for r in rows if r['provider']==s['provider'] and r['method']==s['method']) for s in shown]
        rows.sort(key=lambda r:r['rank'] if r['rank'] is not None else float('inf'))
        sources=['torch_native_summary','cuthbert_native_summary'] if spec['table']=='native_scores' else [spec['table']]
        spec.update(total_methods=len(rows),valid_methods=len(valid),shown=shown,rows=rows,best_names=[label(r) for r in best],best_value=valid[0][spec['metric']],eigen_value=eigen[spec['metric']],eigen_rank=next(r['rank'] for r in rows if r['provider']=='custom' and r['method']=='eigen'),sources=[dict(path=f'results/standard/report/{name}.csv',sha256=hashlib.sha256((REPORT/f'{name}.csv').read_bytes()).hexdigest()) for name in sources],image=f'figures/{spec["id"]}.svg',mobile_image=f'figures/{spec["id"]}-mobile.svg',pdf=f'figures/{spec["id"]}.pdf',png=f'figures/{spec["id"]}.png',csv=f'figures/{spec["id"]}.csv')
        with (OUT/f"{spec['id']}.csv").open('w',newline='') as stream:
            fields=list(dict.fromkeys(key for row in rows for key in row))
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        plot_figure(spec,shown,best)
        plot_figure(spec,shown,best,compact=True)
        figures.append(spec)
    payload=json.dumps(figures,ensure_ascii=False,allow_nan=False,indent=2)+'\n'
    (DIST/'data/paper-figures.json').write_text(payload)
    (DIST/'data/paper-figures.js').write_text('export const paperFigures = '+payload.rstrip()+';\n')
    print(json.dumps([dict(figure=s['number'],id=s['id'],best=s['best_names'],value=s['best_value'],eigen=s['eigen_value'],eigen_rank=s['eigen_rank'],methods=s['total_methods']) for s in figures],indent=2))

if __name__=='__main__':generate()
