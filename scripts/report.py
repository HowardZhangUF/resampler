"""Summaries, plots, audit, and a self-contained interactive HTML report."""
import argparse
import html
import re
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.express as px

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from resampler_bench.adapters import METHODS


def read_trials(root,name):
    frames=[]
    for provider in METHODS:
        p=root/provider/(name+'.csv')
        if p.exists():frames.append(pd.read_csv(p))
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()


def summary(df,keys,metrics,cluster=None):
    records=[]
    for group,part in df.groupby(keys,dropna=False):
        record=dict(zip(keys,group if isinstance(group,tuple) else (group,)))
        record['trials']=len(part)
        record['successful_trials']=int(part[metrics[0]].notna().sum())
        record['failure_rate']=1-record['successful_trials']/len(part)
        for metric in metrics:
            values=part.groupby(cluster)[metric].mean().dropna() if cluster else part[metric].dropna()
            n=len(values);mean=float(values.mean()) if n else np.nan
            se=float(values.std(ddof=1)/np.sqrt(n)) if n>1 else np.nan
            half=float(t.ppf(.975,n-1)*se) if n>1 else np.nan
            record[metric+'_mean']=mean;record[metric+'_ci95_half']=half
            record[metric+'_ci_samples']=n
        records.append(record)
    return pd.DataFrame(records)


RULES={'lower':'lower is better','higher':'higher is better','zero':'closest to 0 is better','nominal':'closest to 95% is better'}


def fmt(v):return f'{v:.4g}' if np.isfinite(v) else 'n/a'


LATEX=[]  # every metric_table is also rendered as a LaTeX table after the report is built


def ranked(df,metric,rule):
    """Rows best first. Equal values share a rank (1, 2, 3, 3, 5); rounding absorbs float noise between deterministic methods."""
    df=df.reset_index(drop=True)
    value=df[metric+'_mean'] if metric+'_mean' in df else df[metric]
    half=df[metric+'_ci95_half'] if metric+'_ci95_half' in df else pd.Series(np.nan,index=df.index)
    score={'lower':value,'higher':-value,'zero':value.abs(),'nominal':(value-.95).abs()}[rule]
    best=score.min();ranks=score.round(12).rank(method='min');rows=[]
    for i in score.sort_values(na_position='last',kind='stable').index:
        r=df.loc[i]
        rows.append(dict(rank=int(ranks[i]) if np.isfinite(ranks[i]) else None,provider=r.provider,method=r.method,
                         trials=f'{int(r.successful_trials)}/{int(r.trials)}' if 'successful_trials' in df else (str(int(r.trials)) if 'trials' in df else ''),
                         value=value[i],half=half[i] if not np.isfinite(half[i]) or half[i]>1e-12*abs(value[i]) else 0.,
                         best=bool(np.isfinite(score[i]) and np.isclose(score[i],best,rtol=1e-9,atol=1e-15))))
    return rows


def metric_table(title,df,metric,rule='lower',label=None):
    """One task and one metric per table. Rows run best first; the best method (and any exact tie) is bold."""
    name=label or metric.replace('_',' ');rows=ranked(df,metric,rule)
    LATEX.append(dict(title=title,metric=metric,name=name,rule=rule,rows=rows))
    lines=[f'### {title} · {name}','',f'Best = {RULES[rule]}. Best method in bold; ± is the 95% interval half-width where available.','',
           f'| rank | provider | method | trials | {name.replace("|",chr(92)+"|")} |','| ---: | --- | --- | ---: | ---: |']
    for r in rows:
        cell=fmt(r['value'])+(f' ± {fmt(r["half"])}' if np.isfinite(r['half']) else '');method=r['method']
        if r['best']:cell,method=f'**{cell}**',f'**{method}**'
        lines.append(f'| {r["rank"] or "–"} | {r["provider"]} | {method} | {r["trials"]} | {cell} |')
    return '\n'.join(lines)


TEX_NAMES={'RFF MMD²':'RFF MMD$^2$','|ŝ − 1|':r'$|\hat{s}-1|$','|ŝ − ŝ_ref|':r'$|\hat{s}-\hat{s}_{\mathrm{ref}}|$'}


def tex_text(text):
    for a,b in [('&',r'\&'),('%',r'\%'),('#',r'\#'),('_',r'\_')]:text=text.replace(a,b)
    return re.sub(r'\b([Nd])=(\d+)',r'$\1=\2$',text)


def tex_num(v):
    if not np.isfinite(v):return r'\text{--}'
    mantissa,_,exponent=f'{v:.4g}'.partition('e')
    return mantissa+(rf'\times10^{{{int(exponent)}}}' if exponent else '')


def latex_table(t):
    name=TEX_NAMES.get(t['name'],tex_text(t['name']))
    slug=re.sub('[^a-z0-9]+','-',re.sub(r'\(.*?\)','',t['title']).lower()+' '+t['metric']).strip('-')
    caption=(f"{tex_text(t['title'])}: {name}. Best = {tex_text(RULES[t['rule']])}. Best method in bold; "
             r"$\pm$ gives the 95\% confidence-interval half-width where available.")
    lines=[r'\begin{table}[!htbp]',r'\centering',r'\small',rf'\caption{{{caption}}}',rf'\label{{tab:{slug}}}',
           r'\begin{tabular}{rllrr}',r'\toprule',rf'Rank & Provider & Method & Trials & {name} \\',r'\midrule']
    for r in t['rows']:
        value=tex_num(r['value'])+(rf' \pm {tex_num(r["half"])}' if np.isfinite(r['half']) else '')
        value=rf'$\mathbf{{{value}}}$' if r['best'] else f'${value}$'
        method=tex_text(r['method']);method=rf'\textbf{{{method}}}' if r['best'] else method
        lines.append(f'{r["rank"] or "--"} & {tex_text(r["provider"])} & {method} & {r["trials"]} & {value} \\\\')
    lines+=[r'\bottomrule',r'\end{tabular}',r'\end{table}']
    return slug,'\n'.join(lines)+'\n'


def write_latex(out):
    """One .tex file per table, all tables in one file, and a standalone document that compiles them."""
    folder=out/'latex';tables=folder/'tables';tables.mkdir(parents=True,exist_ok=True)
    for old in tables.glob('*.tex'):old.unlink()
    combined=['% Generated by scripts/report.py; requires \\usepackage{booktabs}. One task and one metric per table.']
    previous=None
    for t in LATEX:
        slug,body=latex_table(t);(tables/f'{slug}.tex').write_text(body)
        if t['title']!=previous:combined+=['',f'% ---- {t["title"]}'];previous=t['title']
        combined+=['',body.rstrip()]
    (folder/'all_tables.tex').write_text('\n'.join(combined)+'\n')
    (folder/'benchmark_tables.tex').write_text('\n'.join([r'\documentclass{article}',r'\usepackage[margin=2cm]{geometry}',r'\usepackage{booktabs}',r'\usepackage[skip=6pt]{caption}',
        r'\usepackage{amsmath}',r'\begin{document}',r'\input{all_tables}',r'\end{document}'])+'\n')
    return len(LATEX)


def plot_static(summary,out):
    df=summary[(summary.n==128)&(summary.dim==2)].copy()
    df['label']=df.provider+'/'+df.method
    fig,ax=plt.subplots(figsize=(10,7))
    for p,group in df[df.cloud=='mixture'].groupby('provider'):
        ax.scatter(group.forward_ms_mean,group.sliced_w1_mean,label=p,alpha=.8)
        for _,r in group[group.provider=='custom'].iterrows():ax.annotate(r.method,(r.forward_ms_mean,r.sliced_w1_mean),fontsize=9)
    ax.set(xscale='log',xlabel='Adapter forward time (ms, CPU)',ylabel='Sliced W1',title='Mixture: distribution error versus time (N=128, D=2)')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'distribution_vs_runtime.png',dpi=160);plt.close(fig)
    interactive=px.scatter(df,x='forward_ms_mean',y='sliced_w1_mean',color='provider',symbol='method',facet_col='cloud',hover_name='label',
                           hover_data=['trials','covariance_error_mean','eigenvalue_error_mean','rff_mmd2_mean'],log_x=True,
                           title='Distribution preservation versus CPU adapter time · N=128, D=2')
    return interactive


def native_summary(df,keys):
    records=[]
    for group,part in df.groupby(keys):
        row=dict(zip(keys,group if isinstance(group,tuple) else (group,)))
        g=part[['grad_logq','grad_logr']].to_numpy();exact=part[['exact_grad_logq','exact_grad_logr']].iloc[0].to_numpy()
        row.update(trials=len(part),gradient_bias=np.linalg.norm(g.mean(0)-exact),gradient_variance=np.var(g,axis=0,ddof=1).sum(),
                   gradient_mse=np.mean(np.sum((g-exact)**2,axis=1)),loglik_variance=part.loglik.var(),
                   loglik_bias=part.loglik.mean()-part.exact_loglik.iloc[0],forward_ms=part.forward_ms.mean())
        if 'backward_ms' in part:row['backward_ms']=part.backward_ms.mean()
        else:row['forward_backward_ms']=part.forward_backward_ms.mean()
        records.append(row)
    return pd.DataFrame(records)


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',default='results/standard');a=p.parse_args()
    root=ROOT/a.input;out=root/'report';out.mkdir(parents=True,exist_ok=True)
    s,f,g,l=[read_trials(root,name) for name in ['static','filter','gradient','learning']]
    audit=[];skips=[]
    for provider in METHODS:
        status=root/provider/'status.jsonl'
        if status.exists():audit.extend(json.loads(line) for line in status.read_text().splitlines())
        path=root/provider/'skips.jsonl'
        if path.exists():skips.extend(json.loads(line) for line in path.read_text().splitlines())
    # A retried cell supersedes its previous execution status.
    audit=list({(r['provider'],r['method'],r['stage'],r['case']):r for r in audit}.values())
    completed=[r for r in audit if r['status']=='complete'];failed=[r for r in audit if r['status']=='failed']
    numerical=[]
    for provider in METHODS:
        path=root/provider/'numerical_failures.jsonl'
        if path.exists():numerical.extend(json.loads(line) for line in path.read_text().splitlines())
    (out/'audit.json').write_text(json.dumps(dict(completed_cells=len(completed),failures=failed,numerical_trial_failures=numerical,skips=skips),indent=2))
    ss=summary(s,['provider','method','cloud','n','dim'],['sliced_w1','rff_mmd2','mean_error','covariance_error','eigenvalue_error','estimator_mse_x','estimator_mse_x2','estimator_mse_sin','mode_coverage','mode_mass_error','forward_ms','process_peak_rss_mb'])
    timing=read_trials(ROOT/'results/performance','timing')
    serial_timing=not timing.empty
    if serial_timing:
        if 'worker_peak_rss_mb' not in timing:timing['worker_peak_rss_mb']=0.
        timing['worker_peak_rss_mb']=timing.worker_peak_rss_mb.fillna(0.)
        timing['total_peak_rss_mb_bound']=timing.process_peak_rss_mb+timing.worker_peak_rss_mb
        ts=summary(timing,['provider','method','cloud','n','dim'],['forward_ms','process_peak_rss_mb','worker_peak_rss_mb','total_peak_rss_mb_bound'])
        ts=ts.drop(columns=['successful_trials','failure_rate']).rename(columns={'trials':'timing_trials'})
        ss=ss.drop(columns=['forward_ms_mean','forward_ms_ci95_half','forward_ms_ci_samples','process_peak_rss_mb_mean','process_peak_rss_mb_ci95_half','process_peak_rss_mb_ci_samples']).merge(ts,on=['provider','method','cloud','n','dim'],how='left')
    ss.to_csv(out/'static_summary.csv',index=False)
    fs=summary(f,['provider','method','model','n'],['rmse','coverage95','predictive_nll','posterior_mean_rmse','posterior_covariance_error','loglik_error','mean_ess_fraction','filter_ms','recovery_steps','recovered','post_occlusion_rmse'],cluster='data_seed')
    # Likelihood variance is conditional on a fixed observation sequence.
    lv=f.groupby(['provider','method','model','n','data_seed']).loglik.var().groupby(['provider','method','model','n']).mean().rename('conditional_loglik_variance').reset_index()
    fs=fs.merge(lv,on=['provider','method','model','n']);fs.to_csv(out/'filter_summary.csv',index=False)
    gs=pd.DataFrame()
    if len(g):
        gs=summary(g,['provider','method','cloud','function','n'],['gradient','error','squared_error','forward_ms','backward_ms'])
        variance=g.groupby(['provider','method','cloud','function','n']).gradient.var().rename('gradient_variance').reset_index()
        gs=gs.merge(variance,on=['provider','method','cloud','function','n']);gs['gradient_bias']=gs.error_mean.abs();gs.to_csv(out/'gradient_summary.csv',index=False)
        serial_g=read_trials(ROOT/'results/performance','gradient')
        if len(serial_g):
            timings=summary(serial_g,['provider','method','cloud','function','n'],['forward_ms','backward_ms']).drop(columns=['trials','successful_trials','failure_rate'])
            gs=gs.drop(columns=['forward_ms_mean','forward_ms_ci95_half','forward_ms_ci_samples','backward_ms_mean','backward_ms_ci95_half','backward_ms_ci_samples']).merge(timings,on=['provider','method','cloud','function','n'],how='left')
            gs.to_csv(out/'gradient_summary.csv',index=False)
    ls=pd.DataFrame()
    if len(l):
        l['score_squared_error']=l.score_error**2
        ls=summary(l,['provider','method','n'],['parameter_absolute_error','mle_reference_error','score_error','score_squared_error','learning_ms'])
        ls['score_rmse']=np.sqrt(ls.score_squared_error_mean)
        ls.to_csv(out/'learning_summary.csv',index=False)
    figures=[plot_static(ss,out)]
    ff=fs.copy();ff['label']=ff.provider+'/'+ff.method
    figures.append(px.line(ff,x='n',y='rmse_mean',color='label',facet_col='model',markers=True,log_x=True,log_y=True,
                           error_y='rmse_ci95_half',title='Filtering error versus particles · 95% intervals across observation sequences'))
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    selected=['particles/systematic','variational/kl','custom/eigen','custom/gaussian','custom/gumbel','pfjax/ot','pydpf/ot','filterflow/corrected_ot']
    for ax,model in zip(axes.flat,['linear','ungm','volatility','bearing']):
        for label in selected:
            df=ff[(ff.model==model)&(ff.label==label)].sort_values('n')
            if len(df):ax.errorbar(df.n,df.rmse_mean,yerr=df.rmse_ci95_half,marker='.',label=label,capsize=2)
        ax.set(xscale='log',yscale='log',title=model,xlabel='Particles',ylabel='RMSE')
    axes[0,0].legend(fontsize=6);fig.tight_layout();fig.savefig(out/'filter_rmse.png',dpi=160);plt.close(fig)
    runtime=ss[(ss.cloud=='gaussian')&(ss.dim==2)].copy();runtime['label']=runtime.provider+'/'+runtime.method
    figures.append(px.line(runtime,x='n',y='forward_ms_mean',color='label',markers=True,log_x=True,log_y=True,title='CPU forward scaling · Gaussian cloud, D=2 · compilation excluded'))
    if serial_timing:
        figures.append(px.line(runtime,x='n',y='total_peak_rss_mb_bound_mean',color='label',markers=True,log_x=True,
                               hover_data=['process_peak_rss_mb_mean','worker_peak_rss_mb_mean'],
                               title='Whole-process peak RSS bound (MiB) · imports and caches included · Gaussian, D=2'))
    native_tables=[]
    for folder,keys in [('cuthbert_native',['n','stop_gradient']),('torch_native',['provider','method','n'])]:
        path=root/folder/'trials.csv'
        if path.exists():
            ns=native_summary(pd.read_csv(path),keys);ns.to_csv(out/(folder+'_summary.csv'),index=False)
            native_tables.append((folder,ns))
            plot=ns.copy();plot['label']=plot['stop_gradient'].map({False:'Cuthbert standard',True:'Cuthbert stop gradient'}) if folder=='cuthbert_native' else plot.provider+'/'+plot.method
            figures.append(px.line(plot,x='n',y='gradient_bias',color='label',markers=True,log_x=True,log_y=True,
                                   hover_data=['trials','gradient_variance','gradient_mse','loglik_bias'],title=folder+': PF likelihood-gradient bias against exact Kalman reference'))
    if len(gs):
        grad=gs[(gs.n==128)&(gs.cloud=='mixture')].copy();grad['label']=grad.provider+'/'+grad.method
        figures.append(px.scatter(grad,x='forward_ms_mean',y='squared_error_mean',color='provider',symbol='method',facet_col='function',log_x=True,log_y=True,hover_name='label',hover_data=['gradient_bias','gradient_variance','trials'],title='Local expectation gradients · mixture, N=128 · exact discrete-distribution reference'))
    if len(ls):
        plot=ls.copy();plot['label']=plot.provider+'/'+plot.method
        figures.append(px.bar(plot,x='label',y='parameter_absolute_error_mean',error_y='parameter_absolute_error_ci95_half',title='Stochastic-volatility observation-scale estimation · coarse-grid MLE, 10 datasets'))
    profile_path=root/'cuthbert_native/profiles.csv'
    if profile_path.exists():
        df=pd.read_csv(profile_path)
        df['method']=df.stop_gradient.map({False:'PF standard',True:'PF stop gradient'})
        exact=df.drop_duplicates(['parameter','value']).copy();exact['method']='Kalman exact';exact['gradient']=exact.exact_gradient
        profiles=pd.concat([df,exact],ignore_index=True)
        figures.append(px.line(profiles,x='value',y='gradient',color='method',facet_col='parameter',hover_data=['exact_gradient','loglik','exact_loglik'],title='Native Cuthbert profile gradient slices · N=3000'))
    intro=f'<h1>Particle resampler benchmark results</h1><p>{len(s):,} static trials; {len(f):,} filter trials; {len(g):,} local gradient trials; {len(completed)} completed cells; {len(failed)} failures.</p>'
    caveats=f'<p>CPU results. Separate serial timing pass available: {serial_timing}. Forward times include provider adapters; Julia includes IPC. FilterFlow uses float32 inputs; other adapters use float64. Expensive methods use fewer repeats and a smaller filtering grid. RSS includes framework imports and caches; Julia worker peak RSS is recorded separately. See REPORT.md for methods and limits.</p>'
    content=['<!doctype html><html><head><meta charset="utf-8"><title>Resampler benchmarks</title><style>body{font-family:system-ui;margin:32px;background:#fafbfc;color:#18212b}p{max-width:1100px}a{color:#1261aa}</style></head><body>',intro,caveats]
    for i,fig in enumerate(figures):
        fig.update_layout(height=520,template='plotly_white')
        content.append(fig.to_html(full_html=False,include_plotlyjs=True if i==0 else False))
    content.append('</body></html>');(out/'report.html').write_text('\n'.join(content))
    chosen=ss[(ss.n==128)&(ss.dim==2)&(ss.cloud=='mixture')]
    MODELS={'linear':'Linear Gaussian','ungm':'UNGM','volatility':'Stochastic volatility','bearing':'Bearing-only tracking'}
    eigen=ss[ss.method=='eigen']
    vr_rows=pd.DataFrame(dict(provider=['variational','variational','particles','particles','particles'],method=['kl','tv','systematic','ssp','multinomial']))
    vr_static=vr_rows.merge(ss[(ss.n==128)&(ss.dim==2)&(ss.cloud=='mixture')],on=['provider','method'])
    vr_filter=vr_rows.merge(fs[fs.n==128],on=['provider','method'])
    sources=json.loads((ROOT/'results/sources.json').read_text())
    lines=['# Particle-resampler benchmark report','',
           f'Completed {len(completed)} benchmark cells across {len(METHODS)-1} upstream repositories plus custom baselines. Raw trials: {len(s):,} static, {len(f):,} filtering, {len(g):,} local gradients, {len(l):,} parameter-learning trials.',
           '',f'Execution failures: **{len(failed)}**. Recorded numerical trial failures: **{len(numerical)}**. Explicit grid exclusions: **{len(skips)}**. See `audit.json` for the full list. Numerical failures are benchmark outcomes, not silently repaired results. Metric means condition on successful trials; compare them together with the failure rate.',
           '', '## Scope and method','',
           'This is a common benchmark suite that calls the checked-out repositories. It is not a claim to reproduce every experiment or headline number in their papers. The Cuthbert likelihood-gradient experiment additionally runs the native tutorial functions and native Kalman reference.',
           '', 'All six requested families are covered: fixed weighted clouds; Gaussian and Gaussian-mixture clouds; a linear Gaussian state-space model; UNGM; stochastic volatility; and bearing-only target tracking with a ten-step observation gap.',
           '', 'Shared models: linear x_t=0.8x_(t−1)+0.5ε, y_t=x_t+0.8η, initial variance 1; UNGM x_t=0.5x+25x/(1+x²)+8cos(1.2t)+sqrt(10)ε, y_t=x_t²/20+η, initial variance 5; volatility x_t=0.95x+0.2ε, y_t=exp(x_t/2)η, stationary initial state. Tracking has [px,vx,py,vy] constant-velocity dynamics, acceleration SD 0.03, and atan2(py,px) observations with noise SD 0.05 radians. Initial tracking mean is [10,0.35,−6,0.25] with diagonal covariance [4,0.04,4,0.04]. All noises are independent standard normals before scaling. Static input measures, not ideal analytic distributions, are the distribution-metric references.',
           '', 'The standard grid uses N = 32, 128, 512, 2048 and static dimensions 1, 2, 4. Fast methods use 100 resampling/filter repeats. Expensive OT/Gumbel methods use 10 repeats at N ≤ 128 and 3 static scaling repeats above that. Expensive full-filter and gradient runs stop at N=128; covariance-corrected OT also stops at N=128. Other local gradients stop at N=512. Full filters have 50 time steps and resample when ESS < 0.5N. Native likelihood gradients use the tutorial threshold 0.7N.',
           '', 'Dense PyDPF and FilterFlow OT use the Gaussian d=2 case for scaling at N=2048, while retaining the complete static-distribution grid through N=512. PyDPF took about 25 seconds per N=2048 call in initial accuracy measurements. Extra large-N accuracy cells completed before this limit was introduced are retained; serial timing for those extra cells is unavailable. They are not used in the N=2048 common scaling comparison.',
           '', 'Static trials share one fixed weighted cloud per grid cell and vary only the resampling seed. Filtering uses ten observation sequences, with ten particle-filter random seeds per sequence for fast methods. Expensive filters use one seed per sequence. Static confidence intervals condition on the supplied cloud. Filter confidence intervals use successful sequence-level means (up to ten independent clusters), not 100 falsely independent observations. Each metric records its actual interval sample count in the summary CSV. Intervals are Student-t intervals for the mean.',
           '', f'Accuracy workers run concurrently. Separate serial timing pass used for forward/local-gradient timing and process RSS summaries: **{serial_timing}**. Raw accuracy-run times are retained as diagnostics; they should not be used for controlled cross-framework speed claims.',
           '', '## Eigen baseline','',
           'Let μw and Cw be the weighted population mean and covariance. Let Z = X − mean(X), and C0 = ZᵀZ/N. The baseline returns Y = μw + Z C0^(-1/2) Cw^(1/2), with symmetric matrix roots computed by eigendecomposition. When C0 has full rank and no eigenvalue floor is active, mean(Y)=μw and Cov(Y)=Cw. Complexity is O(Nd²+d³), and the algorithm does not construct an N×N transport matrix.',
           '', 'The eigenvalue floor is 1e-10. Rank-deficient inputs remain finite but need not have exact covariance matching. The final implementation uses analytic divided differences for the root functions\u2019 first-order Fréchet derivatives, avoiding unstable eigenvector derivatives at repeated positive eigenvalues. Small covariance eigenvalues can still cause intrinsic ill-conditioning; second-order derivatives are explicitly unsupported. This is an affine moment-matching baseline, not an established distribution-preserving resampler or a new-method claim. It can distort modes and nonlinear expectations. Matching the x² gradient is an algebraic consequence of matching moments; the sin gradient and full PF likelihood gradient are separate tests.',
           '',f'Largest observed eigen-baseline mean error: {eigen.mean_error_mean.max():.3g}; covariance Frobenius error: {eigen.covariance_error_mean.max():.3g}.',
           '', 'The root derivative follows the matrix-function Sylvester-equation relation described in [Higham\u2019s derivation](https://nhigham.com/2020/09/01/what-is-the-sylvester-equation/). The implementation avoids the direct eigenvector-gradient limitation documented by [PyTorch](https://docs.pytorch.org/docs/2.14/generated/torch.linalg.eigh.html).',
           '', '## Variational resampling','',
           'The `variational` provider calls `kl` and `tv` from [okviman/Variational-Resampling](https://github.com/okviman/Variational-Resampling) (Kviman et al., *Variational Resampling*, AISTATS 2024). `kl` greedily assigns offspring counts B_s with a max-heap so that Σ B_s(log w_s − log B_s), the resampling ELBO up to constants, is maximal; `tv` rounds N·w_s and gives the remainders to the largest fractional parts. Both are deterministic, so repeated calls on one cloud return identical offspring (zero offspring variance). `tv` keeps |B_s − N·w_s| < 1; `kl` can exceed 1 (largest observed deviation 1.09 in 400 random weight vectors), because it optimizes the ELBO rather than the count error.',
           '', 'The adapter passes normalized log importance weights, which is the upstream `kl-iw` setting. Upstream `main.py` also runs `kl` on the path log joint log p(x_1:t, y_1:t); that input needs model densities and ancestry, which a generic resampler interface does not receive, so that variant is executed only by `scripts/run_variational_upstream.py`. Both methods return indices and have no gradient path, so they have no local-gradient cells. Upstream `kl` recomputes Σ B_s at every heap step, so its cost grows as O(N²).',
           '', 'The tables below compare `kl` and `tv` with three classical resamplers from `particles` on the mixture cloud (N=128, d=2) and in the four filters (N=128). Every provider/method appears in the full tables later in this report.',
           '', metric_table('Variational vs classical · mixture cloud (N=128, d=2)',vr_static,'sliced_w1',label='sliced W1'),
           '', metric_table('Variational vs classical · mixture cloud (N=128, d=2)',vr_static,'mode_coverage','higher',label='trials keeping both modes'),
           '', metric_table('Variational vs classical · mixture cloud (N=128, d=2)',vr_static,'forward_ms',label='forward time (ms)'),
           '', 'On this mixture cloud every N·w_s is 0.7 or 1.3, so `kl` and `tv` both give each particle exactly one offspring: the output is the input positions with uniform weights, and the 35% mode receives 50% of the mass on every call. Random schemes are unbiased in expectation; a deterministic allocation keeps its rounding error on every call, which is why the static errors here are higher than SSP or multinomial. Offspring-count variance is exactly zero for both methods.',
           *[block for model in MODELS for block in ['',metric_table(f'Variational vs classical · {MODELS[model]} filter (N=128)',vr_filter[vr_filter.model==model],'rmse',label='RMSE')]],
           *[block for model in MODELS for block in ['',metric_table(f'Variational vs classical · {MODELS[model]} filter (N=128)',vr_filter[vr_filter.model==model],'conditional_loglik_variance',label='conditional log-likelihood variance')]],
           '', '## Mixture distribution results (N=128, d=2)',
           *[block for metric,rule,name in [('sliced_w1','lower','sliced W1'),('rff_mmd2','lower','RFF MMD²'),('covariance_error','lower','covariance error (Frobenius)'),
                                            ('mode_coverage','higher','trials keeping both modes'),('forward_ms','lower','forward time (ms)')]
             for block in ['',metric_table('Mixture cloud (N=128, d=2)',chosen,metric,rule,name)]],
           '', '## Filtering results (N=128)','',
           'Trials are successful/total; means condition on successful trials. PFJAX MVN has numerical failures on UNGM.',
           *[block for model in MODELS for metric,rule,name in [('rmse','lower','RMSE'),('coverage95','nominal','95% coverage'),('conditional_loglik_variance','lower','conditional log-likelihood variance')]
             for block in ['',metric_table(f'{MODELS[model]} filter (N=128)',fs[(fs.n==128)&(fs.model==model)],metric,rule,name)]],
           '', 'RMSE measures error against the simulated latent truth. Linear-model posterior error and likelihood error use an exact Kalman reference. Coverage is joint 95% Gaussian-ellipsoid coverage from the particle mean/covariance; it is a diagnostic and is not exact posterior credible-set coverage for multimodal targets. Bearing RMSE uses position only; the coverage ellipsoid uses all four state coordinates. Recovery requires position error below 2 units for three consecutive observed steps; unrecovered runs are right-censored at the end of the sequence.',
           '', 'Bearing RMSE averages squared error across both position coordinates before taking the square root. Post-occlusion RMSE instead uses Euclidean position error, so its squared error is the sum of both coordinate errors. Recovery-step means cap unrecovered runs at 20 steps and are not estimates of unrestricted recovery time.',
           '', 'Likelihood variance is calculated across filter seeds conditional on each observation sequence and then averaged. It is unavailable for expensive filters with only one seed per sequence. Log likelihood bias is meaningful against the exact reference only in the linear Gaussian case. Do not treat transport/Gaussian/eigen likelihood estimates as unbiased likelihood estimates.',
           '', '## Likelihood-gradient results','',
           'Gradient bias is ||mean(g) − g_exact||₂; variance is the trace of the sample gradient covariance; MSE is mean(||g−g_exact||₂²). The Cuthbert experiment uses its native tutorial data (T=50, q=0.4, r=0.8), 100 seeds at each N, and 21-point likelihood/gradient profile slices at N=3000. The PyTorch filter uses that same saved dataset and exact Kalman gradients. Stop-gradient estimators are assessed empirically; finite-particle unbiasedness is not assumed.']
    if native_tables:
        parts=[]
        for name,df in native_tables:
            df=df[df.n==128].copy()
            if name=='cuthbert_native':df['provider']='cuthbert';df['method']=np.where(df.stop_gradient,'systematic_stop','systematic')
            parts.append(df[['provider','method','trials','gradient_bias','gradient_variance','gradient_mse','loglik_variance','loglik_bias']])
        grad=pd.concat(parts,ignore_index=True)
        lines.extend(['','Both experiments use the Cuthbert tutorial dataset at N=128, so their rows share each table.'])
        for metric,rule,name in [('gradient_bias','lower','gradient bias'),('gradient_variance','lower','gradient variance'),('gradient_mse','lower','gradient MSE'),
                                 ('loglik_variance','lower','log-likelihood variance'),('loglik_bias','zero','log-likelihood bias')]:
            lines.extend(['',metric_table('PF likelihood gradient vs exact Kalman (N=128)',grad,metric,rule,name)])
    if len(ls):
        lines.extend(['','## Stochastic-volatility parameter learning','',
                      'The known state model is x_t=0.95 x_(t−1)+0.2 ε_t; y_t=s exp(x_t/2) η_t, with true s=1. Fit s by maximizing the particle likelihood on {0.6,0.8,1.0,1.2,1.4}. Ten independent datasets and common random numbers are used. These are coarse-grid estimates, not continuous optimizer convergence claims. The likelihood-score metric uses central differences in log(s), step 0.01. The independent reference is a numerical grid filter on [−4,4] with 1001 cells; a 2001-cell likelihood cross-check is stored per dataset. This reference is approximate, unlike the exact Kalman reference.', '',
                      metric_table('SV observation-scale learning (N=128)',ls,'parameter_absolute_error',label='|ŝ − 1|'),'',
                      metric_table('SV observation-scale learning (N=128)',ls,'mle_reference_error',label='|ŝ − ŝ_ref|'),'',
                      metric_table('SV observation-scale learning (N=128)',ls,'score_rmse',label='likelihood-score RMSE')])
    lines.extend(['','## Metric definitions and measurement limits','',
                  '- Sliced W1 averages exact one-dimensional Wasserstein-1 distances along 16 fixed random unit directions. In 1D it equals W1. It is not full multivariate W1 or W2.',
                  '- RFF MMD² approximates Gaussian-kernel squared maximum mean discrepancy using 128 fixed random Fourier features; bandwidth is sqrt(trace(Cw)). It is not an exact kernel MMD calculation.',
                  '- Covariances use population moments (no Bessel correction). PFJAX MVN uses its upstream corrected weighted sample covariance, so a small mismatch is expected.',
                  '- Mode coverage counts the two mixture neighborhoods centered at ±3 with radius 1.35 if each has output mass greater than 1%. Mode-mass error additionally compares the negative-halfspace mass. Coverage alone cannot show that mode shapes are preserved.',
                  '- Offspring-count variance and empirical count bias use repeated draws from an identical cloud. Soft-resampling offspring are compared with the actual mixture proposal 0.5w+0.5/N. Their output measures retain importance-correction weights; they are not falsely treated as equally weighted.',
                  '- Forward timing excludes initial compilation/warmup and includes adapter conversion and seed handling; the serial timing pass disables optional ancestor recovery used solely for offspring metrics. Torchfilter soft-weight gathering remains necessary. Raw static accuracy timings also include ancestor recovery. Julia timing includes Python-to-Julia IPC. These are integration timings, not isolated language-kernel speed rankings. JAX results synchronize with block_until_ready; TensorFlow results synchronize by converting to NumPy.',
                  '- Shared filtering normalizes the output correction weights of soft methods, as it does all output measures. This is a self-normalized particle approximation; unbiased likelihood estimates are not assumed. Input order differs across repository implementations (for example, Stone Soup sorts weights), which can change systematic and stratified resampling variance.',
                  '- PyTorch and TensorFlow local backward timings measure backward only. JAX local backward timings measure the compiled forward-and-backward gradient call and are explicitly labeled in raw CSV. Native Cuthbert likewise reports forward-and-backward together. Subtracting those times is not an exact backward-time measurement.',
                  '- Peak RSS is the process high-water mark, in MiB despite the legacy _mb field suffix, including imports and JIT caches; the separate timing pass uses a fresh interpreter for each method and does not calculate distribution metrics. Caches can accumulate across particle counts within that method. Raw accuracy RSS also includes metric calculations. RSS is not isolated per-call native allocation. Julia worker peak RSS is recorded separately from Linux VmHWM; total_peak_rss_mb_bound adds Python and worker high-water marks and is an upper bound because their peaks need not coincide. Other providers have no separate worker. Use these values for process budgeting, not a claim of algorithm-only peak memory.',
                  '- Results are CPU only. The installed NVIDIA GTX 960M has 2 GB VRAM and legacy Maxwell compute capability 5.0. No compatible CUDA benchmark environment was established. GPU speed and GPU memory claims remain unmeasured.',
                  '- Torchfilter upstream soft resampling leaves correction weights in input order. The adapter gathers the corresponding correction weights for the sampled ancestors; this deliberate fix is in adapters.py and vendor code is unchanged. Modern overrides 7.7.0 is used for Python 3.12 compatibility.',
                  '- The Julia bridge sends binary Float64 weights and receives Int64 ancestor indices. Both the explicitly passed RNG and Julia global RNG are seeded, because upstream multinomial/residual sampling includes a global randexp(T) draw. Protocol equivalence and repeated-seed reproducibility were checked for all six methods. Earlier incompletely seeded multinomial measurements were superseded and archived outside this report.',
                  '- FilterFlow runs in a separate Python 3.11 / TensorFlow 2.15.1 environment with float32 input states/weights; the other providers use Python 3.12 and float64 inputs (Julia uses Float64). Upstream internal random numbers and intermediates can use their own defaults. JIT compilation is excluded from forward timing. Cross-framework precision and wrapper differences must be considered.',
                  '', '## Source revisions',''])
    for source in sources:lines.append(f'- [{source["repository"]}](https://github.com/{source["repository"]}/tree/{source["commit"]}): `{source["commit"]}`')
    lines.extend(['','## Artifacts','',
                  '- `report.html`: self-contained interactive plots; open in a browser.',
                  '- `distribution_vs_runtime.png` and `filter_rmse.png`: exportable static plots.',
                  '- `latex/`: every results table as LaTeX (booktabs). `tables/` has one file per table, `all_tables.tex` has all of them, and `benchmark_tables.tex` is a standalone document that compiles them.',
                  '- `figures/`: 13 static PNG/PDF figures covering every section of this report; see `figures/README.md`. Rebuild with `scripts/plot_figures.py`.',
                  '- `*_summary.csv`: means, 95% intervals, gradient bias/variance, and conditional likelihood variance.',
                  '- `../<provider>/*.csv`: raw trial measurements; `status.jsonl` records successful and failed cells.',
                  '- `../../environment-main.txt`, `../../environment-tf.txt`, `../../hardware.txt`: dependency and hardware records.',
                  '- `../../../README.md`: setup and rerun commands.'])
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('LaTeX tables:',write_latex(out))
    print(json.dumps(dict(raw_static=len(s),raw_filter=len(f),raw_gradient=len(g),raw_learning=len(l),completed=len(completed),failed=len(failed),output=str(out)),indent=2))


if __name__=='__main__':main()
