"""Read completed runs; build auditable tables and original scientific figures."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import sys,json,csv
from pathlib import Path
BASE=Path(__file__).parent
sys.path.insert(0,str(BASE/'python_deps'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from mixing_solver import *

FIG=BASE/'figures';FIG.mkdir(exist_ok=True)
COL={'diffusion':'#656565','center':'#356B98','eccentric':'#87639D','pair':'#B3543B','switch':'#168277','mean_switch':'#A28B42','reverse':'#A35F85'}
LABEL={'diffusion':'D: diffusion','center':'C: centered','eccentric':'E: eccentric','pair':'B: steady pair','switch':'S: switching','mean_switch':'M: mean flow','reverse':'R: reversing'}
plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix','font.size':9,
    'axes.labelsize':9,'axes.titlesize':9,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,
    'axes.linewidth':.65,'lines.linewidth':1.6,'savefig.bbox':'tight','pdf.fonttype':42,
    'axes.spines.top':False,'axes.spines.right':False})

def load(tag):
    s=json.loads((BASE/'data'/f'{tag}.json').read_text())
    d=dict(np.load(BASE/'data'/f'{tag}.npz'))
    return s,d

def savefig(fig,name):
    fig.savefig(FIG/f'{name}.pdf',dpi=300);fig.savefig(FIG/f'{name}.png',dpi=180)
    plt.close(fig)

def image_field(ax,d,c,clim,title,cmap='RdBu_r'):
    z=np.full(d['mask'].shape,np.nan);z[d['mask']]=c
    im=ax.imshow(z.T,origin='lower',extent=(-1,1,-1,1),cmap=cmap,vmin=clim[0],vmax=clim[1],interpolation='nearest')
    ax.set(xlim=(-1.02,1.02),ylim=(-1.02,1.02),aspect='equal',title=title)
    ax.set_xticks([-1,0,1]);ax.set_yticks([-1,0,1]);return im

def hminus1(d):
    mask=d['mask'];n=mask.shape[0];g=make_grid(n);nb=g['neighbors'];N=len(g['x']);h=g['h']
    rows=[];cols=[];vals=[]
    for i in range(N):
        count=0
        for k in range(4):
            j=nb[k,i]
            if j!=i: rows.append(i);cols.append(j);vals.append(-1/h**2);count+=1
        rows.append(i);cols.append(i);vals.append(count/h**2)
    A=coo_matrix((vals,(rows,cols)),shape=(N,N)).tocsc()
    lu=splu(A[1:,1:])
    norms=[]
    for c in d['snapshots']:
        b=c-c.mean();phi=np.zeros(N);phi[1:]=lu.solve(b[1:]);phi-=phi.mean()
        norms.append(float(np.mean(b*phi)))
    return np.array(norms)

def summarize():
    records=[]
    for file in sorted((BASE/'data').glob('*.json')):
        s=json.loads(file.read_text());s['tag']=file.stem;records.append(s)
    keys=sorted(set().union(*(x.keys() for x in records)))
    with (BASE/'all_runs.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,keys);writer.writeheader();writer.writerows(records)
    return records

def flow_figure():
    n=220;x=np.linspace(-1,1,n);X,Y=np.meshgrid(x,x,indexing='ij');mask=X*X+Y*Y<=1
    G,_=gram_matrix();fig,ax=plt.subplots(2,3,figsize=(7.1,4.5),layout='constrained')
    vals=[psi_derivatives(X,Y,q,.5,1.) for q in [0.,-.45,.45]]
    for col,proto in enumerate(['center','eccentric','pair']):
        w=coefficients(proto,[0.],G)[0]
        p,px,py,pxx,pyy,pxy=[sum(w[j]*vals[j][k] for j in range(3)) for k in range(6)]
        omg=-(pxx+pyy);strain=4*pxy**2+(pyy-pxx)**2
        omg[~mask]=np.nan;strain[~mask]=np.nan;p[~mask]=np.nan
        im1=ax[0,col].pcolormesh(X,Y,omg,cmap='RdBu_r',vmin=-3,vmax=3,shading='auto',rasterized=True)
        ax[0,col].contour(X,Y,p,levels=9,colors='k',linewidths=.35,alpha=.65)
        im2=ax[1,col].pcolormesh(X,Y,strain,cmap='viridis',vmin=0,vmax=5,shading='auto',rasterized=True)
        ax[0,col].set_title(f'({chr(97+col)}) {LABEL[proto]}')
        for row in [0,1]:
            ax[row,col].set_aspect('equal');ax[row,col].set(xlim=(-1,1),ylim=(-1,1));ax[row,col].set_xticks([-1,0,1]);ax[row,col].set_yticks([-1,0,1])
    fig.colorbar(im1,ax=ax[0,:],label=r'Vorticity $\omega$ (streamfunction contours)',shrink=.75,extend='both')
    fig.colorbar(im2,ax=ax[1,:],label=r'Local dissipation $2S:S$',shrink=.75,extend='max')
    savefig(fig,'01_flow')

def snapshots_figure():
    fig,ax=plt.subplots(3,3,figsize=(7.1,6.4),layout='constrained')
    for i,proto in enumerate(['center','pair','switch']):
        s,d=load('main_switch_4' if proto=='switch' else 'main_'+proto)
        for j,(sn,t) in enumerate([(0,0),(2,20),(4,40)]):
            im=image_field(ax[i,j],d,d['snapshots'][sn],(.05,.95),f'{LABEL[proto]}, t={t}')
            if j: ax[i,j].set_yticklabels([])
    fig.colorbar(im,ax=ax.ravel().tolist(),label='Concentration c (same color scale)',shrink=.68)
    savefig(fig,'02_scalar')

def dynamics_figure():
    fig,ax=plt.subplots(1,3,figsize=(7.1,2.5),layout='constrained')
    for proto in ['diffusion','center','eccentric','pair','switch','mean_switch']:
        s,d=load('main_switch_4' if proto=='switch' else 'main_'+proto)
        h=d['history'];ls='--' if proto in ['diffusion','mean_switch'] else '-'
        ax[0].semilogy(h[:,0],h[:,2]/h[0,2],color=COL[proto],label=LABEL[proto],ls=ls)
        if proto in ['center','pair','switch']:
            ax[1].plot(h[:,0],h[:,4]/h[0,2],color=COL[proto],label=proto)
            ax[2].plot(h[:,0],h[:,3]/h[0,2],color=COL[proto],label=LABEL[proto])
    ax[0].axhline(.05,color='0.65',lw=.7,ls=':');ax[0].set(xlabel='Time',ylabel=r'$V(t)/V(0)$',title='(a) Whole-domain variance')
    handles,labels=ax[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.18),ncol=3,fontsize=7,frameon=False)
    ax[1].set(xlabel='Time',ylabel=r'$\chi_h(t)/V(0)$',title='(b) Scalar dissipation')
    ax[2].set(xlabel='Time',ylabel=r'$V_{wall}(t)/V(0)$',title='(c) Near-wall residual')
    savefig(fig,'03_dynamics')

def frequency_figure():
    fig,ax=plt.subplots(1,3,figsize=(7.1,2.5),layout='constrained')
    taus=np.array([.5,1,2,4,8,10,20,40,80])
    js=[load(f'main_switch_{t:g}')[0]['J'] for t in taus]
    ax[0].semilogx(taus,js,'o-',color=COL['switch'],ms=3,label='Switching')
    for proto,ls in [('pair','-'),('mean_switch','--')]:
        ax[0].axhline(load('main_'+proto)[0]['J'],color=COL[proto],ls=ls,label=LABEL[proto])
    ax[0].set(xlabel=r'Period $\tau$',ylabel=r'$J=V(T)/V(0)$',title='(a) Finite-time frequency response')
    ax[0].legend(frameon=False,fontsize=6.5)
    _,dmean=load('fast_mean');errors=[];fast=np.array([.125,.25,.5,1.,2.,4.,8.])
    for t in fast:
        _,d=load(f'fast_switch_{t:g}')
        errors.append(np.linalg.norm(d['final']-dmean['final'])/np.linalg.norm(d['initial']-d['initial'].mean()))
    ax[1].loglog(fast,errors,'o-',color=COL['switch'],ms=3,label='N=128')
    _,fine_mean=load('fast192_mean');fine_t=np.array([.125,.25,.5]);fine_error=[]
    for t in fine_t:
        _,d=load(f'fast192_switch_{t:g}')
        fine_error.append(np.linalg.norm(d['final']-fine_mean['final'])/np.linalg.norm(d['initial']-d['initial'].mean()))
    ax[1].loglog(fine_t,fine_error,'s',ms=6,mfc='none',mec=COL['center'],label='N=192')
    ax[1].loglog(fast,errors[0]*fast/fast[0],':',color='0.4',label=r'$O(\tau)$')
    ax[1].set(xlabel=r'Period $\tau$',ylabel='Relative final-field error',title='(b) Approach to the mean flow');ax[1].legend(frameon=False)
    G,_=gram_matrix();w=coefficients('switch',np.linspace(0,1,1000,endpoint=False),G,tau=1.)
    m=w.mean(0);pair=coefficients('pair',[0],G)[0]
    ax[2].plot(w[:,1],w[:,2],color=COL['switch'],lw=1.5,label='Instantaneous unit budget')
    ax[2].plot(m[1],m[2],'*',ms=10,color=COL['mean_switch'],label=r'Mean: $P=0.62673$')
    ax[2].plot(pair[1],pair[2],'o',ms=4,color=COL['pair'],label=r'Steady pair: $P=1$')
    ax[2].set(xlabel=r'$a_L$',ylabel=r'$a_R$',title='(c) Dissipation-norm geometry',aspect='equal');ax[2].legend(frameon=False,fontsize=6.0,loc='lower left')
    savefig(fig,'04_frequency')

def robustness_figure():
    fig,ax=plt.subplots(1,3,figsize=(7.1,2.75),layout='constrained')
    protos=['center','eccentric','pair','switch'];names=['Horizontal','Diagonal','Vertical','Localized','Radial']
    tags=['main','initial_diag','initial_vert','initial_blob','initial_radial'];matrix=[]
    for tag in tags:
        row=[]
        for proto in protos:
            key='main_switch_4' if tag=='main' and proto=='switch' else f'{tag}_{proto}'
            row.append(load(key)[0]['J'])
        matrix.append(row)
    matrix=np.array(matrix);relative=matrix/matrix.min(axis=1)[:,None]
    im=ax[0].imshow(np.log10(relative),cmap='YlOrRd',aspect='auto',vmin=0,vmax=1.1)
    ax[0].set_xticks(range(4),['C','E','B','S']);ax[0].set_yticks(range(5),names)
    for i in range(5):
        for j in range(4): ax[0].text(j,i,f'{relative[i,j]:.2f}',ha='center',va='center',fontsize=7,color='white' if np.log10(relative[i,j])>.7 else 'black')
    ax[0].set_title('(a) Initial-condition ranking');fig.colorbar(im,ax=ax[0],label=r'$\log_{10}(J/J_{best})$',shrink=.7)
    asp=[.5,1.,2.]
    for proto in ['center','pair','switch']:
        vals=[load(f'shape_{a:g}_{proto}')[0]['J'] for a in asp]
        ax[1].plot(asp,vals,'o-',color=COL[proto],label=LABEL[proto],ms=3)
    ax[1].set(xlabel=r'Envelope aspect $\lambda$',ylabel='J',title='(b) Shape proxy, equal budget');ax[1].set_xticks(asp);ax[1].legend(frameon=False,fontsize=6.5)
    for a,marker in zip(asp,['s','o','^']):
        ts=[.5,2.,4.,8.,20.]
        vals=[]
        for t in ts:
            key=f'shape_{a:g}_switch' if t==4 else (f'main_switch_{t:g}' if a==1 else f'map_{a:g}_{t:g}')
            vals.append(load(key)[0]['J'])
        ax[2].semilogx(ts,vals,marker+'-',ms=3,label=fr'$\lambda={a:g}$')
    ax[2].set(xlabel=r'Period $\tau$',ylabel='J',title='(c) Frequency--shape coupling');ax[2].legend(frameon=False,fontsize=7)
    savefig(fig,'05_robustness')

def sensitivity_figure():
    fig,ax=plt.subplots(1,3,figsize=(7.1,2.6),layout='constrained')
    for proto in ['center','pair','switch']:
        kap=[.0015,.003,.006,.012];vals=[]
        for k in kap:
            tag=('main_switch_4' if proto=='switch' else 'main_'+proto) if k==.003 else f'diff_{k:g}_{proto}'
            vals.append(load(tag)[0]['J'])
        ax[0].loglog(kap,vals,'o-',color=COL[proto],label=LABEL[proto],ms=3)
        speed=[.5,1.,2.];vals=[]
        for u in speed:
            tag=('main_switch_4' if proto=='switch' else 'main_'+proto) if u==1 else f'speed_{u:g}_{proto}'
            vals.append(load(tag)[0]['J'])
        ax[1].semilogy(np.array(speed)**2,vals,'o-',color=COL[proto],label=LABEL[proto],ms=3)
    ax[0].set(xlabel=r'Diffusivity $\kappa$ ($\times10^{-3}$)',ylabel='J',title='(a) Diffusivity sensitivity')
    ax[0].set_xticks([.0015,.003,.006,.012],['1.5','3','6','12'])
    ax[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[0].legend(frameon=False,fontsize=6.5)
    ax[1].set(xlabel=r'Relative power $b^2$',ylabel='J',title='(b) Increasing speed has a cost')
    ax[1].set_xticks([.25,1,4])
    for tau in [4,20]:
        phases=[0,.25,.5,.75];vals=[]
        for p in phases:
            tag=f'main_switch_{tau:g}' if p==0 else f'phase_{p:g}_{tau:g}'
            vals.append(load(tag)[0]['J'])
        ax[2].plot(phases,vals,'o-',label=fr'$\tau={tau}$',ms=3)
    ax[2].set(xlabel='Starting phase (cycles)',ylabel='J',title='(c) Finite-time phase sensitivity')
    ax[2].legend(frameon=False)
    savefig(fig,'06_sensitivity')

def validation_figure():
    fig,ax=plt.subplots(1,3,figsize=(7.1,2.55),layout='constrained')
    ns=[64,96,128,192,256,384]
    for proto in ['pair','switch']:
        vals=[]
        for n in ns:
            tag=f'conv_{proto}_{n}' if n!=192 else ('main_switch_4' if proto=='switch' else 'main_pair')
            vals.append(load(tag)[0]['J'])
        ax[0].plot(ns,vals,'o-',label=LABEL[proto],color=COL[proto],ms=3)
    ax[0].set(xlabel='Grid size N',ylabel='J',title='(a) Staircase-domain refinement');ax[0].legend(frameon=False)
    diffs=[];fielderr=[]
    for n in ns:
        s,d=load(f'conv_diffusion_{n}' if n!=192 else 'main_diffusion')
        diffs.append(s['analytic_J_error'])
        exact=.5+(d['initial']-.5)*np.exp(-s['kappa']*ALPHA**2*s['T'])
        fielderr.append(np.linalg.norm(d['final']-exact)/np.linalg.norm(exact-.5))
    ax[1].loglog(ns,diffs,'o-',label='Variance relative error',ms=3)
    ax[1].loglog(ns,fielderr,'s-',label='Final-field relative error',ms=3)
    ax[1].set(xlabel='Grid size N',ylabel='Relative analytic error',title='(b) Disk Bessel benchmark');ax[1].legend(frameon=False,fontsize=6.5)
    ax[1].set_xticks([64,128,256,384],['64','128','256','384'])
    ax[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    for proto in ['center','pair','switch']:
        s,d=load('main_switch_4' if proto=='switch' else 'main_'+proto);h=d['history'];dt=np.diff(h[:,0])
        integral=np.r_[0,np.cumsum(dt*(h[1:,4]+h[:-1,4])/2)]
        residual=np.abs((h[0,2]-h[:,2]-integral)/h[0,2])
        ax[2].semilogy(h[:,0],np.maximum(residual,1e-16),color=COL[proto],label=LABEL[proto])
    ax[2].set(xlabel='Time',ylabel='Normalized variance-budget defect',title='(c) Dissipation closure')
    savefig(fig,'07_validation')

def wall_multiscale_figure(metrics):
    fig,ax=plt.subplots(1,3,figsize=(7.1,2.6),layout='constrained')
    bins=np.linspace(0,1,17);norms={}
    for proto in ['center','pair','switch']:
        s,d=load('main_switch_4' if proto=='switch' else 'main_'+proto)
        r=np.sqrt(d['x']**2+d['y']**2);c=d['final'];initial=d['initial'];V0=np.var(initial)
        profile=[np.mean((c[(r>=lo)&(r<hi)]-c.mean())**2)/V0 for lo,hi in zip(bins[:-1],bins[1:])]
        ax[0].plot((bins[1:]+bins[:-1])/2,profile,'o-',color=COL[proto],label=LABEL[proto],ms=2.5)
        Ds=np.linspace(.05,.3,30);fractions=[np.sum((c[r>1-delta]-c.mean())**2)/np.sum((c-c.mean())**2) for delta in Ds]
        ax[1].plot(Ds,fractions,color=COL[proto])
        H=hminus1(d);norms[proto]=H.tolist()
        ax[2].semilogy([0,10,20,30,40],H/H[0],'o-',color=COL[proto],label=LABEL[proto],ms=3)
    ds=np.linspace(.05,.3,30);ax[1].plot(ds,1-(1-ds)**2,':',color='0.4',label='Geometric area fraction')
    ax[0].set(xlabel='Radius r',ylabel='Annular variance / V(0)',title='(a) Radial residual profile');ax[0].legend(frameon=False,fontsize=6.2)
    ax[1].set(xlabel=r'Wall-band thickness $\delta$',ylabel='Fraction of remaining variance',title='(b) Robust wall-band definition');ax[1].legend(frameon=False,fontsize=6.2)
    ax[2].set(xlabel='Time',ylabel=r'$\|c-\bar c\|_{H^{-1}}^2$ / initial',title='(c) Neumann inverse-Laplacian metric')
    metrics['Hminus1_squared']=norms
    savefig(fig,'08_wall_multiscale')

def build_metrics(records):
    metrics={'run_count':len(records),'nonpilot_count':sum(not r['tag'].startswith('pilot_') for r in records)}
    G,e=gram_matrix();w=coefficients('switch',np.arange(8192)/8192,G,tau=1.)
    m=w.mean(0);metrics.update(gram=G.tolist(),gram_condition=float(np.linalg.cond(G)),mean_coefficients=m.tolist(),
        effective_mean_power=float(m@G@m),gram_identity_error=e)
    main=[r for r in records if r['tag'].startswith('main_')]
    metrics['main_mass_max']=max(abs(r['mass_relative_drift']) for r in main)
    metrics['main_budget_residual_max']=max(abs(r['variance_budget_relative_residual']) for r in main)
    metrics['main_results']={r['tag']:r for r in main}
    metrics['time_refinement']={r['tag']:r for r in records if r['tag'].startswith('time_')}
    metrics['convergence']={r['tag']:r for r in records if r['tag'].startswith('conv_')}
    metrics['initial_results']={r['tag']:r for r in records if r['tag'].startswith('initial_')}
    return metrics

if __name__=='__main__':
    records=summarize();metrics=build_metrics(records)
    flow_figure();snapshots_figure();dynamics_figure();frequency_figure();robustness_figure()
    sensitivity_figure();validation_figure();wall_multiscale_figure(metrics)
    (BASE/'analysis_metrics.json').write_text(json.dumps(metrics,indent=2))
    print('Figures and tables written; completed runs:',len(records),flush=True)
