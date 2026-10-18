"""Read-only scientific re-review computations and additional paper graphics."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,csv
from pathlib import Path
ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT/'python_deps'))
import numpy as np
from mixing_solver import make_grid,flow_modes,gram_matrix,coefficients
from review_initials import SEEDS
from postprocess import plt,COL,savefig,load

def flow_audit(a,sigma,aspect,n=192):
    g=make_grid(n);m=flow_modes(g,a,sigma,aspect);G,_=gram_matrix(a,sigma,aspect)
    derivatives=[]
    for mode in range(3):
        U=np.zeros((n,n));V=U.copy()
        U[g['mask']]=(m['faces'][mode,0]-m['faces'][mode,1])/2
        V[g['mask']]=(m['faces'][mode,2]-m['faces'][mode,3])/2
        ux,uy=np.gradient(U,g['h'],edge_order=2);vx,vy=np.gradient(V,g['h'],edge_order=2)
        derivatives.append(np.array([ux[g['mask']],uy[g['mask']],vx[g['mask']],vy[g['mask']]]))
    D=np.array(derivatives);ux,uy,vx,vy=[D[:,k,:] for k in range(4)];shear=uy+vx
    Gh=(2*ux@ux.T+2*vy@vy.T+shear@shear.T)/len(g['x'])
    result=dict(a=a,sigma=sigma,aspect=aspect,n=n)
    for proto in ['center','pair','switch']:
        c=coefficients(proto,np.linspace(0,4,257),G,tau=4.)
        ph=np.einsum('ti,ij,tj->t',c,Gh,c)
        result[proto+'_max_relative_power_error']=float(np.max(abs(ph-1)))
    return result

def main():
    p=['center','eccentric','pair','switch']
    A=np.array([[load(f'holdout_{seed}_{proto}')[0]['J'] for proto in p] for seed in SEEDS])
    wins=np.bincount(A.argmin(axis=1),minlength=4)
    H=dict(seeds=SEEDS,protocols=p,J=A.tolist(),wins=dict(zip(p,map(int,wins))),
           median_J=dict(zip(p,map(float,np.median(A,axis=0)))),
           switch_to_pair_ratio_quartiles=np.quantile(A[:,3]/A[:,2],[.25,.5,.75]).tolist())
    E=[]
    for n,tag in [(128,'fast'),(192,'fast192')]:
        _,dm=load('fast_mean' if n==128 else 'fast192_mean')
        ts=np.array([.125,.25,.5]);errs=[]
        for t in ts:
            _,d=load(f'{tag}_switch_{t:g}')
            errs.append(np.linalg.norm(d['final']-dm['final'])/np.linalg.norm(d['initial']-d['initial'].mean()))
        E.append(dict(n=n,taus=ts.tolist(),errors=list(map(float,errs)),order=float(np.polyfit(np.log(ts),np.log(errs),1)[0])))
    refines=[]
    for kind in ['radial','blob']:
        for proto in ['center','pair','switch']:
            s,_=load(f'initial_{kind}_{proto}');sf,_=load(f'refine_{kind}_{proto}_256')
            refines.append(dict(kind=kind,protocol=proto,J192=s['J'],J256=sf['J'],relative_change=sf['J']/s['J']-1))
    checks=[flow_audit(.45,.5,asp) for asp in [.5,1,2]]
    checks += [flow_audit(a,s,1.) for a in [.3,.6] for s in [.4,.65]]
    out=dict(holdout=H,averaging_two_grids=E,ranking_refinement=refines,flow_power_checks=checks)
    (ROOT/'submission_review_results.json').write_text(json.dumps(out,indent=2))
    # Manufactured-solution and ranking checks.
    mms=json.loads((ROOT/'mms_checks.json').read_text())[:4]
    fig,ax=plt.subplots(1,2,figsize=(7.1,2.9),layout='constrained')
    ns=[r['n'] for r in mms];es=[r['relative_field_L2'] for r in mms]
    ax[0].loglog(ns,es,'o-',color='#356B98',label='Forced advection--diffusion')
    ax[0].set_xticks(ns,[str(n) for n in ns]);ax[0].xaxis.set_minor_formatter(plt.NullFormatter())
    ax[0].set(xlabel='Grid size N',ylabel='Relative exact-field error',title='(a) Nonzero-advection manufactured solution')
    labels=[('rad ' if r['kind']=='radial' else 'blob ')+{'center':'C','pair':'B','switch':'S'}[r['protocol']] for r in refines]
    ax[1].bar(np.arange(6),[abs(r['relative_change'])*100 for r in refines],
        color=[COL[r['protocol']] for r in refines],width=.65)
    ax[1].set_xticks(np.arange(6),labels,rotation=25);ax[1].set(xlabel='Initial type and protocol',ylabel=r'$|J_{256}/J_{192}-1|$ (%)',title='(b) Independent ranking-refinement check')
    savefig(fig,'09_additional_verification')
    # All12 unselected synthetic cases, preserving pairing.
    fig,ax=plt.subplots(1,2,figsize=(7.1,2.9),layout='constrained')
    for row in A:
        ax[0].semilogy(np.arange(4),row,'o-',color='.65',alpha=.65,lw=.7,ms=2.5)
    for j,proto in enumerate(p):
        ax[0].plot(j,np.median(A[:,j]),'D',color=COL[proto],ms=6,label=proto)
    ax[0].set_xticks(range(4),['C','E','B','S']);ax[0].set(xlabel='Protocol (same budget)',ylabel='J for each initial condition',title='(a)12 pre-specified paired cases')
    ratios=A[:,3]/A[:,2]
    ax[1].bar(np.arange(12)+1,ratios,color=[COL['switch'] if r<1 else COL['pair'] for r in ratios],width=.65)
    ax[1].axhline(1,color='.25',ls='--',lw=.8)
    ax[1].set_xticks([1,4,7,10,12]);ax[1].set(xlabel='Seed index (all cases shown)',ylabel=r'$J_S/J_B$',title='(b) Switching / steady-pair residual')
    savefig(fig,'10_holdout')
    # Add the second independent grid to the averaging panel.
    print(json.dumps(out,indent=2),flush=True)

if __name__=='__main__':main()
