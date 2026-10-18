"""Independent nonzero-advection PDE consistency test with an exact source."""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT/'python_deps'))
import numpy as np
from numba import njit
from mixing_solver import make_grid,flow_modes,gram_matrix,coefficients,psi_derivatives,_rhs

@njit(cache=True)
def integrate(c,nb,B,base,adv,cs,cm,ce,amp_s,amp_m,amp_e,dt,diff):
    k1=np.empty_like(c);k2=np.empty_like(c);k3=np.empty_like(c);k4=np.empty_like(c);tmp=np.empty_like(c)
    for step in range(len(cs)):
        _rhs(c,nb,B,cs[step],diff,k1)
        for i in range(len(c)):
            k1[i]+=amp_s[step]*(base[i]+cs[step,0]*adv[0,i]+cs[step,1]*adv[1,i]+cs[step,2]*adv[2,i])
            tmp[i]=c[i]+dt*k1[i]/2
        _rhs(tmp,nb,B,cm[step],diff,k2)
        for i in range(len(c)):
            k2[i]+=amp_m[step]*(base[i]+cm[step,0]*adv[0,i]+cm[step,1]*adv[1,i]+cm[step,2]*adv[2,i])
            tmp[i]=c[i]+dt*k2[i]/2
        _rhs(tmp,nb,B,cm[step],diff,k3)
        for i in range(len(c)):
            k3[i]+=amp_m[step]*(base[i]+cm[step,0]*adv[0,i]+cm[step,1]*adv[1,i]+cm[step,2]*adv[2,i])
            tmp[i]=c[i]+dt*k3[i]
        _rhs(tmp,nb,B,ce[step],diff,k4)
        for i in range(len(c)):
            k4[i]+=amp_e[step]*(base[i]+ce[step,0]*adv[0,i]+ce[step,1]*adv[1,i]+ce[step,2]*adv[2,i])
            c[i]+=dt*(k1[i]+2*k2[i]+2*k3[i]+k4[i])/6
    return c

def run(n,dt_factor=1.):
    start=time.perf_counter();g=make_grid(n);m=flow_modes(g);G,_=gram_matrix()
    x,y,h=g['x'],g['y'],g['h'];r2=x*x+y*y
    f=x*(1-r2)**2;fx=(1-r2)**2-4*x*x*(1-r2);fy=-4*x*y*(1-r2)
    lap=x*(-16+24*r2)
    kappa=.003;T=2.;tau=1.
    adv=[]
    for q in [0.,-.45,.45]:
        _,px,py,*_=psi_derivatives(x,y,q,.5,1.)
        adv.append(py*fx-px*fy)
    adv=np.ascontiguousarray(adv);base=-f-kappa*lap
    dt=min(.005,.16*h*h/kappa)*dt_factor;steps=int(np.ceil(T/dt));dt=T/steps
    t=np.arange(steps)*dt
    cs=coefficients('switch',t,G,tau);cm=coefficients('switch',t+dt/2,G,tau);ce=coefficients('switch',t+dt,G,tau)
    c0=.5+.3*f;exact=.5+.3*np.exp(-T)*f
    final=integrate(c0.copy(),g['neighbors'],m['B'],base,adv,cs,cm,ce,
        .3*np.exp(-t),.3*np.exp(-t-dt/2),.3*np.exp(-t-dt),dt,kappa/h**2)
    error=final-exact
    folder=ROOT/'verification';folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/f'mms_{n}_{dt_factor:g}.npz',initial=c0,final=final,exact=exact,mask=g['mask'],x=x,y=y)
    return dict(n=n,T=T,kappa=kappa,tau=tau,dt=dt,steps=steps,
        relative_field_L2=float(np.linalg.norm(error)/np.linalg.norm(exact-.5)),
        absolute_Linf=float(np.max(abs(error))),absolute_rms=float(np.sqrt(np.mean(error**2))),
        dt_factor=dt_factor,source='0.3 exp(-t) [-f+u dot grad f-kappa Delta f]',
        exact='0.5+0.3 exp(-t) x(1-r^2)^2',runtime_seconds=time.perf_counter()-start)

if __name__=='__main__':
    report=[run(n) for n in [64,128,192,256]]
    report.append(run(128,.5))
    (ROOT/'mms_checks.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
