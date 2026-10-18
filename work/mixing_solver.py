"""Conservative disk mixing experiment. Python 3.10, NumPy/SciPy/Numba.

The flow is a prescribed, no-slip streamfunction family, not a rigid-spoon
Navier--Stokes solution. No clipping or renormalisation of scalar values.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys, json, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'python_deps'))
import numpy as np
from scipy.special import j1, jnp_zeros
from numpy.polynomial.legendre import leggauss
from numba import njit

ALPHA = float(jnp_zeros(1, 1)[0])

def make_grid(n):
    h = 2./n
    coords = -1 + (np.arange(n)+.5)*h
    X,Y = np.meshgrid(coords,coords,indexing='ij')
    mask = X*X+Y*Y < 1
    ids = np.full((n,n), -1, np.int32)
    ids[mask] = np.arange(mask.sum(), dtype=np.int32)
    I,J = np.nonzero(mask)
    neighbors = np.empty((4,len(I)), np.int32)
    for k,(di,dj) in enumerate([(1,0),(-1,0),(0,1),(0,-1)]):
        ni,nj = I+di,J+dj
        valid = (ni>=0)&(ni<n)&(nj>=0)&(nj<n)
        nbr = np.full(len(I),-1,np.int32)
        nbr[valid] = ids[ni[valid],nj[valid]]
        neighbors[k] = np.where(nbr>=0,nbr,ids[I,J])
    # Corner streamfunction is zero on every boundary face. This guarantees
    # telescoping fluxes even on the staircase approximation of the disk.
    padded = np.pad(mask,1,constant_values=False)
    vertex = padded[:-1,:-1]&padded[1:,:-1]&padded[:-1,1:]&padded[1:,1:]
    return dict(n=n,h=h,mask=mask,I=I,J=J,neighbors=neighbors,
                x=X[mask],y=Y[mask],vertex=vertex,area=float(mask.sum()*h*h))

def psi_derivatives(x,y,q,sigma,aspect):
    sx2,sy2 = sigma*sigma*aspect,sigma*sigma/aspect
    dx,dy = x-q,y
    r2 = x*x+y*y
    b = (1-r2)**2
    bx,by = -4*x*(1-r2),-4*y*(1-r2)
    ex,ey = -2*dx/sx2,-2*dy/sy2
    e = np.exp(-dx*dx/sx2-dy*dy/sy2)
    p = b*e
    px,py = e*(bx+b*ex),e*(by+b*ey)
    pxx = e*(-4*(1-r2)+8*x*x+2*bx*ex+b*(ex*ex-2/sx2))
    pyy = e*(-4*(1-r2)+8*y*y+2*by*ey+b*(ey*ey-2/sy2))
    pxy = e*(8*x*y+bx*ey+by*ex+b*ex*ey)
    return p,px,py,pxx,pyy,pxy

def gram_matrix(a=.45,sigma=.50,aspect=1.,quadrature=120):
    # Area-average quadrature in r^2: dA/pi = d(r^2)dtheta/(2pi).
    z,w = leggauss(quadrature)
    radius = np.sqrt((z+1)/2)
    theta = np.arange(2*quadrature)*np.pi/quadrature
    x,y = radius[:,None]*np.cos(theta),radius[:,None]*np.sin(theta)
    weights = np.broadcast_to((w/2)[:,None]/len(theta),x.shape)
    values = [psi_derivatives(x,y,q,sigma,aspect) for q in [0.,-a,a]]
    omg = np.array([-(v[3]+v[4]) for v in values])
    sd = np.array([v[4]-v[3] for v in values])
    sxy = np.array([v[5] for v in values])
    G = np.einsum('ikl,jkl,kl->ij',omg,omg,weights)
    strain = np.einsum('ikl,jkl,kl->ij',sd,sd,weights)+4*np.einsum('ikl,jkl,kl->ij',sxy,sxy,weights)
    discrepancy = float(np.max(np.abs(strain-G))/np.max(np.abs(G)))
    return G,discrepancy

def flow_modes(g,a=.45,sigma=.50,aspect=1.):
    n,h,I,J = g['n'],g['h'],g['I'],g['J']
    edge = np.linspace(-1,1,n+1)
    X,Y = np.meshgrid(edge,edge,indexing='ij')
    B=[]; faces=[]
    for q in [0.,-a,a]:
        p=psi_derivatives(X,Y,q,sigma,aspect)[0]
        p[~g['vertex']] = 0
        ne,nw,se,sw = p[I+1,J+1],p[I,J+1],p[I+1,J],p[I,J]
        # Outward signed normal velocity: east, west, north, south.
        f=np.array([(ne-se)/h,-(nw-sw)/h,-(ne-nw)/h,(se-sw)/h])
        faces.append(f)
        B.append(-f/(2*h))
    return dict(B=np.ascontiguousarray(B),faces=np.array(faces))

def initial_scalar(g,kind='dipole',angle=0.):
    x,y=g['x'],g['y']; r=np.sqrt(x*x+y*y)
    if kind.startswith('mix'):
        from review_initials import mixture
        return mixture(g,int(kind[3:]))
    if kind=='dipole':
        angular=np.where(r>0,(x*np.cos(angle)+y*np.sin(angle))/np.maximum(r,1e-30),0)
        return .5+.45*j1(ALPHA*r)/j1(ALPHA)*angular
    if kind=='blob':
        qx,qy=.45*np.cos(angle),.45*np.sin(angle)
        c=np.exp(-((x-qx)**2+(y-qy)**2)/(.22**2))
        taper=np.clip((.98-r)/.15,0,1)
        taper=taper*taper*(3-2*taper)
        return c*taper
    if kind=='radial':
        return .5+.4*(1-r*r)**2
    raise ValueError(kind)

def coefficients(protocol,t,G,tau=4.,speed=1.,power=1.,phase=0.):
    t=np.atleast_1d(t)
    w=np.zeros((len(t),3))
    if protocol=='diffusion': return w
    if protocol=='mean_switch':
        phase=np.arange(8192)/8192
        s=.5*(1+np.sin(2*np.pi*phase))
        v=np.column_stack([np.zeros_like(s),s,1-s])
        v/=np.sqrt(np.einsum('ti,ij,tj->t',v,G,v))[:,None]
        return np.ascontiguousarray(np.broadcast_to(v.mean(axis=0)*speed*np.sqrt(power),w.shape))
    if protocol in ['center','reverse']:
        w[:,0]=1
    elif protocol=='eccentric': w[:,1]=1
    elif protocol=='pair': w[:,1:]=.5
    elif protocol=='switch':
        # Sinusoidal blending is smooth, has a fixed spatial support family,
        # and does not claim to represent the trajectory of a solid spoon.
        s=.5*(1+np.sin(2*np.pi*(t/tau+phase)))
        w[:,1],w[:,2]=s,1-s
    else: raise ValueError(protocol)
    norm=np.sqrt(np.einsum('ti,ij,tj->t',w,G,w))
    w*=np.sqrt(power)/norm[:,None]
    if protocol=='reverse':
        # Constant total dissipation over whole periods, not instantaneous.
        w[:,0]*=np.sqrt(2)*np.cos(2*np.pi*t/tau)
    return np.ascontiguousarray(w*speed)

@njit(cache=True)
def _rhs(c,nb,B,coef,diff,out):
    for i in range(len(c)):
        val=0.
        for k in range(4):
            adv=coef[0]*B[0,k,i]+coef[1]*B[1,k,i]+coef[2]*B[2,k,i]
            val+=(diff+adv)*(c[nb[k,i]]-c[i])
        out[i]=val

def rhs(c,g,modes,coef,kappa):
    z=np.empty_like(c)
    _rhs(c,g['neighbors'],modes['B'],np.asarray(coef),kappa/g['h']**2,z)
    return z

@njit(cache=True)
def _diagnostic(c,nb,wall,diff):
    mean=np.mean(c); v=0.; wv=0.; nw=0; diss=0.
    for i in range(len(c)):
        d=c[i]-mean
        v+=d*d
        if wall[i]: wv+=d*d; nw+=1
        for k in range(4): diss+=(c[i]-c[nb[k,i]])**2
    return mean,v/len(c),wv/max(nw,1),diff*diss/len(c),np.min(c),np.max(c)

@njit(cache=True)
def _integrate(c,nb,B,wall,cs,cm,ce,dt,diff,output_every):
    steps=len(cs)
    records=steps//output_every+2
    hist=np.empty((records,7)); hi=1
    d=_diagnostic(c,nb,wall,diff)
    hist[0,0]=0.
    for j in range(6): hist[0,j+1]=d[j]
    snaps=np.empty((5,len(c))); snaps[0]=c; si=1
    k1=np.empty_like(c);k2=np.empty_like(c);k3=np.empty_like(c);k4=np.empty_like(c);tmp=np.empty_like(c)
    min_all=np.min(c);max_all=np.max(c)
    for step in range(steps):
        _rhs(c,nb,B,cs[step],diff,k1)
        for i in range(len(c)): tmp[i]=c[i]+.5*dt*k1[i]
        _rhs(tmp,nb,B,cm[step],diff,k2)
        for i in range(len(c)): tmp[i]=c[i]+.5*dt*k2[i]
        _rhs(tmp,nb,B,cm[step],diff,k3)
        for i in range(len(c)): tmp[i]=c[i]+dt*k3[i]
        _rhs(tmp,nb,B,ce[step],diff,k4)
        for i in range(len(c)):
            c[i]+=dt*(k1[i]+2*k2[i]+2*k3[i]+k4[i])/6
            min_all=min(min_all,c[i]);max_all=max(max_all,c[i])
        # Track completed-step extrema. No clipping at any RK stage.
        if (step+1)%output_every==0 or step+1==steps:
            d=_diagnostic(c,nb,wall,diff)
            hist[hi,0]=(step+1)*dt
            for j in range(6): hist[hi,j+1]=d[j]
            hi+=1
            min_all=min(min_all,d[4]);max_all=max(max_all,d[5])
        if si<5 and step+1>=round(si*steps/4):
            snaps[si]=c;si+=1
    return c,hist[:hi],snaps,min_all,max_all

def simulate(n=96,protocol='center',kappa=.003,T=40.,tau=4.,a=.45,sigma=.50,
             aspect=1.,kind='dipole',angle=0.,speed=1.,dt_scale=1.,power=1.,phase=0.,save=None):
    start=time.perf_counter()
    g=make_grid(n); modes=flow_modes(g,a,sigma,aspect)
    G,identity_error=gram_matrix(a,sigma,aspect)
    ts=np.linspace(0,tau,129)
    cv=coefficients(protocol,ts,G,tau,speed,power,phase)
    # Sampled maximum over a complete blending period, including reverse peaks.
    # Step refinement and completed-step extrema are checked separately.
    vmax=float(max(np.max(np.abs(np.einsum('i,ikj->kj',v,modes['faces']))) for v in cv))
    dt=min(.16*g['h']**2/max(kappa,1e-30),.35*g['h']/max(vmax,1e-30),.02)*dt_scale
    if protocol in ['switch','reverse']: dt=min(dt,tau/100*dt_scale)
    # At least four steps are needed for all five quarter-time snapshots.
    steps=max(4,int(np.ceil(T/dt)));dt=T/steps
    t=np.arange(steps)*dt
    cs=coefficients(protocol,t,G,tau,speed,power,phase)
    cm=coefficients(protocol,t+.5*dt,G,tau,speed,power,phase)
    ce=coefficients(protocol,t+dt,G,tau,speed,power,phase)
    c=initial_scalar(g,kind,angle);c0=c.copy()
    wall=np.ascontiguousarray(g['x']**2+g['y']**2>.85**2)
    out_every=max(1,round(.05/dt))
    final,hist,snaps,minc,maxc=_integrate(c,g['neighbors'],modes['B'],wall,
        cs,cm,ce,dt,kappa/g['h']**2,out_every)
    V0=hist[0,2];J=float(hist[-1,2]/V0)
    target=hist[:,2]/V0<=.05
    budget=np.einsum('ti,ij,tj->t',cs,G,cs)
    budget_mid=np.einsum('ti,ij,tj->t',cm,G,cm)
    budget_end=np.einsum('ti,ij,tj->t',ce,G,ce)
    integrated_budget=float(dt*np.sum((budget+4*budget_mid+budget_end)/6))
    diss_integral=float(np.trapz(hist[:,4],hist[:,0]))
    summary=dict(n=n,cells=len(c),h=g['h'],area=g['area'],protocol=protocol,
        kappa=kappa,Pe=1/kappa if kappa>0 else None,T=T,tau=tau,a=a,sigma=sigma,aspect=aspect,kind=kind,
        angle=angle,speed=speed,phase=phase,dt_scale=dt_scale,power=power,dt=dt,steps=steps,J=J,
        t95=float(hist[np.argmax(target),0]) if np.any(target) else None,
        wall_J=float(hist[-1,3]/V0),wall_fraction=float(hist[-1,3]*np.mean(wall)/hist[-1,2]),
        mass_relative_drift=float((hist[-1,1]-hist[0,1])/hist[0,1]),
        variance_budget_relative_residual=float((V0-hist[-1,2]-diss_integral)/V0),
        min_c=float(minc),max_c=float(maxc),max_face_speed=vmax,
        max_cell_Pe=float(vmax*g['h']/kappa) if kappa else None,
        integrated_budget=integrated_budget,budget_min=float(budget.min()),budget_max=float(budget.max()),
        gram_identity_error=identity_error,
        analytic_J=float(np.exp(-2*kappa*ALPHA**2*T)) if protocol=='diffusion' and kind=='dipole' else None)
    if summary['analytic_J'] is not None:
        summary['analytic_J_error']=abs(J/summary['analytic_J']-1)
    summary['runtime_seconds']=time.perf_counter()-start
    import hashlib
    summary['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if kind.startswith('mix'):
        summary['initial_source_sha256']=hashlib.sha256(Path(__file__).with_name('review_initials.py').read_bytes()).hexdigest()
    snapshot_times=np.rint(np.arange(5)*steps/4)*dt
    result=dict(summary=summary,history=hist,snapshots=snaps,initial=c0,final=final,
                x=g['x'],y=g['y'],mask=g['mask'],gram=G,snapshot_times=snapshot_times)
    if save:
        path=Path(save);path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,**{k:v for k,v in result.items() if k!='summary'})
        path.with_suffix('.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--n',type=int,default=96);p.add_argument('--protocol',default='center')
    p.add_argument('--kappa',type=float,default=.003);p.add_argument('--T',type=float,default=40)
    p.add_argument('--tau',type=float,default=4);p.add_argument('--aspect',type=float,default=1)
    p.add_argument('--a',type=float,default=.45);p.add_argument('--sigma',type=float,default=.5)
    p.add_argument('--kind',default='dipole');p.add_argument('--angle',type=float,default=0)
    p.add_argument('--speed',type=float,default=1);p.add_argument('--dt-scale',type=float,default=1)
    p.add_argument('--phase',type=float,default=0.)
    p.add_argument('--save',default='work/data/trial.npz')
    args=p.parse_args();kwargs=vars(args);kwargs['dt_scale']=kwargs.pop('dt_scale')
    r=simulate(**kwargs);print(json.dumps(r['summary'],indent=2),flush=True)
