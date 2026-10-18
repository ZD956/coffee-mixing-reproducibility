"""Pre-specified synthetic initial conditions for the submission re-review."""
import numpy as np

SEEDS=list(range(20261015,20261027))
MODES=[(0,1,'cos'),(0,2,'cos'),(1,1,'cos'),(1,1,'sin'),
       (1,2,'cos'),(1,2,'sin'),(2,1,'cos'),(2,1,'sin'),(3,1,'cos'),(3,1,'sin')]

def mixture(g,seed):
    from scipy.special import jv,jnp_zeros
    r=np.sqrt(g['x']**2+g['y']**2)
    theta=np.arctan2(g['y'],g['x'])
    weights=np.random.default_rng(seed).standard_normal(len(MODES))
    f=np.zeros_like(r)
    for w,(m,n,angular) in zip(weights,MODES):
        alpha=float(jnp_zeros(m,n)[-1])
        norm=abs(jv(m,alpha))*np.sqrt(1. if m==0 else .5*(1-m*m/(alpha*alpha)))
        direction=np.cos(m*theta) if angular=='cos' else np.sin(m*theta)
        f+=(w/alpha)*jv(m,alpha*r)*direction/norm
    f-=f.mean()
    return .5+.45*f/np.max(np.abs(f))
