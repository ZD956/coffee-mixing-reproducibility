import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'python_deps'))
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import expm_multiply
from mixing_solver import *

def matrix_reference():
    g=make_grid(64);m=flow_modes(g);G,_=gram_matrix();coef=coefficients('pair',[0.],G)[0]
    c0=initial_scalar(g);nc=len(c0);diff=.003/g['h']**2
    rows=[];cols=[];values=[]
    for i in range(nc):
        diag=0.
        for k in range(4):
            j=g['neighbors'][k,i]
            rate=diff+np.dot(coef,m['B'][:,k,i])
            rows.append(i);cols.append(j);values.append(rate)
            diag-=rate
        rows.append(i);cols.append(i);values.append(diag)
    A=coo_matrix((values,(rows,cols)),shape=(nc,nc)).tocsr()
    exact=expm_multiply(A*40,c0)
    run=simulate(n=64,protocol='pair',T=40.)
    error=np.linalg.norm(run['final']-exact)/np.linalg.norm(exact-exact.mean())
    return dict(relative_field_error_vs_matrix_exponential=error,matrix_column_sum=float(np.max(abs(np.asarray(A.sum(axis=0))))),
        matrix_row_sum=float(np.max(abs(np.asarray(A.sum(axis=1))))))

def discrete_flow_budgets():
    out=[]
    for n in [80,128,192,256,384]:
        g=make_grid(n);m=flow_modes(g);G,e=gram_matrix();h=g['h'];mask=g['mask']
        ux=[];uy=[];vx=[];vy=[]
        for mode in range(3):
            U=np.zeros((n,n));V=U.copy()
            U[mask]=(m['faces'][mode,0]-m['faces'][mode,1])/2
            V[mask]=(m['faces'][mode,2]-m['faces'][mode,3])/2
            dxu,dyu=np.gradient(U,h,edge_order=2);dxv,dyv=np.gradient(V,h,edge_order=2)
            ux.append(dxu[mask]);uy.append(dyu[mask]);vx.append(dxv[mask]);vy.append(dyv[mask])
        ux,uy,vx,vy=map(np.asarray,[ux,uy,vx,vy]);om=vx-uy;sh=uy+vx
        Go=om@om.T/mask.sum();Gs=(2*ux@ux.T+2*vy@vy.T+sh@sh.T)/mask.sum()
        row=dict(n=n,area=g['area'],continuous_gram_identity=e)
        for p in ['center','pair','switch']:
            c=coefficients(p,np.linspace(0,4,257),G,tau=4)
            po=np.einsum('ti,ij,tj->t',c,Go,c);ps=np.einsum('ti,ij,tj->t',c,Gs,c)
            row[p+'_enstrophy_min']=float(po.min());row[p+'_enstrophy_max']=float(po.max())
            row[p+'_strain_min']=float(ps.min());row[p+'_strain_max']=float(ps.max())
            row[p+'_strain_enstrophy_gap']=float(np.max(abs(po-ps)))
        out.append(row)
    return out

if __name__=='__main__':
    report=dict(matrix_exponential=matrix_reference(),discrete_budgets=discrete_flow_budgets())
    Path(__file__).with_name('independent_checks.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
