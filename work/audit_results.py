"""Audit saved experiments without modifying scientific results."""
import sys,json,hashlib,platform
from pathlib import Path
ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT/'python_deps'))
import numpy as np
import scipy,numba,matplotlib
from mixing_solver import gram_matrix,coefficients

def read(tag):
    return json.loads((ROOT/'data'/f'{tag}.json').read_text()),dict(np.load(ROOT/'data'/f'{tag}.npz'))

def audit():
    results=[]
    for f in sorted((ROOT/'data').glob('*.json')):
        tag=f.stem;s,d=read(tag)
        J=np.var(d['final'])/np.var(d['initial'])
        assert abs(J-s['J'])<1e-11,(tag,J,s['J'])
        assert abs(np.mean(d['final'])/np.mean(d['initial'])-1-s['mass_relative_drift'])<1e-11,tag
        assert len(d['history'])>=2 and np.isfinite(d['history']).all(),tag
        assert np.isfinite(d['final']).all(),tag
        assert abs(s['mass_relative_drift'])<1e-10,tag
        if s['kappa']>0:
            assert s['min_c']>=float(d['initial'].min())-1e-9,(tag,'minimum')
            assert s['max_c']<=float(d['initial'].max())+1e-9,(tag,'maximum')
            assert np.max(np.diff(d['history'][:,2]))<1e-10,(tag,'variance nonmonotone')
        if s['protocol'] not in ['diffusion','mean_switch']:
            assert abs(s['integrated_budget']-s['T']*s['speed']**2)<1e-8,(tag,'budget')
        results.append(dict(tag=tag,**s))
    for stage in ['main','verify','robust','extra','last','supplement','holdout','reviewchecks','geometry']:
        jobs=json.loads((ROOT/f'{stage}_jobs.json').read_text())
        assert all((ROOT/'data'/f"{j['tag']}.json").exists() for j in jobs),stage
    timebase,td=read('main_switch_4')
    time_errors=[]
    for tag in ['time_switch_0.5','time_switch_0.25']:
        s,d=read(tag)
        time_errors.append(abs(s['J']/timebase['J']-1))
    fast=[]
    for base,half in [('fast_switch_0.125','fast_switch_halfstep'),('fast_mean','fast_mean_halfstep')]:
        s,d=read(base);s2,d2=read(half)
        fast.append(np.linalg.norm(d['final']-d2['final'])/np.linalg.norm(d['initial']-np.mean(d['initial'])))
    radialC,_=read('initial_radial_center');radialD,_=read('initial_radial_diffusion')
    G,_=gram_matrix();w=coefficients('switch',np.arange(8192)/8192,G,tau=1);mean=w.mean(0)
    report=dict(records=len(results),nonpilot_records=sum(not s['tag'].startswith('pilot') for s in results),
        all_saved_J_verified=True,all_mass_drift_below_1e_10=True,
        all_positive_diffusion_completed_step_bounds_preserved=True,
        main_time_refinement_relative_J_errors=time_errors,
        fast_time_refinement_relative_field_errors=fast,
        radial_center_minus_diffusion_J=radialC['J']-radialD['J'],
        average_power=float(mean@G@mean),gram_condition_number=float(np.linalg.cond(G)),
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
            numba=numba.__version__,matplotlib=matplotlib.__version__,platform=platform.platform()),
        source_sha256=hashlib.sha256((ROOT/'mixing_solver.py').read_bytes()).hexdigest())
    (ROOT/'final_data_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    return report

if __name__=='__main__': audit()
