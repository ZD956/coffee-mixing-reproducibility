"""Additional code provenance and reproducibility checks; no archived data changed."""
import sys,json,importlib.util,tempfile,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).parent.resolve()
sys.path.insert(0,str(ROOT/'python_deps'))
import numpy as np
import mixing_solver as current
import run_experiments as runner

def main():
    spec=importlib.util.spec_from_file_location('legacy_transport',ROOT/'review_v1'/'mixing_solver.py')
    old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
    comparisons=[]
    for proto in ['center','switch']:
        kwargs=dict(n=48,protocol=proto,T=.2,tau=.1)
        a=old.simulate(**kwargs);b=current.simulate(**kwargs)
        delta=float(np.max(abs(a['final']-b['final'])))
        assert delta<1e-13,(proto,delta)
        assert np.max(abs(a['history']-b['history']))<1e-13
        assert b['summary']['dt_scale']==1.
        assert len(b['snapshot_times'])==5 and b['snapshot_times'][-1]==kwargs['T']
        comparisons.append(dict(protocol=proto,T=.2,n=48,max_field_difference=delta))
    # Check that short mixed-initial runs use the prescribed bounds and zero flux.
    mix=current.simulate(n=48,protocol='switch',T=.1,tau=.1,kind='mix20261015')
    assert mix['initial'].min()>=.05-1e-14 and mix['initial'].max()<=.95+1e-14
    assert abs(mix['summary']['mass_relative_drift'])<1e-12
    assert len(mix['summary']['initial_source_sha256'])==64
    short=current.simulate(n=32,T=.001)
    assert short['summary']['steps']==4
    assert np.isfinite(short['snapshots']).all()
    assert np.array_equal(short['snapshots'][-1],short['final'])
    assert np.all(np.diff(short['snapshot_times'])>0)
    # Isolated cache fixtures verify reuse and rejection, without touching data/.
    previous=runner.BASE
    with tempfile.TemporaryDirectory(prefix='mixing_cache_') as tmp:
        root=Path(tmp);(root/'data').mkdir()
        shutil.copy2(ROOT/'mixing_solver.py',root/'mixing_solver.py')
        runner.BASE=root
        job=dict(tag='cache_fixture',n=32,protocol='center',T=.02,dt_scale=.5)
        _,computed=runner.run_one(dict(job));_,cached=runner.run_one(dict(job))
        assert computed==cached
        rejects=[]
        for kind in ['dt_scale','omitted_nondefault','source','missing_array']:
            candidate=dict(job)
            if kind=='dt_scale':candidate['dt_scale']=.25
            if kind=='omitted_nondefault':candidate.pop('dt_scale')
            if kind=='source':(root/'mixing_solver.py').write_text('# changed\n')
            if kind=='missing_array':
                shutil.copy2(ROOT/'mixing_solver.py',root/'mixing_solver.py')
                (root/'data'/'cache_fixture.npz').unlink()
            try:runner.run_one(candidate)
            except ValueError:rejects.append(kind)
            else:raise AssertionError('Unsafe cache reuse: '+kind)
        runner.BASE=previous
    result=dict(default_transport_matches_preserved_version=comparisons,
        mixed_initial_mass_drift=mix['summary']['mass_relative_drift'],
        new_metadata_checked=True,very_short_run_snapshots_checked=True,
        matching_cache_reused=True,rejected_cache_changes=rejects,
        current_source_sha256=hashlib.sha256((ROOT/'mixing_solver.py').read_bytes()).hexdigest())
    (ROOT/'submission_checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
