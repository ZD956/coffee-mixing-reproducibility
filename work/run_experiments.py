import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import json,time,argparse,hashlib,inspect
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from mixing_solver import simulate

BASE=Path(__file__).parent

def run_one(job):
    tag=job.pop('tag')
    force=job.pop('_force',False)
    dest=BASE/'data'/f'{tag}.npz'
    if dest.with_suffix('.json').exists() and not force:
        cached=json.loads(dest.with_suffix('.json').read_text())
        if not dest.exists():
            raise ValueError(f'Missing cached array for {tag}; use --force')
        source_hash=hashlib.sha256((BASE/'mixing_solver.py').read_bytes()).hexdigest()
        if cached.get('source_sha256')!=source_hash:
            raise ValueError(f'Cache source missing or different for {tag}; use --force')
        if str(job.get('kind','')).startswith('mix'):
            initial_hash=hashlib.sha256((BASE/'review_initials.py').read_bytes()).hexdigest()
            if cached.get('initial_source_sha256')!=initial_hash:
                raise ValueError(f'Initial-condition source changed for {tag}; use --force')
        for key,param in inspect.signature(simulate).parameters.items():
            if key=='save':continue
            value=job.get(key,param.default)
            actual=cached.get(key)
            if actual!=value:
                raise ValueError(f'Cached parameter mismatch for {tag}: {key}; use --force')
        return tag,cached
    result=simulate(**job,save=dest)
    return tag,result['summary']

def execute(jobs,workers=4):
    t=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(run_one,dict(job)):job for job in jobs}
        for f in as_completed(futures):
            tag,s=f.result()
            print(f"{tag}: J={s['J']:.8f}, wall={s['wall_fraction']:.3f}, mass={s['mass_relative_drift']:.2g}, Peh={s['max_cell_Pe']}, sec={s['runtime_seconds']:.1f}",flush=True)
    print(f'Completed {len(jobs)} jobs in {time.perf_counter()-t:.1f}s',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',default='pilot');p.add_argument('--workers',type=int,default=4)
    p.add_argument('--force',action='store_true',help='Recompute and replace cached outputs for selected jobs.')
    args=p.parse_args();jobs=[]
    if args.stage=='pilot':
        for proto in ['diffusion','center','eccentric','pair','reverse']:
            jobs.append(dict(tag='pilot_'+proto,n=80,protocol=proto,T=40.,tau=8.))
        for tau in [.5,1.,2.,4.,8.,16.,32.,80.]:
            jobs.append(dict(tag=f'pilot_switch_{tau:g}',n=80,protocol='switch',T=40.,tau=tau))
    else:
        jobs=json.loads((BASE/f'{args.stage}_jobs.json').read_text(encoding='utf-8'))
    if args.force:
        for job in jobs: job['_force']=True
    execute(jobs,args.workers)
