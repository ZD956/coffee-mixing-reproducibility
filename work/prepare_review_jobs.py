import json
from pathlib import Path
from review_initials import SEEDS,MODES
root=Path(__file__).parent
holdout=[]
for seed in SEEDS:
    for p in ['center','eccentric','pair','switch']:
        holdout.append(dict(tag=f'holdout_{seed}_{p}',n=192,kind=f'mix{seed}',protocol=p,T=40.,tau=4.))
checks=[]
for kind in ['radial','blob']:
    for p in ['center','pair','switch']:
        checks.append(dict(tag=f'refine_{kind}_{p}_256',n=256,kind=kind,protocol=p,T=40.,tau=4.))
checks.append(dict(tag='refine_center_256',n=256,protocol='center',T=40.))
for tau in [.125,.25,.5]:
    checks.append(dict(tag=f'fast192_switch_{tau:g}',n=192,protocol='switch',tau=tau,T=40.))
checks.append(dict(tag='fast192_mean',n=192,protocol='mean_switch',T=40.))
for stage,jobs in [('holdout',holdout),('reviewchecks',checks)]:
    (root/f'{stage}_jobs.json').write_text(json.dumps(jobs,indent=2))
(root/'review_preregistered.json').write_text(json.dumps(dict(seeds=SEEDS,modes=MODES,
    weights='independent standard normal / eigenvalue wavenumber alpha',
    normalization='subtract discrete mean then scale maximum absolute fluctuation to0.45',
    protocols=['center','eccentric','pair','switch'],fixed_switch_tau=4.,
    primary_endpoint='J=V(40)/V(0)',all_runs_reported=True),indent=2))
print('Pre-specified jobs:',len(holdout),len(checks),flush=True)
