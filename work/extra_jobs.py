import json
from pathlib import Path
jobs=[]
for tau in [.125,.25,.5,1.,2.,4.,8.]:
    jobs.append(dict(tag=f'fast_switch_{tau:g}',n=128,protocol='switch',tau=tau,T=40.))
jobs.append(dict(tag='fast_mean',n=128,protocol='mean_switch',T=40.))
for phase in [.25,.5,.75]:
    for tau in [4.,20.]:
        jobs.append(dict(tag=f'phase_{phase:g}_{tau:g}',n=192,protocol='switch',phase=phase,tau=tau,T=40.))
for aspect in [.5,2.]:
    for tau in [.5,2.,8.,20.]:
        jobs.append(dict(tag=f'map_{aspect:g}_{tau:g}',n=192,protocol='switch',aspect=aspect,tau=tau,T=40.))
jobs.append(dict(tag='reverse_fine',n=256,protocol='reverse',tau=8.,T=40.))
Path(__file__).with_name('extra_jobs.json').write_text(json.dumps(jobs,indent=2))
print('Extra jobs',len(jobs))
