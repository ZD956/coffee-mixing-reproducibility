import json
from pathlib import Path
root=Path(__file__).parent

main=[]
for proto in ['diffusion','center','eccentric','pair','mean_switch','reverse']:
    main.append(dict(tag=f'main_{proto}',n=192,protocol=proto,T=40.,tau=8.))
for tau in [.5,1.,2.,4.,8.,10.,20.,40.,80.]:
    main.append(dict(tag=f'main_switch_{tau:g}',n=192,protocol='switch',T=40.,tau=tau))
(root/'main_jobs.json').write_text(json.dumps(main,indent=2))

verify=[]
for n in [64,96,128,256,384]:
    for proto in ['diffusion','pair','switch']:
        verify.append(dict(tag=f'conv_{proto}_{n}',n=n,protocol=proto,T=40.,tau=4.))
for ds in [.5,.25]:
    verify.append(dict(tag=f'time_switch_{ds:g}',n=192,protocol='switch',T=40.,tau=4.,dt_scale=ds))
for n in [96,192]:
    verify.append(dict(tag=f'inviscid_switch_{n}',n=n,protocol='switch',T=40.,tau=4.,kappa=0.))
(root/'verify_jobs.json').write_text(json.dumps(verify,indent=2))

robust=[]
for aspect in [.5,1.,2.]:
    for proto in ['center','eccentric','pair','switch','mean_switch']:
        robust.append(dict(tag=f'shape_{aspect:g}_{proto}',n=192,aspect=aspect,protocol=proto,T=40.,tau=4.))
for kind,angle,name in [('dipole',.7853981633974483,'diag'),('dipole',1.5707963267948966,'vert'),('blob',0.,'blob'),('radial',0.,'radial')]:
    for proto in ['center','eccentric','pair','switch','mean_switch']:
        robust.append(dict(tag=f'initial_{name}_{proto}',n=192,kind=kind,angle=angle,protocol=proto,T=40.,tau=4.))
for kappa in [.0015,.006,.012]:
    for proto in ['diffusion','center','pair','switch','mean_switch']:
        robust.append(dict(tag=f'diff_{kappa:g}_{proto}',n=384 if kappa==.0015 else 192,kappa=kappa,protocol=proto,T=40.,tau=4.))
for speed in [.5,2.]:
    for proto in ['center','pair','switch']:
        robust.append(dict(tag=f'speed_{speed:g}_{proto}',n=384 if speed==2. else 192,speed=speed,protocol=proto,T=40.,tau=4.))
(root/'robust_jobs.json').write_text(json.dumps(robust,indent=2))
print('Jobs',len(main),len(verify),len(robust))
