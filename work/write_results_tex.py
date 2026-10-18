import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parent
def load(tag): return json.loads((ROOT/'data'/f'{tag}.json').read_text())
def threshold(tag):
    s=load(tag);h=np.load(ROOT/'data'/f'{tag}.npz')['history'];v=h[:,2]/h[0,2]
    ix=np.flatnonzero(v<=.05)
    if len(ix)==0:return None,None
    i=int(ix[0]);t=float(np.interp(.05,[v[i],v[i-1]],[h[i,0],h[i-1,0]]))
    if s['protocol']=='reverse':energy=t+s['tau']/(4*np.pi)*np.sin(4*np.pi*t/s['tau'])
    else:energy=t*s['integrated_budget']/s['T']
    return t,energy
def sci(x):
    if x==0:return '0'
    p=int(np.floor(np.log10(abs(x))));a=x/10**p
    return rf'\ensuremath{{{a:.2f}\times10^{{{p}}}}}'
def make():
    B,S,C=load('main_pair'),load('main_switch_4'),load('main_center')
    radialB,radialS=load('initial_radial_pair'),load('initial_radial_switch')
    main=[json.loads(p.read_text()) for p in (ROOT/'data').glob('main_*.json')]
    macros=dict(JPair=f"{B['J']:.3g}",JSwitch=f"{S['J']:.3g}",JCenter=f"{C['J']:.3g}",
        PairGain=f"{100*(1-B['J']/C['J']):.1f}",PairSwitchGain=f"{100*(1-B['J']/S['J']):.1f}",
        RadialGain=f"{100*(1-radialS['J']/radialB['J']):.1f}",
        MassMax=sci(max(abs(x['mass_relative_drift']) for x in main)),
        BudgetMax=sci(max(abs(x['variance_budget_relative_residual']) for x in main)),
        InviscidLoss=sci(1-load('inviscid_switch_192')['J']),TimeChange=r'\ensuremath{10^{-9}}')
    (ROOT/'results_macros.tex').write_text('\n'.join(chr(92)+'newcommand{'+chr(92)+k+'}{'+v+'}' for k,v in macros.items()),encoding='utf-8')
    lines=[r'\begin{table}[H]\centering\small',r'\caption{网格加密与独立圆域解析基准，$\kappa=0.003,T=40$}\label{tab:convergence}',
        r'\begin{tabular}{rrrrr}\toprule',r'$N$ & 纯扩散 $J_h$ & 解析 $J$ 相对误差/\% & B的$J$ & S的$J$\\\midrule']
    for n in [64,96,128,192,256,384]:
        ds=load(f'conv_diffusion_{n}' if n!=192 else 'main_diffusion')
        b=load(f'conv_pair_{n}' if n!=192 else 'main_pair')
        s=load(f'conv_switch_{n}' if n!=192 else 'main_switch_4')
        lines.append(f"{n} & {ds['J']:.7f} & {ds['analytic_J_error']*100:.4f} & {b['J']:.7f} & {s['J']:.7f}"+r'\\')
    lines += [r'\bottomrule\end{tabular}\end{table}']
    (ROOT/'convergence_table.tex').write_text('\n'.join(lines),encoding='utf-8')
    lines=[r'\begin{table}[H]\centering\small',r'\caption{主协议终态指标及达标耗散；壁带为$r>0.85$，$t_{95}$作线性插值}\label{tab:main}',
        r'\begin{tabular}{lrrrrr}\toprule',r'协议 & $J$ & $t_{95}$ & $F_w/\%$ & $\mathcal E(T)$ & $\mathcal E_{95}$\\\midrule']
    for label,tag in [('C：同心','main_center'),('E：偏心','main_eccentric'),('B：双区','main_pair'),('S：切换，$\tau=4$','main_switch_4')]:
        s=load(tag);t,e=threshold(tag);tt=f'{t:.1f}' if t is not None else '$>40$';ee=f'{e:.1f}' if e is not None else '--'
        lines.append(f"{label} & {s['J']:.6f} & {tt} & {s['wall_fraction']*100:.1f} & {s['integrated_budget']:.2f} & {ee}"+r'\\')
    lines.append(r'\midrule')
    for label,tag in [('D：纯扩散','main_diffusion'),('M：平均场','main_mean_switch'),('R：累计等预算反转','reverse_fine')]:
        s=load(tag);t,e=threshold(tag);tt=f'{t:.1f}' if t is not None else '$>40$';ee=f'{e:.1f}' if e is not None else '--'
        lines.append(f"{label} & {s['J']:.6f} & {tt} & {s['wall_fraction']*100:.1f} & {s['integrated_budget']:.2f} & {ee}"+r'\\')
    lines.append(r'\bottomrule\end{tabular}\end{table}')
    (ROOT/'main_table.tex').write_text('\n'.join(lines),encoding='utf-8')
    lines=[r'\begin{table}[H]\centering\small',r'\caption{不同初值下的归一化终态方差，粗体为每行所列协议中的最小值}\label{tab:initial}',
        r'\begin{tabular}{lrrrr}\toprule',r'初值 & C & E & B & S，$\tau=4$\\\midrule']
    for label,prefix in [('水平模态','main'),('对角模态','initial_diag'),('竖直模态','initial_vert'),('局部斑块','initial_blob'),('径向分布','initial_radial')]:
        row=[]
        for proto in ['center','eccentric','pair','switch']:
            tag='main_switch_4' if prefix=='main' and proto=='switch' else prefix+'_'+proto
            row.append(load(tag)['J'])
        minimum=min(row)
        vals=[(r'\textbf{'+f'{v:.7f}'+'}') if v==minimum else f'{v:.7f}' for v in row]
        lines.append(label+' & '+' & '.join(vals)+r'\\')
    lines.append(r'\bottomrule\end{tabular}\end{table}')
    (ROOT/'initial_table.tex').write_text('\n'.join(lines),encoding='utf-8')
    lines=[r'\begin{table}[H]\centering\small',
        r'\caption{偏心距离与包络宽度敏感性：水平初值、相同连续耗散预算}\label{tab:geometry}',
        r'\begin{tabular}{rrrrr}\toprule',r'$a$ & $\sigma$ & C的$J$ & B的$J$ & S的$J$\\\midrule']
    for a in [.3,.6]:
        for sigma in [.4,.65]:
            vals=[load(f'geometry_{a:g}_{sigma:g}_{p}')['J'] for p in ['center','pair','switch']]
            row=[f'{v:.6f}' for v in vals];row[1]=r'\textbf{'+row[1]+'}'
            lines.append(f'{a:.2f} & {sigma:.2f} & '+' & '.join(row)+r'\\')
    lines.append(r'\bottomrule\end{tabular}\end{table}')
    (ROOT/'geometry_table.tex').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(macros,ensure_ascii=False,indent=2))
if __name__=='__main__': make()
