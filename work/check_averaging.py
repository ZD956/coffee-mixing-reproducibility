"""Independent read-only audit of saved fast-switch fields and Gram averages."""
import json
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent


def main():
    reference = np.load(BASE / 'data/fast_mean.npz')
    mean_meta = json.loads((BASE / 'data/fast_mean.json').read_text())
    theta0 = reference['initial'] - reference['initial'].mean()
    denominator = np.linalg.norm(theta0)
    rows = []
    for tau in [.125, .25, .5, 1., 2., 4., 8.]:
        path = BASE / f'data/fast_switch_{tau:g}.npz'
        run = np.load(path)
        meta = json.loads(path.with_suffix('.json').read_text())
        for key in ['initial', 'x', 'y', 'mask', 'gram']:
            assert np.array_equal(run[key], reference[key]), key
        for key in ['n', 'kappa', 'T', 'a', 'sigma', 'aspect', 'kind', 'angle', 'speed']:
            assert meta[key] == mean_meta[key], key
        error = np.linalg.norm(run['final'] - reference['final']) / denominator
        rows.append(dict(tau=tau, field_error=float(error), error_over_tau=float(error/tau),
                         J=meta['J'], J_minus_mean=meta['J']-mean_meta['J'],
                         dt=meta['dt'], steps=meta['steps'], budget=meta['integrated_budget'],
                         initial_fluctuation_rms=float(np.sqrt(np.mean(theta0**2)))))
    fits = {}
    for count in [3, 4, 5]:
        x = np.log([r['tau'] for r in rows[:count]])
        y = np.log([r['field_error'] for r in rows[:count]])
        slope, intercept = np.polyfit(x, y, 1)
        residual = y - (slope*x+intercept)
        fits[f'first_{count}'] = dict(exponent=float(slope), coefficient=float(np.exp(intercept)),
                                     log_R2=float(1-np.sum(residual**2)/np.sum((y-y.mean())**2)))
    G = reference['gram']
    phase = np.arange(262144)/262144
    s = .5*(1+np.sin(2*np.pi*phase))
    a = np.column_stack([np.zeros_like(s), s, 1-s])
    a /= np.sqrt(np.einsum('ti,ij,tj->t', a, G, a))[:, None]
    avg = a.mean(axis=0)
    deviations = a - avg
    average_power = np.mean(np.einsum('ti,ij,tj->t', a, G, a))
    mean_power = avg@G@avg
    fluctuation_power = np.mean(np.einsum('ti,ij,tj->t', deviations, G, deviations))
    results = dict(mean_J=mean_meta['J'], rows=rows, fits=fits,
                   averaging=dict(coefficients=avg.tolist(), average_power=float(average_power),
                                  mean_power=float(mean_power), fluctuation_power=float(fluctuation_power),
                                  jensen_residual=float(average_power-mean_power-fluctuation_power)))
    print(json.dumps(results, indent=2))
    (BASE/'averaging_check_results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
