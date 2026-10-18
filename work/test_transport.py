"""Scientific invariants: run before parameter experiments."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'python_deps'))
import numpy as np
from mixing_solver import make_grid, flow_modes, gram_matrix, initial_scalar, rhs, simulate

def tests():
    g = make_grid(64)
    modes = flow_modes(g, a=.45, sigma=.50, aspect=1.)
    G, discrepancy = gram_matrix(.45, .50, 1., quadrature=100)
    coef = np.array([1., 0., 0.]) / np.sqrt(G[0,0])
    c = initial_scalar(g, 'dipole', 0.)
    div = np.sum(modes['B'], axis=1)
    assert np.max(np.abs(div)) < 1e-10, np.max(np.abs(div))
    z = rhs(c, g, modes, coef, 0.)
    assert abs(z.sum()) < 1e-9, z.sum()
    assert abs(np.dot(c, z)) < 1e-9, np.dot(c, z)
    assert np.max(np.abs(rhs(np.ones_like(c), g, modes, coef, .01))) < 1e-12
    z = rhs(c, g, modes, np.zeros(3), .01)
    assert np.dot(c, z) < 0
    assert discrepancy < 1e-8, discrepancy
    result = simulate(n=64, protocol='diffusion', kappa=.01, T=1., dt_scale=.6)
    assert abs(result['summary']['mass_relative_drift']) < 1e-11
    assert abs(result['summary']['analytic_J_error']) < .015
    assert result['summary']['min_c'] > 0
    print('PASS: discrete divergence, mass, skew advection, constant state, diffusion sign, continuum strain/enstrophy identity, disk Bessel diffusion benchmark.')

if __name__ == '__main__':
    tests()
