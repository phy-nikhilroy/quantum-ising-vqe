"""
Quantum VQE Ising Simulation Package
"""

from .hamiltonian import IsingHamiltonian, construct_hamiltonian
from .vqe import VQEAnsatz, VQESolver, run_vqe_simulation
from .exact_solver import ExactSolver, compute_exact_energies
from .plots import plot_energy_comparison, plot_convergence

__version__ = '1.0.0'
__author__ = 'Your Name'

__all__ = [
    'IsingHamiltonian',
    'construct_hamiltonian',
    'VQEAnsatz',
    'VQESolver',
    'run_vqe_simulation',
    'ExactSolver',
    'compute_exact_energies',
    'plot_energy_comparison',
    'plot_convergence'
]