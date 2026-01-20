"""
Main execution script for VQE simulation.

This script runs the complete hybrid quantum-classical simulation
comparing VQE with exact diagonalization.
"""

import numpy as np
import os
from typing import Dict

from hamiltonian import IsingHamiltonian
from vqe import run_vqe_simulation
from exact_solver import compute_exact_energies, analyze_phase_transition
from plots import generate_summary_plot


def run_complete_simulation(
    n_qubits: int = 4,
    J: float = 1.0,
    h_min: float = 0.2,
    h_max: float = 2.0,
    n_points: int = 15,
    depth: int = 2,
    optimizer: str = 'COBYLA',
    max_iter: int = 200
) -> Dict:
    """
    Run complete VQE simulation and benchmarking.
    
    Args:
        n_qubits: Number of qubits
        J: Coupling strength
        h_min: Minimum transverse field
        h_max: Maximum transverse field
        n_points: Number of field strengths to sample
        depth: Ansatz circuit depth
        optimizer: VQE optimizer
        max_iter: Maximum iterations per VQE run
        
    Returns:
        dict: Complete simulation results
    """
    print("="*70)
    print("Hybrid Quantum-Classical Simulation of 1D Ising Model")
    print("="*70)
    print(f"\nSimulation Parameters:")
    print(f"  Qubits: {n_qubits}")
    print(f"  Coupling J: {J}")
    print(f"  Field range: [{h_min}, {h_max}]")
    print(f"  Samples: {n_points}")
    print(f"  Ansatz depth: {depth}")
    print(f"  Optimizer: {optimizer}")
    print(f"  Max iterations: {max_iter}")
    print("\n" + "="*70 + "\n")
    
    # Generate field values
    h_values = np.linspace(h_min, h_max, n_points)
    
    # Run VQE
    print("Running VQE simulations...")
    vqe_results = run_vqe_simulation(
        n_qubits=n_qubits,
        J=J,
        h_values=h_values,
        depth=depth,
        optimizer=optimizer,
        max_iter=max_iter
    )
    
    # Run exact solver
    print("\nComputing exact solutions...")
    exact_energies = compute_exact_energies(n_qubits, J, h_values)
    
    # Phase transition analysis
    print("\nAnalyzing phase transition...")
    phase_analysis = analyze_phase_transition(n_qubits, J, h_values)
    
    # Compile results
    results = {
        'n_qubits': n_qubits,
        'J': J,
        'h_values': h_values,
        'vqe_energies': vqe_results['energies'],
        'exact_energies': exact_energies,
        'convergence': vqe_results['convergence'],
        'phase_analysis': phase_analysis
    }
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    rel_errors = np.abs((results['vqe_energies'] - exact_energies) / exact_energies) * 100
    
    print(f"\nAccuracy Metrics:")
    print(f"  Mean relative error: {np.mean(rel_errors):.3f}%")
    print(f"  Max relative error: {np.max(rel_errors):.3f}%")
    print(f"  Min relative error: {np.min(rel_errors):.3f}%")
    
    print(f"\nPhase Transition:")
    print(f"  Critical field: h_c ≈ {phase_analysis['critical_h']:.3f}")
    print(f"  Minimum gap: {phase_analysis['min_gap']:.6f}")
    
    print(f"\nSample Energies (h = {h_values[len(h_values)//2]:.2f}):")
    idx = len(h_values) // 2
    print(f"  VQE:   {results['vqe_energies'][idx]:.6f}")
    print(f"  Exact: {exact_energies[idx]:.6f}")
    print(f"  Error: {rel_errors[idx]:.3f}%")
    
    print("\n" + "="*70)
    
    return results


def main():
    """
    Main entry point.
    """
    # Run simulation with default parameters
    results = run_complete_simulation(
        n_qubits=4,
        J=1.0,
        h_min=0.2,
        h_max=2.0,
        n_points=12,
        depth=2,
        optimizer='COBYLA',
        max_iter=150
    )
    
    # Generate visualizations
    print("\nGenerating plots...")
    generate_summary_plot(results, save_dir='results')
    
    print("\n✓ Simulation complete! Check the 'results' directory for outputs.")


if __name__ == "__main__":
    main()