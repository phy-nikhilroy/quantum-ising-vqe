"""
Visualization utilities for VQE results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional
import os

# Set style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)
plt.rcParams['figure.figsize'] = (12, 8)


def plot_energy_comparison(
    h_values: np.ndarray,
    vqe_energies: np.ndarray,
    exact_energies: np.ndarray,
    n_qubits: int,
    J: float,
    save_path: Optional[str] = None
):
    """
    Plot VQE vs exact energies across transverse field strengths.
    
    Args:
        h_values: Transverse field strengths
        vqe_energies: VQE ground state energies
        exact_energies: Exact ground state energies
        n_qubits: Number of qubits
        J: Coupling strength
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Energy comparison
    ax1.plot(h_values, exact_energies, 'o-', label='Exact', 
             linewidth=2, markersize=8, color='#2E86AB')
    ax1.plot(h_values, vqe_energies, 's--', label='VQE', 
             linewidth=2, markersize=8, color='#A23B72')
    
    ax1.axvline(x=J, color='gray', linestyle=':', linewidth=2, 
                label=f'Critical point (h/J=1)')
    
    ax1.set_xlabel('Transverse Field (h/J)', fontsize=14)
    ax1.set_ylabel('Ground State Energy', fontsize=14)
    ax1.set_title(f'VQE vs Exact Diagonalization ({n_qubits} qubits, J={J})', 
                  fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Relative error
    rel_error = np.abs((vqe_energies - exact_energies) / exact_energies) * 100
    
    ax2.plot(h_values, rel_error, 'o-', linewidth=2, 
             markersize=8, color='#F18F01')
    ax2.axvline(x=J, color='gray', linestyle=':', linewidth=2)
    
    ax2.set_xlabel('Transverse Field (h/J)', fontsize=14)
    ax2.set_ylabel('Relative Error (%)', fontsize=14)
    ax2.set_title('VQE Accuracy', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()


def plot_convergence(
    convergence_data: list,
    h_values: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Plot VQE convergence for different field strengths.
    
    Args:
        convergence_data: List of energy histories
        h_values: Corresponding field strengths
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(h_values)))
    
    for i, (history, h) in enumerate(zip(convergence_data, h_values)):
        ax.plot(history, label=f'h={h:.2f}', 
                linewidth=2, color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Iteration', fontsize=14)
    ax.set_ylabel('Energy', fontsize=14)
    ax.set_title('VQE Convergence', fontsize=16, fontweight='bold')
    ax.legend(fontsize=10, ncol=2, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved convergence plot to {save_path}")
    
    plt.show()


def plot_phase_diagram(
    h_values: np.ndarray,
    energies: np.ndarray,
    gaps: Optional[np.ndarray] = None,
    critical_h: Optional[float] = None,
    save_path: Optional[str] = None
):
    """
    Plot phase diagram showing energy and gap.
    
    Args:
        h_values: Transverse field strengths
        energies: Ground state energies
        gaps: Energy gaps (optional)
        critical_h: Critical field strength (optional)
        save_path: Path to save figure
    """
    if gaps is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    else:
        fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Energy per site
    ax1.plot(h_values, energies, 'o-', linewidth=2, 
             markersize=8, color='#2E86AB')
    
    if critical_h:
        ax1.axvline(x=critical_h, color='red', linestyle='--', 
                   linewidth=2, label=f'Critical h ≈ {critical_h:.2f}')
        ax1.legend(fontsize=12)
    
    ax1.set_xlabel('Transverse Field (h/J)', fontsize=14)
    ax1.set_ylabel('Ground State Energy', fontsize=14)
    ax1.set_title('Phase Diagram', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    if gaps is not None:
        ax2.plot(h_values, gaps, 'o-', linewidth=2, 
                markersize=8, color='#A23B72')
        
        if critical_h:
            ax2.axvline(x=critical_h, color='red', linestyle='--', linewidth=2)
        
        ax2.set_xlabel('Transverse Field (h/J)', fontsize=14)
        ax2.set_ylabel('Energy Gap', fontsize=14)
        ax2.set_title('Excitation Gap', fontsize=16, fontweight='bold')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved phase diagram to {save_path}")
    
    plt.show()


def generate_summary_plot(
    results: Dict,
    save_dir: str = 'results'
):
    """
    Generate comprehensive summary visualization.
    
    Args:
        results: Dictionary containing all simulation results
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Main comparison plot
    plot_energy_comparison(
        results['h_values'],
        results['vqe_energies'],
        results['exact_energies'],
        results['n_qubits'],
        results['J'],
        save_path=os.path.join(save_dir, 'energy_comparison.png')
    )
    
    # Convergence plot
    if 'convergence' in results:
        plot_convergence(
            results['convergence'],
            results['h_values'],
            save_path=os.path.join(save_dir, 'convergence_plot.png')
        )
    
    print(f"\nAll plots saved to {save_dir}/")