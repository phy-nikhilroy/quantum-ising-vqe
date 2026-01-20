"""
Exact Diagonalization Solver

Classical exact solution for benchmarking VQE results.
"""

import numpy as np
from scipy.sparse.linalg import eigsh
from typing import Tuple, List
from tqdm import tqdm

from hamiltonian import IsingHamiltonian


class ExactSolver:
    """
    Exact diagonalization solver for quantum Hamiltonians.
    """
    
    def __init__(self, hamiltonian: IsingHamiltonian):
        """
        Initialize exact solver.
        
        Args:
            hamiltonian: The Hamiltonian to diagonalize
        """
        self.hamiltonian = hamiltonian
        self.eigenvalues = None
        self.eigenvectors = None
        
    def solve(self, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find the k lowest eigenvalues and eigenvectors.
        
        Args:
            k: Number of eigenvalues to compute
            
        Returns:
            tuple: (eigenvalues, eigenvectors)
        """
        H_matrix = self.hamiltonian.get_matrix()
        
        # For small systems, use full diagonalization
        if self.hamiltonian.n_qubits <= 10:
            eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)
            self.eigenvalues = eigenvalues[:k]
            self.eigenvectors = eigenvectors[:, :k]
        else:
            # For larger systems, use sparse methods
            self.eigenvalues, self.eigenvectors = eigsh(
                H_matrix, k=k, which='SA'
            )
        
        return self.eigenvalues, self.eigenvectors
    
    def ground_state_energy(self) -> float:
        """
        Get ground state energy.
        
        Returns:
            float: Lowest eigenvalue
        """
        if self.eigenvalues is None:
            self.solve(k=1)
        return self.eigenvalues[0]
    
    def ground_state(self) -> np.ndarray:
        """
        Get ground state vector.
        
        Returns:
            np.ndarray: Ground state eigenvector
        """
        if self.eigenvectors is None:
            self.solve(k=1)
        return self.eigenvectors[:, 0]
    
    def energy_gap(self) -> float:
        """
        Compute energy gap between ground and first excited state.
        
        Returns:
            float: Energy gap
        """
        if self.eigenvalues is None or len(self.eigenvalues) < 2:
            self.solve(k=2)
        return self.eigenvalues[1] - self.eigenvalues[0]


def compute_exact_energies(
    n_qubits: int,
    J: float,
    h_values: List[float]
) -> np.ndarray:
    """
    Compute exact ground state energies for multiple field strengths.
    
    Args:
        n_qubits: Number of qubits
        J: Coupling strength
        h_values: List of transverse field strengths
        
    Returns:
        np.ndarray: Array of ground state energies
    """
    energies = []
    
    for h in tqdm(h_values, desc="Exact Solver Progress"):
        hamiltonian = IsingHamiltonian(n_qubits, J, h)
        solver = ExactSolver(hamiltonian)
        energy = solver.ground_state_energy()
        energies.append(energy)
    
    return np.array(energies)


def analyze_phase_transition(
    n_qubits: int,
    J: float,
    h_values: List[float]
) -> dict:
    """
    Analyze quantum phase transition characteristics.
    
    Args:
        n_qubits: Number of qubits
        J: Coupling strength
        h_values: Transverse field strengths
        
    Returns:
        dict: Phase transition analysis
    """
    energies = []
    gaps = []
    
    for h in h_values:
        hamiltonian = IsingHamiltonian(n_qubits, J, h)
        solver = ExactSolver(hamiltonian)
        solver.solve(k=2)
        
        energies.append(solver.ground_state_energy())
        gaps.append(solver.energy_gap())
    
    # Find minimum gap (critical point)
    gaps = np.array(gaps)
    critical_idx = np.argmin(gaps)
    
    return {
        'h_values': np.array(h_values),
        'energies': np.array(energies),
        'gaps': gaps,
        'critical_h': h_values[critical_idx],
        'min_gap': gaps[critical_idx]
    }


if __name__ == "__main__":
    # Example usage
    print("Exact Solver Example: 4-qubit Ising model\n")
    
    n_qubits = 4
    J = 1.0
    h = 1.0
    
    # Single point calculation
    hamiltonian = IsingHamiltonian(n_qubits, J, h)
    solver = ExactSolver(hamiltonian)
    
    print(f"Problem: {n_qubits} qubits, J={J}, h={h}")
    print(f"Hilbert space dimension: {2**n_qubits}\n")
    
    eigenvalues, _ = solver.solve(k=5)
    
    print("Lowest 5 eigenvalues:")
    for i, E in enumerate(eigenvalues):
        print(f"  E_{i}: {E:.6f}")
    
    print(f"\nGround state energy: {solver.ground_state_energy():.6f}")
    print(f"Energy gap: {solver.energy_gap():.6f}")
    
    # Phase transition analysis
    print("\n" + "="*50)
    print("Phase Transition Analysis")
    print("="*50)
    
    h_values = np.linspace(0.1, 2.0, 10)
    analysis = analyze_phase_transition(n_qubits, J, h_values)
    
    print(f"\nCritical field: h_c ≈ {analysis['critical_h']:.3f}")
    print(f"Minimum gap: Δ_min = {analysis['min_gap']:.6f}")