"""
Unit tests for Hamiltonian construction.
"""

import pytest
import numpy as np
from src.hamiltonian import IsingHamiltonian


def test_hamiltonian_creation():
    """Test basic Hamiltonian creation."""
    n_qubits = 4
    J = 1.0
    h = 0.5
    
    hamiltonian = IsingHamiltonian(n_qubits, J, h)
    H = hamiltonian.build()
    
    assert len(H.paulis) == 2 * n_qubits - 1
    assert hamiltonian.n_qubits == n_qubits


def test_hamiltonian_hermitian():
    """Test that Hamiltonian is Hermitian."""
    hamiltonian = IsingHamiltonian(n_qubits=4, J=1.0, h=1.0)
    H_matrix = hamiltonian.get_matrix()
    
    assert np.allclose(H_matrix, H_matrix.conj().T)


def test_ground_state_energy_bounds():
    """Test that ground state energy is within expected bounds."""
    hamiltonian = IsingHamiltonian(n_qubits=4, J=1.0, h=1.0)
    H_matrix = hamiltonian.get_matrix()
    
    eigenvalues = np.linalg.eigvalsh(H_matrix)
    ground_energy = eigenvalues[0]
    
    # Energy should be negative for ferromagnetic coupling
    assert ground_energy < 0
    
    # Energy should be bounded by sum of all terms
    n = hamiltonian.n_qubits
    upper_bound = n * hamiltonian.h
    assert ground_energy > -upper_bound


def test_critical_point():
    """Test behavior near critical point h/J = 1."""
    J = 1.0
    h_values = [0.9, 1.0, 1.1]
    energies = []
    
    for h in h_values:
        hamiltonian = IsingHamiltonian(n_qubits=4, J=J, h=h)
        H_matrix = hamiltonian.get_matrix()
        ground_energy = np.min(np.linalg.eigvalsh(H_matrix))
        energies.append(ground_energy)
    
    # Energy should be continuous
    assert all(np.isfinite(energies))


if __name__ == "__main__":
    pytest.main([__file__])