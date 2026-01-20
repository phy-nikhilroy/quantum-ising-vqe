"""
Ising Hamiltonian Construction Module

This module constructs the transverse-field Ising model Hamiltonian
for quantum simulation using VQE.
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from typing import List, Tuple


class IsingHamiltonian:
    """
    Constructs and manages the 1D transverse-field Ising Hamiltonian.
    
    H = -J Σᵢ ZᵢZᵢ₊₁ - h Σᵢ Xᵢ
    
    Attributes:
        n_qubits (int): Number of qubits/spins in the chain
        J (float): Coupling strength (default: 1.0)
        h (float): Transverse field strength
    """
    
    def __init__(self, n_qubits: int, J: float = 1.0, h: float = 1.0):
        """
        Initialize the Ising Hamiltonian.
        
        Args:
            n_qubits: Number of spins in the chain
            J: Nearest-neighbor coupling strength
            h: Transverse field strength
        """
        self.n_qubits = n_qubits
        self.J = J
        self.h = h
        self._hamiltonian = None
        
    def build(self) -> SparsePauliOp:
        """
        Construct the Hamiltonian as a SparsePauliOp.
        
        Returns:
            SparsePauliOp: The Ising Hamiltonian
        """
        pauli_list = []
        coeffs = []
        
        # ZZ interaction terms (nearest-neighbor)
        for i in range(self.n_qubits - 1):
            pauli_str = ['I'] * self.n_qubits
            pauli_str[i] = 'Z'
            pauli_str[i + 1] = 'Z'
            pauli_list.append(''.join(pauli_str))
            coeffs.append(-self.J)
        
        # X transverse field terms
        for i in range(self.n_qubits):
            pauli_str = ['I'] * self.n_qubits
            pauli_str[i] = 'X'
            pauli_list.append(''.join(pauli_str))
            coeffs.append(-self.h)
        
        self._hamiltonian = SparsePauliOp(pauli_list, coeffs)
        return self._hamiltonian
    
    def get_matrix(self) -> np.ndarray:
        """
        Get the full matrix representation of the Hamiltonian.
        
        Returns:
            np.ndarray: Dense Hamiltonian matrix
        """
        if self._hamiltonian is None:
            self.build()
        return self._hamiltonian.to_matrix()
    
    def expected_value(self, state_vector: np.ndarray) -> float:
        """
        Compute the expected energy for a given state.
        
        Args:
            state_vector: Quantum state vector
            
        Returns:
            float: Expected energy <ψ|H|ψ>
        """
        if self._hamiltonian is None:
            self.build()
        
        H_matrix = self.get_matrix()
        return np.real(np.conj(state_vector) @ H_matrix @ state_vector)
    
    def __repr__(self) -> str:
        return f"IsingHamiltonian(n_qubits={self.n_qubits}, J={self.J}, h={self.h})"


def construct_hamiltonian(n_qubits: int, J: float, h: float) -> SparsePauliOp:
    """
    Convenience function to construct an Ising Hamiltonian.
    
    Args:
        n_qubits: Number of qubits
        J: Coupling strength
        h: Transverse field strength
        
    Returns:
        SparsePauliOp: The Hamiltonian operator
    """
    ising = IsingHamiltonian(n_qubits, J, h)
    return ising.build()


if __name__ == "__main__":
    # Example usage
    print("Constructing 4-qubit Ising Hamiltonian...")
    hamiltonian = IsingHamiltonian(n_qubits=4, J=1.0, h=0.5)
    H = hamiltonian.build()
    
    print(f"\nHamiltonian: {hamiltonian}")
    print(f"Number of terms: {len(H.paulis)}")
    print(f"\nPauli terms:")
    for pauli, coeff in zip(H.paulis, H.coeffs):
        print(f"  {coeff:+.2f} * {pauli}")
    
    print(f"\nMatrix shape: {hamiltonian.get_matrix().shape}")
    print(f"Matrix is Hermitian: {np.allclose(hamiltonian.get_matrix(), hamiltonian.get_matrix().conj().T)}")