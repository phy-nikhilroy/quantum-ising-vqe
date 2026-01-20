"""
Variational Quantum Eigensolver (VQE) Implementation

This module implements the VQE algorithm for finding ground state
energies of the transverse-field Ising model.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional, Callable
from tqdm import tqdm
import time

from hamiltonian import IsingHamiltonian


class VQEAnsatz:
    """
    Hardware-efficient ansatz for VQE.
    
    Structure:
    - Layer of RY rotations
    - Layer of CNOT entanglement (linear chain)
    - Repeat for depth layers
    """
    
    def __init__(self, n_qubits: int, depth: int = 2):
        """
        Initialize the ansatz.
        
        Args:
            n_qubits: Number of qubits
            depth: Circuit depth (number of repetitions)
        """
        self.n_qubits = n_qubits
        self.depth = depth
        self.n_params = n_qubits * depth
        
    def build_circuit(self, parameters: Optional[np.ndarray] = None) -> QuantumCircuit:
        """
        Build the parameterized quantum circuit.
        
        Args:
            parameters: Circuit parameters (if None, use symbolic parameters)
            
        Returns:
            QuantumCircuit: The ansatz circuit
        """
        qc = QuantumCircuit(self.n_qubits)
        
        if parameters is None:
            params = ParameterVector('θ', self.n_params)
        else:
            params = parameters
        
        param_idx = 0
        
        for layer in range(self.depth):
            # Rotation layer
            for qubit in range(self.n_qubits):
                qc.ry(params[param_idx], qubit)
                param_idx += 1
            
            # Entanglement layer (skip on last layer)
            if layer < self.depth - 1:
                for qubit in range(self.n_qubits - 1):
                    qc.cx(qubit, qubit + 1)
        
        return qc


class VQESolver:
    """
    VQE solver for quantum ground state problems.
    """
    
    def __init__(self, hamiltonian: IsingHamiltonian, ansatz: VQEAnsatz):
        """
        Initialize VQE solver.
        
        Args:
            hamiltonian: The problem Hamiltonian
            ansatz: Parameterized quantum circuit
        """
        self.hamiltonian = hamiltonian
        self.ansatz = ansatz
        self.H_op = hamiltonian.build()
        
        # Track optimization history
        self.energy_history = []
        self.param_history = []
        
    def energy_evaluation(self, parameters: np.ndarray) -> float:
        """
        Evaluate energy for given circuit parameters.
        
        Args:
            parameters: Circuit parameters
            
        Returns:
            float: Expected energy
        """
        # Build circuit with parameters
        qc = self.ansatz.build_circuit(parameters)
        
        # Get statevector
        state = Statevector(qc)
        
        # Compute expectation value
        energy = state.expectation_value(self.H_op).real
        
        # Store history
        self.energy_history.append(energy)
        self.param_history.append(parameters.copy())
        
        return energy
    
    def optimize(
        self,
        initial_params: Optional[np.ndarray] = None,
        method: str = 'COBYLA',
        max_iter: int = 200,
        tol: float = 1e-6
    ) -> Dict:
        """
        Run VQE optimization.
        
        Args:
            initial_params: Starting parameters (random if None)
            method: Optimization method ('COBYLA', 'SLSQP', etc.)
            max_iter: Maximum iterations
            tol: Convergence tolerance
            
        Returns:
            dict: Optimization results
        """
        # Initialize parameters
        if initial_params is None:
            initial_params = np.random.uniform(
                -np.pi, np.pi, self.ansatz.n_params
            )
        
        # Reset history
        self.energy_history = []
        self.param_history = []
        
        print(f"Starting VQE optimization with {method}...")
        start_time = time.time()
        
        # Run optimization
        result = minimize(
            self.energy_evaluation,
            initial_params,
            method=method,
            options={'maxiter': max_iter, 'disp': False},
            tol=tol
        )
        
        elapsed_time = time.time() - start_time
        
        return {
            'energy': result.fun,
            'params': result.x,
            'success': result.success,
            'n_iterations': result.nfev,
            'time': elapsed_time,
            'history': {
                'energies': np.array(self.energy_history),
                'params': np.array(self.param_history)
            }
        }


def run_vqe_simulation(
    n_qubits: int,
    J: float,
    h_values: List[float],
    depth: int = 2,
    optimizer: str = 'COBYLA',
    max_iter: int = 200
) -> Dict:
    """
    Run VQE simulation across multiple field strengths.
    
    Args:
        n_qubits: Number of qubits
        J: Coupling strength
        h_values: List of transverse field strengths
        depth: Ansatz depth
        optimizer: Optimization method
        max_iter: Maximum iterations per optimization
        
    Returns:
        dict: Results for all field strengths
    """
    results = {
        'h_values': h_values,
        'energies': [],
        'params': [],
        'convergence': []
    }
    
    ansatz = VQEAnsatz(n_qubits, depth)
    
    for h in tqdm(h_values, desc="VQE Progress"):
        hamiltonian = IsingHamiltonian(n_qubits, J, h)
        solver = VQESolver(hamiltonian, ansatz)
        
        result = solver.optimize(method=optimizer, max_iter=max_iter)
        
        results['energies'].append(result['energy'])
        results['params'].append(result['params'])
        results['convergence'].append(result['history']['energies'])
    
    results['energies'] = np.array(results['energies'])
    
    return results


if __name__ == "__main__":
    # Example usage
    print("VQE Example: 4-qubit Ising model\n")
    
    n_qubits = 4
    J = 1.0
    h = 1.0
    
    # Create Hamiltonian and ansatz
    hamiltonian = IsingHamiltonian(n_qubits, J, h)
    ansatz = VQEAnsatz(n_qubits, depth=2)
    
    print(f"Problem: {n_qubits} qubits, J={J}, h={h}")
    print(f"Ansatz: {ansatz.n_params} parameters\n")
    
    # Run VQE
    solver = VQESolver(hamiltonian, ansatz)
    result = solver.optimize(method='COBYLA', max_iter=100)
    
    print(f"\nVQE Results:")
    print(f"  Ground state energy: {result['energy']:.6f}")
    print(f"  Iterations: {result['n_iterations']}")
    print(f"  Time: {result['time']:.2f}s")
    print(f"  Success: {result['success']}")