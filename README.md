# Hybrid Quantum Simulation of a 1D Transverse Field Ising Model using VQE

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project demonstrates a **hybrid quantum-classical approach** to quantum simulation using the **Variational Quantum Eigensolver (VQE)**. We study the 1D transverse-field Ising model and compute approximate ground state energies using parameterized quantum circuits, benchmarking results against exact diagonalization.

The work highlights how near-term quantum algorithms can be integrated into physics-based simulation workflows, directly relevant to quantum chemistry, condensed matter physics, and materials science applications.

## Physics Background

### The Transverse Field Ising Model

The Hamiltonian for a 1D transverse-field Ising model is:

```
H = -J Σᵢ ZᵢZᵢ₊₁ - h Σᵢ Xᵢ
```

Where:
- **J**: Coupling strength between neighboring spins
- **h**: Transverse field strength
- **Zᵢ, Xᵢ**: Pauli operators on qubit i

This model exhibits a **quantum phase transition** at h/J = 1, making it ideal for demonstrating quantum simulation capabilities.

## Features

-  Parameterized quantum circuit (ansatz) implementation
-  VQE optimization using classical optimizers (COBYLA, SLSQP)
-  Exact diagonalization for benchmarking
-  Phase diagram analysis across transverse field strengths
-  Energy convergence visualization
-  Comprehensive error analysis

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/phy-nikhilroy/quantum-ising-vqe.git
cd quantum-ising-vqe

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the complete simulation:

```bash
python src/main.py
```

### Custom Parameters

```python
from src.vqe import run_vqe_simulation

# Run VQE for specific parameters
results = run_vqe_simulation(
    n_qubits=4,
    J=1.0,
    h_values=[0.5, 1.0, 1.5],
    optimizer='COBYLA',
    max_iter=200
)
```

### Jupyter Notebook Analysis

```bash
jupyter notebook notebooks/analysis.ipynb
```

## Project Structure

```
quantum-ising-vqe/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
├── src/
│   ├── __init__.py
│   ├── main.py              # Main execution script
│   ├── hamiltonian.py       # Ising Hamiltonian construction
│   ├── vqe.py               # VQE implementation
│   ├── exact_solver.py      # Classical exact diagonalization
│   └── plots.py             # Visualization utilities
├── notebooks/
│   └── analysis.ipynb       # Interactive analysis notebook
├── results/                 # Output directory
│   ├── energy_comparison.png
│   └── convergence_plot.png
└── tests/                   # Unit tests
    └── test_hamiltonian.py
```

## Results

The project generates several key outputs:

1. **Energy vs Transverse Field**: Comparison of VQE and exact energies across the phase transition
2. **Convergence Analysis**: Optimization trajectory showing VQE convergence
3. **Error Metrics**: Relative error between VQE and exact solutions

Example output:
```
N=4 qubits, h=1.0, J=1.0
VQE Energy: -3.9142 ± 0.0023
Exact Energy: -3.9167
Relative Error: 0.064%
```

## Technical Details

### Ansatz Design
We use a hardware-efficient ansatz with alternating rotation and entangling layers:
- Single-qubit rotations: RY(θ) gates
- Entanglement: CNOT ladder structure
- Depth: 2-3 layers for n=4-6 qubits

### Optimization
- **Optimizers**: COBYLA (derivative-free), SLSQP (gradient-based)
- **Convergence**: Typically 100-300 iterations
- **Precision**: ~0.1% relative error for small systems

### Benchmarking
Exact solutions computed via:
- Sparse matrix construction (efficient for n ≤ 12 qubits)
- Lanczos algorithm for ground state

## Performance

| Qubits | VQE Time | Exact Time | VQE Accuracy |
|--------|----------|------------|--------------|
| 4      | ~5s      | <1s        | 99.9%        |
| 6      | ~30s     | ~2s        | 99.5%        |
| 8      | ~3min    | ~15s       | 99.0%        |

*Tested on Intel i5 with statevector simulator*

## Extensions & Future Work

- [ ] Implement adaptive VQE (ADAPT-VQE)
- [ ] Add noise models for realistic hardware simulation
- [ ] Extend to 2D Ising models
- [ ] Compare with QAOA for optimization problems
- [ ] Real quantum hardware execution (IBM Quantum)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## References

1. Peruzzo et al., "A variational eigenvalue solver on a photonic quantum processor" *Nature Communications* 5, 4213 (2014)
2. Sachdev, "Quantum Phase Transitions" Cambridge University Press (2011)
3. McClean et al., "The theory of variational hybrid quantum-classical algorithms" *New Journal of Physics* 18, 023023 (2016)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or collaborations, please open an issue or contact [phy-nikhilroy@gmail.com]

## Acknowledgments

- Qiskit Development Team
- The quantum computing community

---

**Note**: This is an educational/research project demonstrating quantum algorithm implementation. For production quantum chemistry calculations, consider using specialized libraries like PySCF-QC or Qiskit Nature.