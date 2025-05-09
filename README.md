# Quantum Many-Body Simulation: DMRG vs Neural Quantum States

This project explores and compares two modern approaches to solving the quantum many-body problem:

- **Density Matrix Renormalization Group (DMRG)** using [TeNPy](https://github.com/tenpy/tenpy)
- **Neural Quantum States (NQS)** based on Restricted Boltzmann Machines, implemented with [NetKet](https://github.com/netket/netket)

We focus on the 1D transverse field Ising model, a prototypical model for quantum phase transitions. The project includes code for running DMRG simulations, training NQS with Variational Monte Carlo (VMC), and analyzing quantities such as ground state energy, magnetization, and entanglement entropy.

---

## Features

- DMRG simulation pipeline using Matrix Product States (MPS)
- NQS-VMC pipeline using RBM wavefunctions
- Calculation of:
  - Ground state energy
  - Transverse magnetization ⟨σᶻ⟩
  - Entanglement entropy
- Visualization of phase transition behavior across different transverse field values

---

## Installation

### Dependencies

- Python 3.8+
- NumPy
- Matplotlib
- Pandas
- [TeNPy](https://tenpy.readthedocs.io/en/latest/)
- [NetKet](https://www.netket.org/)

To install dependencies:

```bash
pip install numpy matplotlib pandas
pip install git+https://github.com/tenpy/tenpy.git
pip install netket

___

## Usage

To re-run QIC_TFI_NNQSvsDMRG.ipynb, make sure all related scripts and data files are in the same folder. 
