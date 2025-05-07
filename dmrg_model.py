import numpy as np
from scipy import sparse, integrate
from scipy.sparse.linalg import eigsh
import time
import warnings

# Placeholder imports (! TeNPy must be installed)
from tenpy.models import TFIModel
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg


class DMRG:
    """
    Density Matrix Renormalization Group (DMRG) simulator for 1D quantum systems.

    Supports both exact diagonalization (finite size) and analytical solutions (thermodynamic limit),
    as well as a numerical DMRG implementation.
    """

    def __init__(self, size, J=1.0, g=1.0, bound='finite', dmrg_params=None):
        self.size = size
        self.J = J
        self.g = g
        self.bound = bound
        self.dmrg_params = dmrg_params or self.default_dmrg_params()

        if self.bound == "infinite":
            raise NotImplementedError("Infinite DMRG is not supported in this implementation.")

    @staticmethod
    def default_dmrg_params():
        return {
            'mixer': True,
            'max_E_err': 1e-10,
            'trunc_params': {
                'chi_max': 1000,
                'svd_min': 1e-10,
            },
            'verbose': 1,
            'N_sweeps_check': 2,
        }

    @property
    def model_params(self):
        return {
            'L': self.size,
            'J': self.J,
            'g': self.g,
            'bc_MPS': self.bound,
        }

    def run_dmrg(self):
        """Run the DMRG algorithm and return results."""
        model = TFIModel(self.model_params)
        product_state = ["up"] * model.lat.N_sites
        psi = MPS.from_product_state(model.lat.mps_sites(), product_state, bc=self.bound)
        start_time = time.time()
        info = dmrg.run(psi, model, self.dmrg_params)
        end_time = time.time()
        return info, end_time - start_time, psi

    @staticmethod
    def finite_gs_energy(L, J, g):
        """Calculate the ground state energy for a finite system via exact diagonalization."""
        if L > 20:
            warnings.warn("Exact diagonalization skipped for L > 20.")
            return np.nan

        sx = sparse.csr_matrix(np.array([[0., 1.], [1., 0.]]))
        sz = sparse.csr_matrix(np.array([[1., 0.], [0., -1.]]))
        id = sparse.identity(2, format='csr')

        sx_list, sz_list = [], []
        for i in range(L):
            x_ops = [id] * L
            z_ops = [id] * L
            x_ops[i] = sx
            z_ops[i] = sz
            X = x_ops[0]
            Z = z_ops[0]
            for j in range(1, L):
                X = sparse.kron(X, x_ops[j], 'csr')
                Z = sparse.kron(Z, z_ops[j], 'csr')
            sx_list.append(X)
            sz_list.append(Z)

        H_xx = sparse.csr_matrix((2**L, 2**L))
        H_z = sparse.csr_matrix((2**L, 2**L))
        for i in range(L - 1):
            H_xx += sx_list[i] @ sx_list[i + 1]
        for i in range(L):
            H_z += sz_list[i]

        H = -J * H_xx - g * H_z
        E, _ = eigsh(H, k=1, which='SA', return_eigenvectors=True, ncv=20)
        return E[0]

    @staticmethod
    def infinite_gs_energy(J, g):
        """Calculate ground state energy density in the thermodynamic limit using analytical formula."""
        def integrand(k, lam):
            return np.sqrt(1 + lam**2 + 2 * lam * np.cos(k))

        lam = J / g
        result, _ = integrate.quad(integrand, -np.pi, np.pi, args=(lam,))
        return -g / (2 * np.pi) * result


if __name__ == "__main__":
    pass
