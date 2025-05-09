# -*- coding: utf-8 -*-
"""
Created on Thu May  8 23:02:44 2025

@author: petka
"""

import netket as nk
from netket.operator.spin import sigmax, sigmaz
from scipy.sparse.linalg import eigsh
import time

class Ising():
    '''
    This class implements Ising model: it contains Hamiltonian and Hilbert space
    method find_exact_gs finds exact ground state energy if system is not too big
    '''
    def __init__(self, N:int, l:float, periodic:bool, spin:float=1/2):
        #physical constants
        self.N = N
        self.periodic = periodic
        self.l = l
        self.exact_max_size = 20 #maximum size of exactly solvable system

        #Hilberst space & Hamiltonian
        self.hi = nk.hilbert.Spin(s= spin, N=N)
        self.H = -sum([l*sigmax(self.hi, i) for i in range(N)])
        self.H += -sum([sigmaz(self.hi, i)*sigmaz(self.hi, (i+1)) for i in range(N-1)]) # Finite
        if self.periodic and N!=2:
            self.H += -sigmaz(self.hi, 0)*sigmaz(self.hi, N-1) # Periodic term

    def find_exact_gs(self, full_out=False):
        if self.N <= self.exact_max_size:
            start_time = time.time()
            eig_vals, eig_vecs = eigsh(self.H.to_sparse(), k=1, which="SA")
            self.exact_gs = eig_vals[0]
            self.exact_gs_d = self.exact_gs/self.N #energy density
            self.exact_sol_time = time.time() - start_time
            if full_out:
                return eig_vals[0], eig_vecs[:,0]
            else:
                return eig_vals[0]
        else:
            self.exact_gs = None
            self.exact_gs_d = None #energy density
            self.exact_sol_time = 0
            print("System too big to make a sparse matrix")
            return None
