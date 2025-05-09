# -*- coding: utf-8 -*-
"""
Created on Thu May  8 23:13:04 2025

@author: petka
"""

import netket as nk
import jax
import jax.numpy as jnp
from flax import nnx
import flax.linen as nn
import netket.nn as nknn
import time
from Ising import Ising

class FFN(nnx.Module):
    '''
    Neural Network Quantum State (NNQS)
    alpha - density of internal layer
    '''
    def __init__(self, N:int, alpha:int=1, *, rngs:nnx.Rngs):
        self.alpha = alpha
        self.linear = nnx.Linear(in_features=N, out_features=N*alpha, rngs=rngs)
    def __call__(self, x:jax.Array):
        y = self.linear(x)
        y = nnx.relu(y)
        return jnp.sum(y, axis=-1)
    
class SNN(nnx.Module):
    '''
    Neural Network Quantum State (NNQS) with implemented translational symmetry
    '''
    def __init__(self, N:int, alpha: int = 1, *, rngs:nnx.Rngs):
        self.alpha = alpha
        dense_symm_linen = nknn.DenseSymm(
            symmetries = nk.graph.Chain(length=N, pbc=True).translation_group()
            , features = alpha
            , kernel_init = nn.initializers.normal(stddev=0.01)
        )
        self.linear_symm = nnx.bridge.ToNNX(dense_symm_linen,  rngs=rngs).lazy_init(jnp.ones([1,1,N]))

    def __call__(self, x: jax.Array):
            x = x.reshape(-1, 1, x.shape[-1])
            x = self.linear_symm(x)
            x = nnx.relu(x)
            return jnp.sum(x, axis=(-1, -2))
        
        
def NNQS_run(chain:Ising, nn_vstate, n_iter:int, learning_rate = .1, diag_shift = .1):
    #implements run of NNNQS, returns log and runtime
    nn_start_time = time.time()
    optimizer = nk.optimizer.Sgd(learning_rate = learning_rate)
    gs = nk.driver.VMC(
        chain.H,
        optimizer,
        variational_state=nn_vstate,
        preconditioner=nk.optimizer.SR(diag_shift = diag_shift)
    )
    log = nk.logging.RuntimeLog()
    gs.run(n_iter = n_iter, out=log)

    return log, time.time() - nn_start_time