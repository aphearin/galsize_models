"""
"""
import numpy as np

__all__ = ('component_size_vs_rhalo', 'galaxy_size_vs_rhalo')


def component_size_vs_rhalo(rhalo, normalization, alpha, R0=1., scatter=0.):
    """
    """
    mean_size = normalization*(rhalo/R0)**alpha
    return 10**np.random.normal(loc=np.log10(mean_size), scale=scatter)


def galaxy_size_vs_rhalo(rhalo, bt, norm1, norm2, alpha1, alpha2, R0=1., scatter=0.):
    size1 = component_size_vs_rhalo(rhalo, norm1, alpha1, R0, scatter)
    size2 = component_size_vs_rhalo(rhalo, norm2, alpha2, R0, scatter)
    return bt*size1 + (1-bt)*size2


