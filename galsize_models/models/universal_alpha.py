"""
"""
import numpy as np
from .kravtsov13 import halo_radius_vs_stellar_mass

__all__ = ('galaxy_size_vs_mstar', 'component_size_vs_mstar')


def component_size_vs_mstar(total_mstar_planck15, normalization, alpha, R0, scatter=0.):
    """
    """
    rhalo = halo_radius_vs_stellar_mass(total_mstar_planck15)
    mean_size = normalization*(rhalo/R0)**alpha
    return 10**np.random.normal(loc=np.log10(mean_size), scale=scatter)


def galaxy_size_vs_mstar(total_mstar_planck15, bt, norm1, norm2, alpha1, alpha2, R0, scatter=0.):
    size1 = component_size_vs_mstar(total_mstar_planck15, norm1, alpha1, R0, scatter=scatter)
    size2 = component_size_vs_mstar(total_mstar_planck15, norm2, alpha2, R0, scatter=scatter)
    return bt*size1 + (1-bt)*size2


