"""
"""
import numpy as np
from astropy.cosmology import Planck15
from halotools.empirical_models import halo_mass_to_virial_velocity


__all__ = ('cvir_to_nfw_conc', )


def cvir_to_nfw_conc(cvir):
    return np.interp(cvir, [1, 1.5], [4, 20])


def halo_vvir_mpeak(mpeak, scale_factor_mpeak, cosmology=Planck15):

    total_mass = mpeak
    redshift = 1./scale_factor_mpeak - 1.
    return halo_mass_to_virial_velocity(total_mass, cosmology, redshift, 'vir')


def cvir_mpeak(mpeak, vmax_at_mpeak, scale_factor_mpeak, cosmology=Planck15):
    vvir_at_mpeak = halo_vvir_mpeak(mpeak, scale_factor_mpeak, cosmology=cosmology)
    return vmax_at_mpeak/vvir_at_mpeak


def nfw_conc_at_mpeak(mpeak, vmax_at_mpeak, scale_factor_mpeak, cosmology=Planck15):
    """
    Examples
    --------
    >>> nhalos = 1000
    >>> mpeak = np.logspace(11, 15, nhalos)
    >>> vmax_at_mpeak = np.logspace(2, 3, nhalos)
    >>> scale_factor_mpeak = np.random.uniform(0, 1, nhalos)
    >>> result = nfw_conc_at_mpeak(mpeak, vmax_at_mpeak, scale_factor_mpeak)
    """
    cvir_at_mpeak = cvir_mpeak(mpeak, vmax_at_mpeak, scale_factor_mpeak, cosmology=cosmology)
    return cvir_to_nfw_conc(cvir_at_mpeak)
