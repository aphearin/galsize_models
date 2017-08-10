"""
"""
import numpy as np
from astropy.cosmology import Planck15
from halotools.empirical_models import halo_mass_to_virial_velocity, halo_mass_to_halo_radius


__all__ = ('cvir_to_nfw_conc', 'halo_radius_at_mpeak')


def cvir_to_nfw_conc(cvir):
    return np.interp(cvir, [1, 1.5], [4, 20])


def halo_vvir_mpeak(mpeak, scale_factor_mpeak, cosmology=Planck15):

    total_mass = mpeak
    redshift = 1./scale_factor_mpeak - 1.
    return halo_mass_to_virial_velocity(total_mass, cosmology, redshift, 'vir')


def halo_radius_at_mpeak(mpeak, scale_factor_mpeak, cosmology=Planck15):
    """
    Parameters
    ----------
    mpeak : float or ndarray
        Float or Numpy array of shape (num_halos, ) storing peak halo mass in units of Msun,
        assuming little-h equal to ``cosmology.h``.

    scale_factor_mpeak : float or ndarray
        Float or Numpy array of shape (num_halos, ) storing the scale factor
        where halos attain Mpeak.

    cosmology : Astropy.cosmology object, optional
        Default is ``Planck15``

    Returns
    -------
    halo_radius_comoving : ndarray
        Numpy array of shape (num_halos, ) storing halo radius at the time of peak mass,
        in units of comoving kpc assuming little-h equal to ``cosmology.h``.

    Examples
    --------
    >>> num_halos = 100
    >>> mpeak = np.logspace(10, 15, num_halos)
    >>> scale_factor_mpeak = 1.0
    >>> radius = halo_radius_at_mpeak(mpeak, scale_factor_mpeak)
    """

    total_mass_unity_h = mpeak*cosmology.h
    redshift = 1./scale_factor_mpeak - 1.
    halo_radius_comoving_unity_h = halo_mass_to_halo_radius(
            total_mass_unity_h, cosmology, redshift, 'vir')
    halo_radius_comoving = halo_radius_comoving_unity_h/cosmology.h
    return halo_radius_comoving*1000.


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
