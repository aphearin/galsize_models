"""
"""
import numpy as np
from halotools.empirical_models import halo_mass_to_halo_radius
from halotools.empirical_models import Moster13SmHm
from astropy.cosmology import Planck15


__all__ = ('halo_radius_vs_stellar_mass', 'galaxy_radius_vs_stellar_mass_linear',
        'galaxy_radius_vs_stellar_mass_power_law')


def halo_radius_vs_stellar_mass(stellar_mass_planck15,
            cosmology=Planck15, redshift=0, mdef='vir', **kwargs):
    """
    Parameters
    ----------
    stellar_mass_planck15 : float or ndarray
        Float or Numpy array of shape (num_gals, ) storing stellar mass in linear units
        of Msun assuming Planck+15 cosmology for little-h.

    Returns
    -------
    rhalo_comoving : float or ndarray
        Float or Numpy array of shape (num_gals, ) storing halo radius in
        comoving kpc units.

    Examples
    --------
    >>> stellar_mass_planck15 = np.logspace(9.5, 11.5, 25)
    >>> halo_radius = halo_radius_vs_stellar_mass(stellar_mass_planck15)

    """
    stellar_mass_unity_h = cosmology.h*cosmology.h*stellar_mass_planck15

    smhm_model = Moster13SmHm()
    #  Update Moster13 param_dict, if applicable
    for key in kwargs.keys():
        if key in list(smhm_model.param_dict.keys()):
            smhm_model.param_dict[key] = kwargs[key]

    mhalo_unity_h_table = np.logspace(9, 15, 1000)
    sm_unity_h_table = smhm_model.mean_stellar_mass(
            prim_haloprop=mhalo_unity_h_table, redshift=redshift)

    mhalo_unity_h = 10**np.interp(np.log10(stellar_mass_unity_h),
            np.log10(sm_unity_h_table), np.log10(mhalo_unity_h_table))
    rhalo_unity_h_comoving = halo_mass_to_halo_radius(mhalo_unity_h, Planck15, redshift, mdef)

    rhalo_comoving_planck15 = rhalo_unity_h_comoving/cosmology.h

    return rhalo_comoving_planck15*1000.


def galaxy_radius_vs_halo_radius_power_law(rhalo_physical_planck15, normalization, index):
    """
    """
    return normalization*(rhalo_physical_planck15**index)


def galaxy_radius_vs_stellar_mass_linear(stellar_mass_planck15, normalization=0.015,
            cosmology=Planck15, redshift=0, mdef='vir', **kwargs):
    """
    """
    rhalo = halo_radius_vs_stellar_mass(stellar_mass_planck15,
            cosmology=Planck15, redshift=redshift, mdef=mdef, **kwargs)

    index = 1
    return galaxy_radius_vs_halo_radius_power_law(rhalo, normalization, index)


def galaxy_radius_vs_stellar_mass_power_law(stellar_mass_planck15, normalization, index,
            cosmology=Planck15, redshift=0, mdef='vir', **kwargs):
    """
    """
    rhalo = halo_radius_vs_stellar_mass(stellar_mass_planck15,
            cosmology=Planck15, redshift=redshift, mdef=mdef, **kwargs)
    return galaxy_radius_vs_halo_radius_power_law(rhalo, normalization, index)


