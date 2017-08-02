"""
"""
import numpy as np
from halotools.empirical_models import halo_mass_to_halo_radius
from astropy.cosmology import Planck15


__all__ = ('galsize_vs_rvir', 'kravtsov13')


def galsize_vs_rvir(r200c, normalization=0.015, scatter_in_dex=0.2):
    """
    Parameters
    ----------
    r200c : float or ndarray
        Float or ndarray of shape (ngals, ) storing the halo radius,
        given in physical coordinates with units of kpc,
        where halo radius is defined as 200 times the critical density.

    normalization : float, optional
        Coefficient controlling the linear scaling between ``r200c`` of the halo
        and half-mass radius of the galaxy. Default value is 0.015, as in Kravtsov (2013).

    scatter_in_dex : float, optional
        Level of log-normal scatter between halo and galaxy size.
        Default value is 0.2 dex, as in Kravtsov (2013).

    Returns
    -------
    galsize : float or ndarray
        Float or ndarray of shape (ngals, ) storing the half-mass radius of the
        model galaxy in units of kpc.

    Examples
    --------
    >>> ngals = 500
    >>> r200c = np.logspace(0.1, 2, ngals)
    >>> galsize = galsize_vs_rvir(r200c)
    """
    return 10**np.random.normal(loc=np.log10(normalization*r200c), scale=scatter_in_dex)


def kravtsov13(halo_mass_in_Msun, cosmology=Planck15, redshift=0, **kwargs):
    """
    Examples
    --------
    >>> halo_mass_in_Msun = 10**np.random.uniform(11, 15, 100)
    >>> rhalf = kravtsov13(halo_mass_in_Msun)
    """
    halo_mass_in_Msun_by_h = halo_mass_in_Msun*cosmology.h
    mdef = '200c'
    r200c_in_Mpc_by_h = halo_mass_to_halo_radius(halo_mass_in_Msun_by_h, cosmology, redshift, mdef)
    r200c_in_kpc = 1000.*r200c_in_Mpc_by_h/cosmology.h

    rhalf_mstar = galsize_vs_rvir(r200c_in_kpc, **kwargs)
    return rhalf_mstar
