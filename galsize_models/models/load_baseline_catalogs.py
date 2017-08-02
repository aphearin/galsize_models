"""
"""
import numpy as np
from halotools.empirical_models import Moster13SmHm
from halotools.sim_manager import CachedHaloCatalog


__all__ = ('load_moster13_mock', )


def load_moster13_mock(logmstar_cut=9.75, simname='bolplanck', redshift=0.):
    """
    Examples
    --------
    >>> mock = load_moster13_mock(logmstar_cut=10)
    """

    halocat = CachedHaloCatalog(simname=simname, redshift=redshift)
    mpeak = halocat.halo_table['halo_mpeak']
    redshift = 1./halocat.halo_table['halo_scale_factor_mpeak'] - 1.

    mstar_model = Moster13SmHm()
    halocat.halo_table['mstar'] = mstar_model.mc_stellar_mass(
            prim_haloprop=mpeak, redshift=redshift)

    mstar_mask = halocat.halo_table['mstar'] > 10**logmstar_cut
    return halocat.halo_table[mstar_mask]
