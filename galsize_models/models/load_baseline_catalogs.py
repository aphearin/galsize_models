"""
"""
import numpy as np
from halotools.empirical_models import Moster13SmHm
from halotools.sim_manager import CachedHaloCatalog
from umachine_pyio.load_mock import load_mock_from_binaries, value_added_mock
from .new_haloprops import halo_radius_at_mpeak


__all__ = ('load_moster13_mock', 'load_umachine_mock')


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
    mock = halocat.halo_table[mstar_mask]
    mock['comoving_radius_at_mpeak'] = halo_radius_at_mpeak(
            mock['halo_mpeak'], mock['halo_scale_factor_mpeak'])
    return mock


def load_umachine_mock():
    """
    """
    subvolumes = np.arange(144)
    galprops = list((
        'sm', 'sfr', 'obs_sm', 'obs_sfr', 'icl', 'halo_id', 'upid',
        'x', 'y', 'z', 'vx', 'vy', 'vz', 'rvir', 'mvir', 'mpeak',
        'a_first_infall', 'dvmax_zscore', 'vmax_at_mpeak'))
    return value_added_mock(load_mock_from_binaries(subvolumes, galprops=galprops), 250.)
