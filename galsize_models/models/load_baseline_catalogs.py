"""
"""
import numpy as np
from halotools.empirical_models import Moster13SmHm
from halotools.sim_manager import CachedHaloCatalog
from umachine_pyio.load_mock import load_mock_from_binaries, value_added_mock
from halotools.empirical_models import halo_mass_to_halo_radius
from astropy.cosmology import Planck15

from .new_haloprops import halo_radius_at_mpeak
from .random_bt_assignment import value_add_random_bt

from ..measurements import load_umachine_sdss_with_meert15


__all__ = ('load_moster13_mock', 'load_umachine_mock')

default_umachine_galprops = list((
    'sm', 'sfr', 'obs_sm', 'obs_sfr', 'icl', 'halo_id', 'upid',
    'x', 'y', 'z', 'vx', 'vy', 'vz', 'rvir', 'mvir', 'mpeak',
    'a_first_infall', 'dvmax_zscore', 'vmax_at_mpeak'))


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


def load_umachine_mock(galprops=default_umachine_galprops, Lbox=250):
    """
    """
    subvolumes = np.arange(144)
    mock = value_added_mock(load_mock_from_binaries(subvolumes, galprops=galprops), Lbox=Lbox)

    mock['ssfr'] = np.log10(mock['sfr']/mock['sm'])
    mock['is_main_sequence'] = mock['ssfr'] >= -10.75
    mock['is_green_valley'] = (mock['ssfr'] < -10.75) & (mock['ssfr'] >= -11.25)
    mock['is_quenched'] = (mock['ssfr'] < -11.25)
    mock['logsm'] = np.log10(mock['obs_sm'])

    redshift = 1./mock['a_first_infall'] - 1.
    mock['rvir_halo_kpc'] = halo_mass_to_halo_radius(mock['mpeak']*Planck15.h,
            Planck15, redshift, 'vir')*1000./Planck15.h

    mock['rvir_halo_kpc_present_day'] = halo_mass_to_halo_radius(mock['mvir']*Planck15.h,
            Planck15, 0., 'vir')*1000./Planck15.h

    spin_at_infall_fname = '/Users/aphearin/work/UniverseMachine/temp_galsize_models/spin_at_infall_umachine.npy'
    mock['spin_at_infall'] = np.load(spin_at_infall_fname)
    nonzero_spin_mask = mock['spin_at_infall'] != 0
    num_impute = np.count_nonzero(~nonzero_spin_mask)
    mock['spin_at_infall'][~nonzero_spin_mask] = np.random.choice(
                np.copy(mock['spin_at_infall'][nonzero_spin_mask].data), num_impute, replace=True)

    full_sdss, behroozi_complete, good_profile_mask = load_umachine_sdss_with_meert15()

    mask = good_profile_mask & behroozi_complete
    sdss = full_sdss[mask]

    return value_add_random_bt(mock, sdss)
