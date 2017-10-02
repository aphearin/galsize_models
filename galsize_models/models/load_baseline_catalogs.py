"""
"""
import numpy as np
from scipy.stats import norm
from halotools.empirical_models import Moster13SmHm
from halotools.sim_manager import CachedHaloCatalog
from umachine_pyio.load_mock import load_mock_from_binaries, value_added_mock
from halotools.empirical_models import halo_mass_to_halo_radius
from astropy.cosmology import Planck15
from astropy.utils.misc import NumpyRNGContext
from astropy.table import Table
from halotools.empirical_models import solve_for_polynomial_coefficients

from .random_bt_assignment import value_add_random_bt

from ..measurements import load_umachine_sdss_with_meert15


__all__ = ('moster13_based_mock', 'load_umachine_mock', 'load_baseline_halocat')

default_umachine_galprops = list((
    'sm', 'sfr', 'obs_sm', 'obs_sfr', 'icl', 'halo_id', 'upid',
    'x', 'y', 'z', 'vx', 'vy', 'vz', 'rvir', 'mvir', 'mpeak',
    'a_first_infall', 'dvmax_zscore', 'vmax_at_mpeak'))


moster13_halocat_keys = ('halo_upid', 'halo_mpeak', 'halo_scale_factor_mpeak',
        'halo_x', 'halo_y', 'halo_z', 'halo_zpeak', 'halo_vmax_mpeak', 'halo_mvir',
        'halo_scale_factor_firstacc', 'halo_mvir_firstacc', 'halo_halfmass_scale_factor',
        'halo_vx', 'halo_vy', 'halo_vz', 'halo_rvir_zpeak', 'halo_vmax_at_mpeak_percentile',
        'halo_mvir_host_halo', 'halo_spin', 'halo_uran')


def load_baseline_halocat(simname='bolplanck', redshift=0, fixed_seed=411):
    halocat = CachedHaloCatalog(simname=simname, redshift=redshift)

    halocat.halo_table['halo_zpeak'] = 1./halocat.halo_table['halo_scale_factor_mpeak'] - 1.

    rvir_peak_physical_unity_h = halo_mass_to_halo_radius(halocat.halo_table['halo_mpeak'],
                                halocat.cosmology, halocat.halo_table['halo_zpeak'], 'vir')
    rvir_peak_physical = rvir_peak_physical_unity_h/halocat.cosmology.h
    halocat.halo_table['halo_rvir_zpeak'] = rvir_peak_physical*1000.

    nhalos = len(halocat.halo_table)
    with NumpyRNGContext(fixed_seed):
        halocat.halo_table['halo_uran'] = np.random.rand(nhalos)

    mask = halocat.halo_table['halo_mvir_host_halo'] == 0
    halocat.halo_table['halo_mvir_host_halo'][mask] = halocat.halo_table['halo_mpeak'][mask]

    vmax_percentile_fname = '/Users/aphearin/work/UniverseMachine/temp_galsize_models/vmax_percentile.npy'
    halocat.halo_table['halo_vmax_at_mpeak_percentile'] = np.load(vmax_percentile_fname)
    return halocat


def moster13_based_mock(halocat=None, keys_to_keep=moster13_halocat_keys,
            scatter_ordinates=(0.4, 0.3, 0.5), mpeak_key='halo_mpeak',
            zpeak_key='halo_zpeak', **moster13_params):
    """
    """
    if halocat is None:
        halocat = load_baseline_halocat()
    else:
        rvir_peak_physical_unity_h = halo_mass_to_halo_radius(halocat.halo_table[mpeak_key],
                                    halocat.cosmology, halocat.halo_table[zpeak_key], 'vir')
        rvir_peak_physical = rvir_peak_physical_unity_h/halocat.cosmology.h
        halocat.halo_table['halo_rvir_zpeak'] = rvir_peak_physical*1000.


    model = Moster13SmHm()
    model.param_dict.update(moster13_params)

    mean_mstar_unity_h = model.mean_stellar_mass(
            prim_haloprop=halocat.halo_table[mpeak_key],
            redshift=halocat.halo_table[zpeak_key])

    mean_mstar = mean_mstar_unity_h/halocat.cosmology.h/halocat.cosmology.h
    mean_logmstar = np.log10(mean_mstar)

    try:
        uran = halocat.halo_table['halo_uran']
    except KeyError:
        uran = np.random.rand(len(halocat.halo_table))

    mc_logmstar = norm.isf(1-uran, loc=mean_logmstar,
            scale=model.param_dict[u'scatter_model_param1'])

    #  Apply M* correction to account for Meert+15 photometry differences
    scatter_abscissa = 9.5, 10.5, 11.75
    a0, a1, a2 = solve_for_polynomial_coefficients(scatter_abscissa, scatter_ordinates)
    logmstar_correction = a0 + a1*mc_logmstar + a2*mc_logmstar**2
    logmstar_correction = np.where(mc_logmstar < scatter_abscissa[0], scatter_ordinates[0], logmstar_correction)
    logmstar_correction = np.where(mc_logmstar > scatter_abscissa[2], scatter_ordinates[2], logmstar_correction)
    corrected_mc_logmstar = mc_logmstar + logmstar_correction

    mock = Table()
    mstar_mask = corrected_mc_logmstar > 9
    for key in keys_to_keep:
        mock[key[5:]] = halocat.halo_table[key][mstar_mask]

    mock['mstar'] = 10**corrected_mc_logmstar[mstar_mask]
    mock['mstar_moster13'] = 10**mc_logmstar[mstar_mask]
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
    raise ValueError("``rvir_halo_kpc`` is not computed correctly - fix this before using this mock")
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
