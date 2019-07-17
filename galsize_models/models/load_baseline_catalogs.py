"""
"""
import os
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
from AbundanceMatching import AbundanceFunction, calc_number_densities
from halotools.empirical_models import enforce_periodicity_of_box
from halotools.utils import crossmatch
from halotools.empirical_models import noisy_percentile

from .random_bt_assignment import value_add_random_bt

from ..measurements import load_umachine_sdss_with_meert15


__all__ = ('moster13_based_mock', 'load_umachine_mock', 'load_baseline_halocat',
        'load_orphan_mock', 'moustakas_sham', 'load_orphan_subhalos', 'orphan_selection',
        'random_orphan_selection')

default_umachine_galprops = list((
    'sm', 'sfr', 'obs_sm', 'obs_sfr', 'icl', 'halo_id', 'upid',
    'x', 'y', 'z', 'vx', 'vy', 'vz', 'rvir', 'mvir', 'mpeak',
    'a_first_infall', 'dvmax_zscore', 'vmax_at_mpeak'))


moster13_halocat_keys = ('halo_upid', 'halo_mpeak', 'halo_scale_factor_mpeak',
        'halo_x', 'halo_y', 'halo_z', 'halo_zpeak', 'halo_vmax_mpeak', 'halo_mvir',
        'halo_scale_factor_firstacc', 'halo_mvir_firstacc', 'halo_halfmass_scale_factor',
        'halo_vx', 'halo_vy', 'halo_vz', 'halo_rvir_zpeak', 'halo_vmax_at_mpeak_percentile',
        'halo_mvir_host_halo', 'halo_spin', 'halo_uran')

smf_dirname = "/Users/aphearin/work/repositories/c/universemachine/obs"
smf_basename = "moustakas_z0.01_z0.20.smf"
smf_fname = os.path.join(smf_dirname, smf_basename)
smf = np.loadtxt(smf_fname)
log10_sm_table_h0p7 = 0.5*(smf[:, 0] + smf[:, 1])
dn_dlog10_sm_h0p7 = smf[:, 2]
sham_ext_range = (8, 12.75)
moustakas_af = AbundanceFunction(log10_sm_table_h0p7, dn_dlog10_sm_h0p7,
        sham_ext_range, faint_end_first=True)


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


def load_orphan_mock():
    """ Note that this function calls a Moster+13 model and applies a cut on stellar mass.
    """

    dirname = "/Users/aphearin/work/sims/bolplanck/orphan_catalog_z0"
    basename = "cross_matched_orphan_catalog.hdf5"

    halo_table = Table.read(os.path.join(dirname, basename), path='data')

    halo_table['vmax_at_mpeak_percentile'] = np.load(
        os.path.join(dirname, 'vmax_percentile.npy'))

    halo_table['zpeak'] = 1./halo_table['mpeak_scale']-1.

    class HaloCatalog(object):
        def __init__(self, halo_table):
            self.halo_table = halo_table
            self.cosmology = Planck15
            for key in self.halo_table.keys():
                if key[:5] != 'halo_':
                    halo_table.rename_column(key, 'halo_'+key)

    halocat = HaloCatalog(halo_table)

    from galsize_models.models import moster13_based_mock

    keys_to_keep = list(halocat.halo_table.keys())
    keys_to_keep.append('halo_rvir_zpeak')

    mock = moster13_based_mock(halocat=halocat, mpeak_key='halo_mpeak', zpeak_key='halo_zpeak',
                              keys_to_keep=keys_to_keep)


    mock['noisy_vmax_at_mpeak_percentile'] = noisy_percentile(
        mock['vmax_at_mpeak_percentile'], 0.5)

    return mock


def load_orphan_subhalos():
    """
    """
    dirname = "/Users/aphearin/work/sims/bolplanck/orphan_catalog_z0"
    basename = "cross_matched_orphan_catalog.hdf5"

    halo_table = Table.read(os.path.join(dirname, basename), path='data')

    Lbox = 250.
    halo_table['x'] = enforce_periodicity_of_box(halo_table['x'], Lbox)
    halo_table['y'] = enforce_periodicity_of_box(halo_table['y'], Lbox)
    halo_table['z'] = enforce_periodicity_of_box(halo_table['z'], Lbox)

    halo_table['vmax_at_mpeak_percentile'] = np.load(
        os.path.join(dirname, 'vmax_percentile.npy'))
    halo_table['noisy_vmax_at_mpeak_percentile'] = noisy_percentile(
        halo_table['vmax_at_mpeak_percentile'], 0.5)

    halo_table['orphan_mass_loss_percentile'] = -1.
    halo_table['orphan_mass_loss_percentile'][halo_table['orphan']] = np.load(
            os.path.join(dirname, 'orphan_mass_loss_percentile.npy'))
    halo_table['orphan_vmax_at_mpeak_percentile'] = -1.
    halo_table['orphan_vmax_at_mpeak_percentile'][halo_table['orphan']] = np.load(
            os.path.join(dirname, 'orphan_vmax_at_mpeak_percentile.npy'))
    halo_table['orphan_vmax_loss_percentile'] = -1.
    halo_table['orphan_vmax_loss_percentile'][halo_table['orphan']] = np.load(
            os.path.join(dirname, 'orphan_vmax_loss_percentile.npy'))

    halo_table['orphan_fixed_mpeak_mhost_percentile'] = -1.
    halo_table['orphan_fixed_mpeak_mhost_percentile'][halo_table['orphan']] = np.load(
            os.path.join(dirname, 'orphan_fixed_mpeak_mhost_percentile.npy'))

    halo_table['zpeak'] = 1./halo_table['mpeak_scale']-1.

    halo_table['zpeak_no_splashback'] = 0.
    satmask = halo_table['upid'] != -1
    halo_table['zpeak_no_splashback'][satmask] = halo_table['zpeak'][satmask]

    rvir_peak_physical_unity_h = halo_mass_to_halo_radius(halo_table['mpeak'],
                                Planck15, halo_table['zpeak'], 'vir')
    rvir_peak_physical = rvir_peak_physical_unity_h/Planck15.h
    halo_table['rvir_zpeak'] = rvir_peak_physical*1000.

    rvir_peak_no_spl_physical_unity_h = halo_mass_to_halo_radius(halo_table['mpeak'],
                                Planck15, halo_table['zpeak_no_splashback'], 'vir')
    rvir_peak_no_spl_physical = rvir_peak_no_spl_physical_unity_h/Planck15.h
    halo_table['rvir_zpeak_no_splashback'] = rvir_peak_no_spl_physical*1000.

    halo_table['hostid'] = np.nan
    hostmask = halo_table['upid'] == -1
    halo_table['hostid'][hostmask] = halo_table['halo_id'][hostmask]
    halo_table['hostid'][~hostmask] = halo_table['upid'][~hostmask]

    idxA, idxB = crossmatch(halo_table['hostid'], halo_table['halo_id'])
    halo_table['host_mvir'] = np.nan
    halo_table['host_mvir'][idxA] = halo_table['mvir'][idxB]

    halo_table = halo_table[~np.isnan(halo_table['host_mvir'])]

    halo_table['frac_mpeak_remaining'] = halo_table['mvir']/halo_table['mpeak']
    halo_table['frac_vpeak_remaining'] = halo_table['vmax']/halo_table['vmax_at_mpeak']

    return halo_table


def moustakas_sham(sham_subhalo_property, scatter):
    """
    """
    Lbox_h0p7 = 250./0.7
    _remainder = moustakas_af.deconvolute(scatter, 20)
    return moustakas_af.match(calc_number_densities(sham_subhalo_property, Lbox_h0p7),
                scatter=scatter, do_add_scatter=True, do_rematch=True)


def orphan_selection(catalog, mpeak_abscissa=(11, 13), prob_select_ordinates=(0.5, 0.),
            selection_key='orphan_fixed_mpeak_mhost_percentile'):
    num_subhalos = len(catalog)
    selection_indices = np.ones(num_subhalos).astype(bool)

    eligible_orphan_mask = catalog['orphan']
    orphans = catalog[eligible_orphan_mask]
    prob_select = np.interp(np.log10(orphans['mpeak']), mpeak_abscissa, prob_select_ordinates)
    selected_orphan_mask = orphans[selection_key] > 1-prob_select
    selection_indices[eligible_orphan_mask] = selected_orphan_mask
    return selection_indices


def random_orphan_selection(catalog, num_to_select):
    """
    """
    num_subhalos = len(catalog)
    selection_indices = np.arange(num_subhalos).astype(int)
    surviving_subhalo_indices = selection_indices[~catalog['orphan']]
    disrupted_subhalo_indices = selection_indices[catalog['orphan']]
    selected_orphan_indices = np.random.choice(disrupted_subhalo_indices,
            num_to_select, replace=False)
    return np.concatenate((surviving_subhalo_indices, selected_orphan_indices))

