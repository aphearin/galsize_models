"""
"""
import numpy as np

from ..measurements.measure_one_points import mock_ssfr_sequence_one_points
from ..measurements.sdss_covariance import assemble_data_vector


__all__ = ('component_size_vs_rhalo', 'galaxy_size_vs_rhalo')


def component_size_vs_rhalo(rvir_halo_kpc, normalization, alpha, R0=1., scatter=0.):
    """
    """
    mean_size = normalization*(rvir_halo_kpc/R0)**alpha
    return 10**np.random.normal(loc=np.log10(mean_size), scale=scatter)


def galaxy_size_vs_rhalo(rvir_halo_kpc, bt, norm1, norm2, alpha1, alpha2, R0=1., scatter=0.):
    size1 = component_size_vs_rhalo(rvir_halo_kpc, norm1, alpha1, R0, scatter)
    size2 = component_size_vs_rhalo(rvir_halo_kpc, norm2, alpha2, R0, scatter)
    return bt*size1 + (1-bt)*size2


def data_vector_prediction(params, mock, logsm_bins):
    """
    """
    norm1, norm2, alpha1, alpha2, scatter = params
    rvir_halo_kpc = mock['rvir_halo_kpc']
    bt = mock['bt_meert15_random']
    r50_kpc = galaxy_size_vs_rhalo(rvir_halo_kpc, bt, norm1, norm2, alpha1, alpha2,
                R0=1., scatter=scatter)

    logsm = mock['logsm']
    mask_sf = mock['is_main_sequence']
    mask_gv = mock['is_green_valley']
    mask_q = mock['is_quenched']
    _x = mock_ssfr_sequence_one_points(logsm_bins, logsm, r50_kpc, mask_sf, mask_gv, mask_q)
    return assemble_data_vector(*_x)


def lnprior(params):
    """
    """
    norm1, norm2, alpha1, alpha2, scatter = params

    forbidden = norm1 < 0
    forbidden *= norm1 > 1
    forbidden *= norm2 < 0
    forbidden *= norm2 > 1
    forbidden *= alpha1 < 0
    forbidden *= alpha1 > 2
    forbidden *= alpha2 < 0
    forbidden *= alpha2 > 2
    forbidden *= scatter < 0
    forbidden *= scatter > 1
    if forbidden:
        return -np.inf
    else:
        return 0.0
