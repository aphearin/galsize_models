"""
"""
import numpy as np

from ..measurements.measure_one_points import mock_ssfr_sequence_one_points
from ..measurements.sdss_covariance import assemble_data_vector

norm_bulge_priors = (0, 0.1)
bulge_to_disk_size_ratio_priors = (0.1, 0.5)
alpha_bulge_priors = (0.5, 1.5)
alpha_disk_priors = (0.5, 1.5)
scatter_priors = (0.1, 0.4)


__all__ = ('component_size_vs_rhalo', 'galaxy_size_vs_rhalo')


def component_size_vs_rhalo(rvir_halo_kpc, normalization, alpha, R0=1., scatter=0.):
    """
    """
    mean_size = normalization*(rvir_halo_kpc/R0)**alpha
    return 10**np.random.normal(loc=np.log10(mean_size), scale=scatter)


def galaxy_size_vs_rhalo(rvir_halo_kpc, bt, norm_bulge, norm_disk, alpha_bulge, alpha_disk,
            R0=1., scatter=0.):
    size1 = component_size_vs_rhalo(rvir_halo_kpc, norm_bulge, alpha_bulge, R0, scatter)
    size2 = component_size_vs_rhalo(rvir_halo_kpc, norm_disk, alpha_disk, R0, scatter)
    return bt*size1 + (1-bt)*size2


def data_vector_prediction(params, mock, logsm_bins, statistic='mean'):
    """
    """
    norm_bulge, bulge_to_disk_size_ratio, alpha_bulge, alpha_disk, scatter = params
    norm_disk = norm_bulge/float(bulge_to_disk_size_ratio)
    rvir_halo_kpc = mock['rvir_halo_kpc']
    bt = mock['bt_meert15_random']
    r50_kpc = galaxy_size_vs_rhalo(rvir_halo_kpc, bt, norm_bulge, norm_disk, alpha_bulge, alpha_disk,
                R0=1., scatter=scatter)

    logsm = mock['logsm']
    mask_sf = mock['is_main_sequence']
    mask_gv = mock['is_green_valley']
    mask_q = mock['is_quenched']
    _x = mock_ssfr_sequence_one_points(logsm_bins, logsm, r50_kpc,
                mask_sf, mask_gv, mask_q, statistic=statistic)
    return assemble_data_vector(*_x)


def lnprior(params):
    """
    Examples
    --------
    >>> norm_bulge, bulge_to_disk_size_ratio, alpha_bulge, alpha_disk, scatter = 0.05, 0.2, 0.7, 0.7, 0.25
    >>> params = (norm_bulge, bulge_to_disk_size_ratio, alpha_bulge, alpha_disk, scatter)
    >>> prior = lnprior(params)
    >>> assert ~np.isinf(prior)

    >>> norm_bulge, bulge_to_disk_size_ratio, alpha_bulge, alpha_disk, scatter = -0.05, 0.2, 0.7, 0.7, 0.25
    >>> params = (norm_bulge, bulge_to_disk_size_ratio, alpha_bulge, alpha_disk, scatter)
    >>> prior = lnprior(params)
    >>> assert np.isinf(prior)

    >>> norm_bulge, bulge_to_disk_size_ratio, alpha_bulge, alpha_disk, scatter = 0.05, 2, 0.7, 0.7, 0.25
    >>> params = (norm_bulge, bulge_to_disk_size_ratio, alpha_bulge, alpha_disk, scatter)
    >>> prior = lnprior(params)
    >>> assert np.isinf(prior)

    >>> norm_bulge, bulge_to_disk_size_ratio, alpha_bulge, alpha_disk, scatter = 0.05, 2, 0.1, 0.7, 0.25
    >>> params = (norm_bulge, bulge_to_disk_size_ratio, alpha_bulge, alpha_disk, scatter)
    >>> prior = lnprior(params)
    >>> assert np.isinf(prior)

    >>> norm_bulge, bulge_to_disk_size_ratio, alpha_bulge, alpha_disk, scatter = 0.05, 2, 0.1, 0.7, 0.25
    >>> params = (norm_bulge, bulge_to_disk_size_ratio, alpha_bulge, alpha_disk, scatter)
    >>> prior = lnprior(params)
    >>> assert np.isinf(prior)

    >>> norm_bulge, bulge_to_disk_size_ratio, alpha_bulge, alpha_disk, scatter = 0.05, 2, 0.1, 0.7, 0.5
    >>> params = (norm_bulge, bulge_to_disk_size_ratio, alpha_bulge, alpha_disk, scatter)
    >>> prior = lnprior(params)
    >>> assert np.isinf(prior)
    """
    norm_bulge, bulge_to_disk_size_ratio, alpha_bulge, alpha_disk, scatter = params

    acceptable = norm_bulge_priors[0] < norm_bulge < norm_bulge_priors[1]
    acceptable *= bulge_to_disk_size_ratio_priors[0] < bulge_to_disk_size_ratio < bulge_to_disk_size_ratio_priors[1]
    acceptable *= alpha_bulge_priors[0] < alpha_bulge < alpha_bulge_priors[1]
    acceptable *= alpha_disk_priors[0] < alpha_disk < alpha_disk_priors[1]
    acceptable *= scatter_priors[0] < scatter < scatter_priors[1]

    if bool(acceptable) is True:
        return 0.0
    else:
        return -np.inf


def set_mcmc_p0(nwalkers):
    """
    Examples
    --------
    >>> nwalkers = 10
    >>> p0 = set_mcmc_p0(nwalkers)
    >>> assert np.shape(p0) == (nwalkers, 5)
    >>> priors_init = [lnprior(params) for params in p0]
    >>> assert ~np.any(np.isinf(priors_init))
    """
    norm_bulge_init = np.random.uniform(norm_bulge_priors[0], norm_bulge_priors[1], nwalkers)
    bulge_to_disk_size_ratio_init = np.random.uniform(bulge_to_disk_size_ratio_priors[0], bulge_to_disk_size_ratio_priors[1], nwalkers)
    alpha_bulge_init = np.random.uniform(alpha_bulge_priors[0], alpha_bulge_priors[1], nwalkers)
    alpha_disk_init = np.random.uniform(alpha_disk_priors[0], alpha_disk_priors[1], nwalkers)
    scatter_init = np.random.uniform(scatter_priors[0], scatter_priors[1], nwalkers)

    return np.vstack((norm_bulge_init, bulge_to_disk_size_ratio_init, alpha_bulge_init,
                alpha_disk_init, scatter_init)).T


