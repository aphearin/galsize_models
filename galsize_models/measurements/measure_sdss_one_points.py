"""
"""
import os
import numpy as np

from .load_cross_matched_umachine import load_umachine_sdss_with_meert15


__all__ = ('sdss_size_vs_stellar_mass', 'tabulate_sdss_size_vs_stellar_mass',
        'load_sdss_size_vs_stellar_mass')


default_datadir = "/Users/aphearin/work/sdss/cross_matched_catalogs/meert15"


def sdss_size_vs_stellar_mass(logsm, size, redshift, logsm_bins, statistic='mean',
        datadir=default_datadir, zmin=0.02):
    """
    Parameters
    ----------
    logsm : ndarray
        Numpy array of shape (ngals, ) storing log10(M*/Msun)

    size : ndarray
        Numpy array of shape (ngals, ) storing galaxy size in units of kpc

    logsm_bins : ndarray
        Numpy array of shape (nbins+1, ) storing the bin edges used to compute mean size

    statistic : string, optional
        Either ``mean`` or ``median``. Default is ``mean``.

    Returns
    -------
    result : ndarray
        Numpy array of shape (nbins, )

    logsm_mids : ndarray
        Numpy array of shape (nbins, ) storing the bin midpoints
    """
    completeness_table = np.loadtxt(os.path.join(datadir, 'completeness.dat'))
    logsm_mids = 0.5*(logsm_bins[:-1] + logsm_bins[1:])
    assert np.all(logsm_bins >= 9.5), "SDSS volume is too small for logsm bins < 9.5"
    if statistic == 'mean':
        f = np.mean
    elif statistic == 'median':
        f = np.median
    else:
        raise ValueError("Choose ``mean`` or ``median`` for ``statistic``")

    nbins = len(logsm_mids)
    result = np.zeros(nbins)

    for i, logsm_low, logsm_high in zip(range(nbins), logsm_bins[:-1], logsm_bins[1:]):
        zcut = np.interp(logsm_low, completeness_table[:, 0], completeness_table[:, 1])
        mask = (redshift < zcut) & (redshift >= zmin)
        mask *= (logsm >= logsm_low) & (logsm < logsm_high)
        mask *= ~np.isnan(size)
        result[i] = f(size[mask])

    return result, logsm_mids


def tabulate_sdss_size_vs_stellar_mass(output_dirname=os.path.abspath('.')):
    """
    """
    logsm_min, logsm_max = 9.75, 11.25
    full_sdss, __ = load_umachine_sdss_with_meert15(sm_limit=logsm_min)

    dlogsm = 0.15
    logsm_bins = np.arange(logsm_min, logsm_max+dlogsm, dlogsm)

    mask_all = np.ones(len(full_sdss), dtype=bool)
    sm = full_sdss['sm'][mask_all]
    size = full_sdss['r50_magr_kpc_meert15'][mask_all]
    redshift = full_sdss['z'][mask_all]
    mean_size_all, logsm_mids = sdss_size_vs_stellar_mass(sm, size, redshift, logsm_bins)

    mask_q = full_sdss['ssfr'] < -11.25
    sm = full_sdss['sm'][mask_q]
    size = full_sdss['r50_magr_kpc_meert15'][mask_q]
    redshift = full_sdss['z'][mask_q]
    mean_size_q, logsm_mids = sdss_size_vs_stellar_mass(sm, size, redshift, logsm_bins)

    mask_sf = full_sdss['ssfr'] >= -10.75
    sm = full_sdss['sm'][mask_sf]
    size = full_sdss['r50_magr_kpc_meert15'][mask_sf]
    redshift = full_sdss['z'][mask_sf]
    mean_size_sf, logsm_mids = sdss_size_vs_stellar_mass(sm, size, redshift, logsm_bins)

    mask_gv = (full_sdss['ssfr'] <= -10.75) & (full_sdss['ssfr'] > -11.25)
    sm = full_sdss['sm'][mask_gv]
    size = full_sdss['r50_magr_kpc_meert15'][mask_gv]
    redshift = full_sdss['z'][mask_gv]
    mean_size_gv, logsm_mids = sdss_size_vs_stellar_mass(sm, size, redshift, logsm_bins)

    np.save(os.path.join(output_dirname, 'logsm_bins'), logsm_bins)
    np.save(os.path.join(output_dirname, 'mean_size_all'), mean_size_all)
    np.save(os.path.join(output_dirname, 'mean_size_q'), mean_size_q)
    np.save(os.path.join(output_dirname, 'mean_size_sf'), mean_size_sf)
    np.save(os.path.join(output_dirname, 'mean_size_gv'), mean_size_gv)


def load_sdss_size_vs_stellar_mass(output_dirname):
    output_dirname = os.path.abspath(output_dirname)
    logsm_bins = np.load(os.path.join(output_dirname, 'logsm_bins.npy'))
    mean_size_all = np.load(os.path.join(output_dirname, 'mean_size_all.npy'))
    mean_size_q = np.load(os.path.join(output_dirname, 'mean_size_q.npy'))
    mean_size_sf = np.load(os.path.join(output_dirname, 'mean_size_sf.npy'))
    mean_size_gv = np.load(os.path.join(output_dirname, 'mean_size_gv.npy'))
    return (logsm_bins, mean_size_all, mean_size_q, mean_size_sf, mean_size_gv)






