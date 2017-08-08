"""
"""
import os
import numpy as np


__all__ = ('sdss_size_vs_stellar_mass', )


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


def tabulate_sdss_size_vs_stellar_mass():
    """
    """
    pass
