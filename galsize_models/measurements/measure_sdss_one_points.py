"""
"""
from scipy.stats import binned_statistic


__all__ = ('size_vs_stellar_mass', )


def size_vs_stellar_mass(logsm, size, logsm_bins, statistic='mean'):
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
    result, bin_edges, binnumber = binned_statistic(logsm, size, bins=logsm_bins,
            statistic=statistic)
    logsm_mids = 0.5*(logsm_bins[:-1] + logsm_bins[1:])
    return result, logsm_mids
