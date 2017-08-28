"""
"""
from numpy import interp
from scipy.stats import binned_statistic


__all__ = ('sdss_sample_below_median_size', )


def sdss_sample_below_median_size(sample):
    """
    """
    logsm = sample['sm']
    r50 = sample['r50_magr_kpc_meert15']
    median_size_table, logsm_bin_edges, __ = binned_statistic(logsm, r50, statistic='median')
    logsm_table = 0.5*(logsm_bin_edges[:-1] + logsm_bin_edges[1:])
    size_cut = interp(logsm, logsm_table, median_size_table)
    return r50 < size_cut
