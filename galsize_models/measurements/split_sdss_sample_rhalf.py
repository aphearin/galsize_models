"""
"""
import numpy as np
from scipy.stats import binned_statistic


__all__ = ('sdss_sample_below_median_size', 'mock_sample_below_median_size')


def sdss_sample_below_median_size(sample):
    """
    """
    logsm = sample['sm']
    r50 = sample['r50_magr_kpc_meert15']
    median_size_table, logsm_bin_edges, __ = binned_statistic(logsm, r50, statistic='median')
    logsm_table = 0.5*(logsm_bin_edges[:-1] + logsm_bin_edges[1:])
    size_cut = np.interp(logsm, logsm_table, median_size_table)
    return r50 < size_cut


def mock_sample_below_median_size(sample, size_key='r50_magr_kpc_meert15'):
    """
    """
    logsm = np.log10(sample['obs_sm'])
    r50 = sample[size_key]
    median_size_table, logsm_bin_edges, __ = binned_statistic(logsm, r50, statistic='median')
    logsm_table = 0.5*(logsm_bin_edges[:-1] + logsm_bin_edges[1:])
    size_cut = np.interp(logsm, logsm_table, median_size_table)
    return r50 < size_cut

