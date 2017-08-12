"""
"""
import numpy as np
from halotools.utils import monte_carlo_from_cdf_lookup, build_cdf_lookup

from ..measurements.ellipse_selection_functions import ellipse_selector


def value_add_random_bt(mock, sdss):
    """
    """
    logsm_mock, ssfr_mock = np.log10(mock['obs_sm']), mock['ssfr']
    logsm_data, ssfr_data, bt_data = sdss['sm'], sdss['ssfr'], sdss['bulge_to_total_rband_meert15']
    mock['bt_meert15_random'] = assign_random_bt(logsm_mock, ssfr_mock,
                logsm_data, ssfr_data, bt_data)
    return mock


def get_percentile_based_bins(arr, nbins, lowp=0.025, highp=0.975):
    """ Calculate a binning scheme for ``arr`` based on its rank-order percentiles.

    Percentile bin spacing is linear between lowp and highp, with 0 and 1 for endpoints.

    Parameters
    ----------
    arr : ndarray
        Numpy array of shape (ndata, )

    nbins : int

    lowp : float, optional

    highp : float, optional

    Returns
    -------
    arr_bin_edges : ndarray of shape (nbins+1, )

    percentile_bin_edges : ndarray of shape (nbins+1, )
    """
    sorted_arr = np.sort(arr)
    percentile_bin_edges = np.linspace(lowp, highp, nbins-1)
    percentile_bin_edges = np.insert(np.append(percentile_bin_edges, 1.), 0, 0.)

    percentile_indices = np.floor((len(arr)-1)*percentile_bin_edges).astype(int)
    percentile_indices = np.insert(np.append(percentile_indices, len(arr)-1), 0, 0)

    arr_bin_edges = sorted_arr[percentile_indices]
    return arr_bin_edges, percentile_bin_edges


def assign_random_bt(logsm_mock, ssfr_mock, logsm_data, ssfr_data, bt_data,
        logsm_bins=np.linspace(9.8, 11.75, 15)):

    sm_bins = 10**logsm_bins
    sm_bins = np.insert(np.append(sm_bins, np.inf), 0, -np.inf)
    logsm_bins = np.insert(np.append(logsm_bins, np.inf), 0, -np.inf)

    nkeep = 2500

    bt_random = np.zeros_like(logsm_mock) + np.nan

    msg = "Working on logsm_mock = {0:.1f}, ssfr = {1:.1f}\nSDSS <sm> = {2:.1f} SDSS <ssfr> = {3:.1f}"

    for sm_low, sm_high in zip(sm_bins[:-1], sm_bins[1:]):
        ism_mask = (logsm_mock >= sm_low) & (logsm_mock < sm_high)

        if np.count_nonzero(ism_mask) > 0:
            logsm_mid = np.log10(np.median(logsm_mock[ism_mask]))

            ssfr_bins = get_percentile_based_bins(ssfr_mock[ism_mask], 15)[0]
            for jssfr, ssfr_low, ssfr_high in zip(range(len(ssfr_bins)), ssfr_bins[:-1], ssfr_bins[1:]):
                ssfr_mid = 0.5*(ssfr_low + ssfr_high)

                ij_mask = ism_mask & (ssfr_mock >= ssfr_low) & (ssfr_mock < ssfr_high)
                num_ij = np.count_nonzero(ij_mask)

                if num_ij > 0:
                    data_mask = ellipse_selector(logsm_data, ssfr_data, logsm_mid, ssfr_mid, 0., 1., nkeep)
                    data_bt_ij_sample = bt_data[data_mask]
                    x_table, y_table = build_cdf_lookup(data_bt_ij_sample)
                    bt_random[ij_mask] = monte_carlo_from_cdf_lookup(x_table, y_table, num_draws=num_ij)

    #  Fill in the tiny number of unassigned values with random values beween (0, 1)
    unassigned_mask = np.isnan(bt_random)
    bt_random[unassigned_mask] = np.random.uniform(0, 1, np.count_nonzero(unassigned_mask))

    return bt_random
