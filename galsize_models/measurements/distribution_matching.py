"""
"""
import numpy as np


def distribution_matching_indices(distribution1, distribution2, nselect, bins=100):
    """ """
    hist2, bins = np.histogram(distribution2, density=True, bins=bins)
    hist1 = np.histogram(distribution1, bins=bins, density=True)[0].astype(float)

    hist_ratio = np.zeros_like(hist2, dtype=float)
    hist_ratio[hist1 > 0] = hist2[hist1 > 0] / hist1[hist1 > 0]

    bin_mids = 0.5 * (bins[:-1] + bins[1:])
    hist_ratio_interp = np.interp(distribution1, bin_mids, hist_ratio)
    prob_select = hist_ratio_interp / float(hist_ratio_interp.sum())

    candidate_indices = np.arange(len(distribution1))
    indices = np.random.choice(
        candidate_indices, size=nselect, replace=True, p=prob_select
    )
    return indices
