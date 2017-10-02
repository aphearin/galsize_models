"""
"""
import numpy as np

from .bulge_disk_double_power_law import data_vector_prediction
from .bulge_disk_double_power_law import lnprior as model_lnprior

from ..measurements.sdss_covariance import sdss_measurements_and_cov, logsm_bins
from ..models.load_baseline_catalogs import load_umachine_mock


try:
    mock = load_umachine_mock()
except:
    pass

sdss_data_vector, sdss_cov = sdss_measurements_and_cov()
sdss_invcov = np.linalg.inv(sdss_cov)


def lnlike(params, observations, icov, mock, logsm_bins):
    model_data_vector = data_vector_prediction(params, mock, logsm_bins, statistic='mean')
    diff = model_data_vector - sdss_data_vector
    return -np.dot(diff, np.dot(icov, diff))/2.0


def lnprob(params, observations, icov, mock, logsm_bins):
    prior = model_lnprior(params)
    if np.isinf(prior):
        return prior
    else:
        return lnlike(params, observations, icov, mock, logsm_bins) + prior


try:
    lnprob_args = tuple((sdss_data_vector, sdss_invcov, mock, logsm_bins))
except:
    pass
