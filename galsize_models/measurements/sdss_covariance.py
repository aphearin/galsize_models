"""
"""
import numpy as np
import os

repo_dirname = "/Users/aphearin/work/repositories/python/galsize_models"
data_subdirname = "galsize_models/measurements/data/one_point_functions"
data_dirname = os.path.join(repo_dirname, data_subdirname)

logsm_bins = np.load(os.path.join(data_dirname, 'logsm_bins.npy'))


def sdss_ssfr_sequence_means():
    """
    """
    mean_size_sf = np.load(os.path.join(data_dirname, 'mean_size_sf.npy'))
    mean_size_gv = np.load(os.path.join(data_dirname, 'mean_size_gv.npy'))
    mean_size_q = np.load(os.path.join(data_dirname, 'mean_size_q.npy'))
    return mean_size_sf, mean_size_gv, mean_size_q


def sdss_ssfr_sequence_scatter_in_dex():
    """
    """
    scatter_size_sf = np.load(os.path.join(data_dirname, 'scatter_size_sf.npy'))
    scatter_size_gv = np.load(os.path.join(data_dirname, 'scatter_size_gv.npy'))
    scatter_size_q = np.load(os.path.join(data_dirname, 'scatter_size_q.npy'))
    return scatter_size_sf, scatter_size_gv, scatter_size_q


def sdss_ssfr_sequence_means_errors():
    mean_size_sf, mean_size_gv, mean_size_q = sdss_ssfr_sequence_means()
    return 0.25*mean_size_sf, 0.25*mean_size_gv, 0.25*mean_size_q


def sdss_ssfr_sequence_scatter_errors():
    scatter_size_sf, scatter_size_gv, scatter_size_q = sdss_ssfr_sequence_scatter_in_dex()
    npts = scatter_size_sf.shape[0]
    return np.zeros(npts)+0.1, np.zeros(npts)+0.1, np.zeros(npts)+0.1


def assemble_data_vector(mean_size_sf, mean_size_gv, mean_size_q,
            scatter_size_sf, scatter_size_gv, scatter_size_q):
    return np.concatenate((mean_size_sf, mean_size_gv, mean_size_q,
                scatter_size_sf, scatter_size_gv, scatter_size_q))


def assemble_icov(size_sf_err, size_gv_err, size_q_err,
            scatter_sf_err, scatter_gv_err, scatter_q_err):
    error_vector = np.concatenate((size_sf_err, size_gv_err, size_q_err,
            scatter_sf_err, scatter_gv_err, scatter_q_err))
    return np.diag(error_vector*error_vector)


def sdss_measurements_and_errors():
    mean_size_sf, mean_size_gv, mean_size_q = sdss_ssfr_sequence_means()
    scatter_size_sf, scatter_size_gv, scatter_size_q = sdss_ssfr_sequence_scatter_in_dex()
    sdss_data_vector = assemble_data_vector(mean_size_sf, mean_size_gv, mean_size_q,
                    scatter_size_sf, scatter_size_gv, scatter_size_q)

    size_sf_err, size_gv_err, size_q_err = sdss_ssfr_sequence_means_errors()
    scatter_sf_err, scatter_gv_err, scatter_q_err = sdss_ssfr_sequence_scatter_errors()
    sdss_icov = assemble_icov(size_sf_err, size_gv_err, size_q_err,
                scatter_sf_err, scatter_gv_err, scatter_q_err)
    return sdss_data_vector, sdss_icov

