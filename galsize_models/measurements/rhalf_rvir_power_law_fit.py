"""
"""
import os
import numpy as np
from scipy.stats import binned_statistic
from halotools.empirical_models import halo_radius_to_halo_mass, Moster13SmHm
from astropy.cosmology import Planck15


__all__ = ("rvir_rhalf_power_law_fit",)

default_datadir = "/Users/aphearin/Dropbox/UniverseMachine/data/sdss"


def _completeness_redshift_vs_stellar_mass(sm_planck15, datadir=default_datadir):
    completeness_table = np.loadtxt(os.path.join(datadir, "completeness.dat"))
    completeness_redshift = np.interp(
        np.log10(sm_planck15), completeness_table[:, 0], completeness_table[:, 1]
    )
    return completeness_redshift


def _completeness_redshift_vs_rhalo(
    rhalo_planck15_kpc,
    cosmology=Planck15,
    median_redshift=0,
    mdef="vir",
    model=Moster13SmHm(),
    **kwargs
):
    rhalo_unity_h_mpc = cosmology.h * rhalo_planck15_kpc / 1000.0
    halo_mass_unity_h = halo_radius_to_halo_mass(
        rhalo_unity_h_mpc, cosmology, median_redshift, mdef
    )
    # halo_mass_planck15 = halo_mass_unity_h/cosmology.h
    stellar_mass_unity_h = model.mean_stellar_mass(prim_haloprop=halo_mass_unity_h)
    sm_planck15 = stellar_mass_unity_h / cosmology.h / cosmology.h
    completeness_redshift = _completeness_redshift_vs_stellar_mass(sm_planck15)
    return completeness_redshift


def rvir_rhalf_power_law_fit(
    rhalf_planck15_kpc,
    rhalo_planck15_kpc,
    log10_rhalo_min=np.log10(100.0),
    log10_rhalo_max=np.log10(2500.0),
    num_bins=20,
):
    """Fit for the normalization and power law index of the relation
    Rhalf = A*Rvir**alpha.

    Parameters
    ----------
    rhalf_planck15_kpc : ndarray

    rhalo_planck15_kpc : ndarray

    Returns
    -------
    alpha : float
        power law index

    normalization : float
        Constant factor ``A`` in the scaling relation
    """
    rhalo_bins = np.logspace(log10_rhalo_min, log10_rhalo_max, num_bins)
    median_rhalf, bin_edges, __ = binned_statistic(
        rhalo_planck15_kpc, rhalf_planck15_kpc, bins=rhalo_bins, statistic="median"
    )
    bin_mids = 10 ** (0.5 * (np.log10(rhalo_bins[:-1]) + np.log10(rhalo_bins[1:])))

    nan_mask = ~np.isnan(bin_mids) & ~np.isnan(median_rhalf)
    polyfit_degree = 1
    c1, logc0 = np.polyfit(
        np.log(bin_mids[nan_mask]), np.log(median_rhalf[nan_mask]), polyfit_degree
    )
    c0 = np.exp(logc0)
    return c1, c0
