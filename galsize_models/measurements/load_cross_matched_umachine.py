"""
"""
import os
import numpy as np
from astropy.table import Table


__all__ = ('load_umachine_sdss_with_meert15', 'mendel13_bulge_to_total')

default_datadir = "/Users/aphearin/work/sdss/cross_matched_catalogs/meert15"


def load_umachine_sdss_with_meert15(datadir=default_datadir):
    """
    Examples
    --------
    >>> full_sdss, is_complete, has_profile = load_umachine_sdss_with_meert15()
    """
    basename = "umachine_sdss_dr10_meert15.hdf5"
    fname = os.path.join(datadir, basename)
    full_sdss = Table.read(fname, path='data')

    completeness_table_dirname = "/Users/aphearin/Dropbox/UniverseMachine/data/sdss"
    completeness_table = np.loadtxt(os.path.join(completeness_table_dirname, 'completeness.dat'))

    z_limit = np.interp(full_sdss['sm'],
            completeness_table[:, 0], completeness_table[:, 1])
    is_complete = (full_sdss['z'] < z_limit)

    full_sdss['ssfr'] = np.log10(full_sdss['sfr']/10**full_sdss['sm'])

    good_profile_mask = ~np.isnan(full_sdss['Magr_tot_meert15'])
    good_profile_mask *= ~np.isnan(full_sdss['bulge_to_total_rband_meert15'])

    return full_sdss, is_complete.astype(bool), good_profile_mask.astype(bool)


def mendel13_bulge_to_total(sample):
    msg = "Must first make a cut on having a good Mendel+13 B/D decomposition"
    assert ~np.any(np.isnan(sample['logMB_mendel13'])), msg
    assert ~np.any(np.isnan(sample['logMD_mendel13'])), msg
    assert np.all(sample['logMB_mendel13'] > 0), msg
    assert np.all(sample['logMD_mendel13'] > 0), msg

    bulge_mass = 10**sample['logMB_mendel13']
    disk_mass = 10**sample['logMD_mendel13']
    return bulge_mass/(bulge_mass + disk_mass)
