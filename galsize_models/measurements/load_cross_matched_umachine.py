"""
"""
import os
import numpy as np
from astropy.table import Table


__all__ = ('load_umachine_sdss_with_meert15', )

default_datadir = "/Users/aphearin/work/sdss/cross_matched_catalogs/meert15"


def load_umachine_sdss_with_meert15(datadir=default_datadir):
    """
    Examples
    --------
    >>> full_sdss, is_complete = load_umachine_sdss_with_meert15(sm_limit=9.75)
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
    return full_sdss, is_complete
