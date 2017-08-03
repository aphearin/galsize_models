"""
"""
import os
import numpy as np
from astropy.table import Table


__all__ = ('load_umachine_sdss_with_meert15', )

default_datadir = "/Users/aphearin/work/sdss/cross_matched_catalogs/meert15"


def load_umachine_sdss_with_meert15(datadir=default_datadir,
        zmin=-np.inf, zmax=np.inf, sm_low=-np.inf, sm_high=np.inf):
    """
    Examples
    --------
    >>> full_sdss, is_complete = load_umachine_sdss_with_meert15()
    """
    basename = "umachine_sdss_dr10_meert15.hdf5"
    fname = os.path.join(datadir, basename)
    full_sdss = Table.read(fname, path='data')

    completeness_table = np.loadtxt(os.path.join(datadir, 'completeness.dat'))
    completeness_limit = np.interp(full_sdss['sm'], completeness_table[:, 0], completeness_table[:, 1])
    is_complete = full_sdss['z'] < completeness_limit

    full_sdss['ssfr'] = np.log10(full_sdss['sfr']/10**full_sdss['sm'])
    return full_sdss, is_complete
