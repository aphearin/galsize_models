"""
"""
import os
import numpy as np
from astropy.table import Table

from .read_meert15 import load_meert15


__all__ = ('cross_match_umachine_sdss_mendel13_with_meert15', )


def cross_match_umachine_sdss_mendel13_with_meert15():
    umachine_sdss_dirname = "/Users/aphearin/work/sdss/cross_matched_catalogs"
    umachine_sdss_basename = "umachine_sdss_dr10_value_added_bt.hdf5"
    umachine_sdss_fname = os.path.join(umachine_sdss_dirname, umachine_sdss_basename)
    umachine_sdss = Table.read(umachine_sdss_fname, path='data')

    meert15 = load_meert15()
    raise NotImplementedError()
