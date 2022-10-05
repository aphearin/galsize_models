"""
"""
import os
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord, Distance
from astropy import units as u

from .read_meert15 import load_meert15


__all__ = ("cross_match_umachine_sdss_meert15_with_meert15",)


def cross_match_umachine_sdss_meert15_with_meert15(
    mpc_match_dist_cut=1.7, umachine_sdss_fname=None, meert15_dir=None, verbose=False
):
    """
    Examples
    --------
    >>> result = cross_match_umachine_sdss_meert15_with_meert15()
    """
    if umachine_sdss_fname is None:
        umachine_sdss_dirname = "/Users/aphearin/work/DATA/SDSS/cross_matched_catalogs"
        umachine_sdss_basename = "umachine_sdss_dr10_value_added_bt.hdf5"
        umachine_sdss_fname = os.path.join(
            umachine_sdss_dirname, umachine_sdss_basename
        )
    umachine_sdss = Table.read(umachine_sdss_fname, path="data")

    meert15_sdss = load_meert15(datadir=meert15_dir)

    #  Cross-match the two catalogs on ra and dec
    meert15_sdss_coords = SkyCoord(
        ra=meert15_sdss["ra"] * u.degree,
        dec=meert15_sdss["dec"] * u.degree,
        distance=Distance(z=meert15_sdss["z"]),
    )

    umachine_sdss_coords = SkyCoord(
        ra=umachine_sdss["ra"] * u.degree,
        dec=umachine_sdss["dec"] * u.degree,
        distance=Distance(z=umachine_sdss["z"]),
    )

    idx, d2d, d3d = umachine_sdss_coords.match_to_catalog_3d(meert15_sdss_coords)
    cross_matched_meert15_sdss = meert15_sdss[idx]

    good_match_mask = d3d.value <= mpc_match_dist_cut
    num_good_matches = len(d3d[good_match_mask])
    if verbose:
        print(
            "Fraction of objects in umachine catalog with a clean match = {0:.2f}".format(
                num_good_matches / float(len(d3d))
            )
        )

    umachine_sdss["has_meert15_match"] = 0
    umachine_sdss["has_meert15_match"][good_match_mask] = 1

    keys_to_inherit = (
        "r50_magr_kpc",
        "Magr_tot",
        "Magr_bulge",
        "Magr_disk",
        "gr_bulge",
        "gr_disk",
        "morph_type_T",
        "gr_kcorr",
        "bulge_to_total_rband",
        "logsm_bell03",
    )

    for key in keys_to_inherit:
        umachine_sdss[key + "_meert15"] = np.nan
        umachine_sdss[key + "_meert15"][good_match_mask] = cross_matched_meert15_sdss[
            key
        ][good_match_mask]

    return umachine_sdss


def store_cross_matched_catalog_to_disk(
    commit_hash, output_fname, overwrite=False, **kwargs
):
    """ """
    new_catalog = cross_match_umachine_sdss_meert15_with_meert15(**kwargs)
    new_catalog.write(output_fname, path="data", overwrite=overwrite)

    import h5py

    f = h5py.File(output_fname)

    key = "comments"
    msg = (
        "This catalog was initially created by create_value_added_dr10_catalog.ipynb in "
        "bt_models/master, \nwith further cross-matching against the Meert+15 morphology catalogs \n"
        " done by the cross_match_umachine_sdss_meert15_with_meert15 function of the \n"
        "https://github.com/aphearin/galsize_models repo, commit {0}.\n"
        "The store_cross_matched_catalog_to_disk function in the galsize_models repo \n"
        "created this hdf5 file on disk.\n"
        "Briefly, the umachine DR10 sample "
        "has been cross-matched against my own query to DR7 to get DR7-objIDs; "
        "these objIDs have been cross-matched against those in Mendel+13. ".format(
            commit_hash
        )
    )

    f.attrs.create(key, msg)
    f.close()
