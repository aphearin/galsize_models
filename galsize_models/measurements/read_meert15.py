"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import os
from astropy.table import Table
from colossus.cosmology import cosmology


cosmo = cosmology.setCosmology("planck15")
default_datadir = r"/Users/aphearin/work/sdss/meert15"

__all__ = ("load_meert15",)


def kravtsov_read_meert_catalog(datadir=default_datadir, phot_type=4):
    """Load the Meert et al. 2015 catalog from the collection of .fits files on disk

    This catalog provides improved photometric measurements for galaxies in the
    SDSS DR7 main galaxy sample.

    Parameters
    ----------
    datadir : string, optional
        Path where the collection of .fits files are stored.
        Default value is set at the top of this module.

    phot_type : int, optional
        integer corresponding to the photometry model fit type from the catalog:
        1=best fit, 2=deVaucouleurs, 3=Sersic, 4=DeVExp, 5=SerExp. Default is 4.

    Returns
    -------
    catalogs : FITS records
        Tuple of 7 .fits records

    """
    from astropy.io import fits as pyfits

    if (phot_type < 1) or (phot_type > 5):
        raise Exception(
            "unsupported type of Meert et al. photometry: %d, choose number between 1 and 5"
        )

    datameertnonpar = os.path.join(datadir, "UPenn_PhotDec_nonParam_rband.fits")
    datameertnonparg = os.path.join(datadir, "UPenn_PhotDec_nonParam_gband.fits")
    datameert = os.path.join(datadir, "UPenn_PhotDec_Models_rband.fits")
    datasdss = os.path.join(datadir, "UPenn_PhotDec_CAST.fits")
    datasdssmodels = os.path.join(datadir, "UPenn_PhotDec_CASTmodels.fits")
    datameertg = os.path.join(datadir, "UPenn_PhotDec_Models_gband.fits")
    #  morphology probabilities from Huertas-Company et al. 2011
    datamorph = os.path.join(datadir, "UPenn_PhotDec_H2011.fits")

    # mdata tables: 1=best fit, 2=deVaucouleurs, 3=Sersic, 4=DeVExp, 5=SerExp
    mdata = pyfits.open(datameert)[phot_type].data
    mdatag = pyfits.open(datameertg)[phot_type].data
    mnpdata = pyfits.open(datameertnonpar)[1].data
    mnpdatag = pyfits.open(datameertnonparg)[1].data
    sdata = pyfits.open(datasdss)[1].data
    phot_r = pyfits.open(datasdssmodels)[1].data
    morph = pyfits.open(datamorph)[1].data

    # eliminate galaxies with bad photometry
    fflag = mdata["finalflag"]
    # print("# galaxies in initial Meert et al. sample = {0}".format(np.size(fflag)))

    def isset(flag, bit):
        """Return True if the specified bit is set in the given bit mask"""
        return (flag & (1 << bit)) != 0

    # use minimal quality cuts and flags recommended by Alan Meert
    igood = [
        (phot_r["petroMag"] > 0.0)
        & (phot_r["petroMag"] < 100.0)
        & (mnpdata["kcorr"] > 0)
        & (mdata["m_tot"] > 0)
        & (mdata["m_tot"] < 100)
        & (isset(fflag, 1) | isset(fflag, 4) | isset(fflag, 10) | isset(fflag, 14))
    ]

    sdata = sdata[igood]
    phot_r = phot_r[igood]
    mdata = mdata[igood]
    mnpdata = mnpdata[igood]
    mdatag = mdatag[igood]
    mnpdatag = mnpdatag[igood]
    morph = morph[igood]

    return sdata, mdata, mnpdata, phot_r, mdatag, mnpdatag, morph


def load_meert15(datadir=default_datadir, phot_type=4):
    """
    Parameters
    ----------
    datadir : string, optional
        Path where the collection of .fits files are stored.
        Default value is set at the top of this module.

    phot_type : int, optional
        integer corresponding to the photometry model fit type from the catalog:
        1=best fit, 2=deVaucouleurs, 3=Sersic, 4=DeVExp, 5=SerExp. Default is 4.
    """
    if datadir is None:
        datadir = default_datadir

    result = kravtsov_read_meert_catalog(datadir=datadir, phot_type=phot_type)
    sdata, mdata, mnpdata, phot_r, mdatag, mnpdatag, morph = result

    t = Table()
    t["objid"] = sdata["objid"]
    t["z"] = sdata["z"]
    t["ra"] = sdata["ra"]
    t["dec"] = sdata["dec"]

    t["magr_tot"] = mdata["m_tot"]
    t["magg_tot"] = mdatag["m_tot"]

    t["extmg"] = mnpdata["extinction"]
    t["extmr"] = mnpdatag["extinction"]

    # ext. corrected g-r color
    t["gr_uncorrected"] = t["magg_tot"] - t["magr_tot"]
    t["gr_kcorr"] = t["magg_tot"] - t["magr_tot"] - t["extmg"] + t["extmr"]

    # redshift and k-correction
    t["kcorr"] = mnpdata["kcorr"]

    # half-light radius of total light for the chosen photometric model in arcsec
    t["r50_magr_arcsec"] = mdata["r_tot"]
    t["r50_magg_arcsec"] = mdatag["r_tot"]

    # compute luminosity and angular distances in Mpc
    t["lum_dist"] = cosmo.luminosityDistance(t["z"]) / cosmo.h
    t["ang_dist"] = t["lum_dist"] / (1.0 + t["z"]) ** 2

    # abs. magnitude from the Meert et al. photometry using fit specified by phot_type above
    # corrected for extinction, evolution, and k-correction
    t["Magr_tot"] = (
        t["magr_tot"]
        - 5.0 * np.log10(t["lum_dist"] / 1e-5)
        - t["extmr"]
        + 1.3 * t["z"]
        - t["kcorr"]
    )

    # r-band luminosity in Lsun
    t["Lumr_tot"] = 0.4 * (4.67 - t["Magr_tot"])

    # Bell et al. 2003 conversion
    MsLr = -0.306 + 1.097 * t["gr_kcorr"] - 0.1

    # log10(Mstar)
    t["logsm_bell03"] = t["Lumr_tot"] + MsLr

    # other relevant quantities for plots below
    # half-light radius arcsec -> kpc
    t["r50_magr_kpc"] = (
        t["r50_magr_arcsec"] * np.pi * t["ang_dist"] * 1000.0 / (180.0 * 3600.0)
    )
    t["r50_magg_kpc"] = (
        t["r50_magg_arcsec"] * np.pi * t["ang_dist"] * 1000.0 / (180.0 * 3600.0)
    )

    t["probaSab"] = morph["probaSab"]
    t["probaScd"] = morph["probaScd"]
    t["probaS0"] = morph["probaS0"]
    t["probaEll"] = morph["probaEll"]

    # define non-overlapping morphological classes types using eqs 7 in Meert et al. (2015)
    t["morph_type_T"] = (
        -4.6 * t["probaEll"]
        - 2.4 * t["probaS0"]
        + 2.5 * t["probaSab"]
        + 6.1 * t["probaScd"]
    )

    # component magnitudes
    t["magr_bulge"] = mdata["m_bulge"]
    t["magr_disk"] = mdata["m_disk"]
    t["magg_bulge"] = mdatag["m_bulge"]
    t["magg_disk"] = mdatag["m_disk"]

    t["Magr_bulge"] = (
        t["magr_bulge"]
        - 5.0 * np.log10(t["lum_dist"] / 1e-5)
        - t["extmr"]
        + 1.3 * t["z"]
        - t["kcorr"]
    )
    t["Magr_disk"] = (
        t["magr_disk"]
        - 5.0 * np.log10(t["lum_dist"] / 1e-5)
        - t["extmr"]
        + 1.3 * t["z"]
        - t["kcorr"]
    )
    t["Lumr_bulge"] = 10**0.4 * (4.67 - t["Magr_bulge"])
    t["Lumr_disk"] = 10**0.4 * (4.67 - t["Magr_disk"])

    bulge_term = 10.0 ** (-t["Magr_bulge"] / 2.5)
    disk_term = 10.0 ** (-t["Magr_disk"] / 2.5)
    t["bulge_to_total_rband"] = bulge_term / (bulge_term + disk_term)
    # t['bulge_to_total_rband'] = t['Lumr_bulge']/(t['Lumr_bulge']+t['Lumr_disk'])

    # component half-light in arcsec
    t["r50_magr_disk_arcsec"] = mdata["r_disk"]
    t["r50_magr_bulge_arcsec"] = mdata["r_bulge"]

    # ext. corrected g-r color of components
    t["gr_bulge"] = t["magg_bulge"] - t["magr_bulge"] - t["extmg"] + t["extmr"]
    t["gr_disk"] = t["magg_disk"] - t["magr_disk"] - t["extmg"] + t["extmr"]

    # define selection criteria
    mask = (
        (t["z"] > 0.001)
        & (t["z"] < 0.2)
        & (t["magr_tot"] > 14.0)
        & (t["magr_tot"] < 17.77)
        & (t["r50_magr_arcsec"] > 0.75)
        & (t["extmr"] > 0.0)
        & (t["magg_tot"] > 0.0)
        & (t["extmg"] > 0.0)
        & (t["gr_kcorr"] > -0.5)
        & (t["gr_kcorr"] < 2.2)
    )

    return t[mask]
