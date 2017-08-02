"""
"""
import numpy as np


__all__ = ('galsize_vs_rvir', )


def galsize_vs_rvir(r200c, normalization=0.015, scatter_in_dex=0.2):
    """
    Parameters
    ----------
    r200c : float or ndarray
        Float or ndarray of shape (ngals, ) storing the halo radius,
        given in physical coordinates with units of kpc,
        where halo radius is defined as 200 times the critical density.

    normalization : float, optional
        Coefficient controlling the linear scaling between ``r200c`` of the halo
        and half-mass radius of the galaxy. Default value is 0.015, as in Kravtsov (2013).

    scatter_in_dex : float, optional
        Level of log-normal scatter between halo and galaxy size.
        Default value is 0.2 dex, as in Kravtsov (2013).

    Returns
    -------
    galsize : float or ndarray
        Float or ndarray of shape (ngals, ) storing the half-mass radius of the
        model galaxy in units of kpc.

    Examples
    --------
    >>> ngals = 500
    >>> r200c = np.logspace(0.1, 2, ngals)
    >>> galsize = galsize_vs_rvir(r200c)
    """
    return 10**np.random.normal(loc=np.log10(normalization*r200c), scale=scatter_in_dex)
