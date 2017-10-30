"""
"""
import numpy as np
from halotools.mock_observables import return_xyz_formatted_array, wp


__all__ = ('masked_wp', )


def masked_wp(subhalos, mask=None):
    """
    """
    if mask is None:
        mask = np.ones_like(subhalos).astype(bool)

    rp_bins, pi_max = np.logspace(-1, 1.35, 25), 20.
    rmids = 10**(0.5*(np.log10(rp_bins[:-1]) + np.log10(rp_bins[1:])))
    pos = return_xyz_formatted_array(
            subhalos['x'], subhalos['y'], subhalos['z'],
            mask=mask, period=250,
            velocity=subhalos['vz'], velocity_distortion_dimension='z')
    return rmids, wp(pos, rp_bins, pi_max, period=250.)
