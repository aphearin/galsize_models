"""
"""
import numpy as np
from halotools.mock_observables import return_xyz_formatted_array, wp


__all__ = ('wp_ssfr_sequence_mock', )


def wp_ssfr_sequence_mock(mock, logsm_min, ssfr_sample, size_key='r50_magr_kpc_meert15',
        rp_bins=np.logspace(-1, 1.25, 25), pi_max=20., period=250.):
    """
    """
    rp_mids = 10**(0.5*(np.log10(rp_bins[:-1]) + np.log10(rp_bins[1:])))

    mask_sm_thresh = mock['obs_sm'] > 10**logsm_min

    if ssfr_sample == 'sf':
        mask_sm_ssfr = mask_sm_thresh & (mock['ssfr'] >= -10.75)
    elif ssfr_sample == 'q':
        mask_sm_ssfr = mask_sm_thresh & (mock['ssfr'] < -11.25)
    else:
        raise ValueError("ssfr_sample = {0} not recognized")

    size_cut = np.median(mock[size_key][mask_sm_ssfr])
    mask_sm_ssfr_small = mask_sm_ssfr & (mock[size_key] < size_cut)
    mask_sm_ssfr_large = mask_sm_ssfr & (mock[size_key] >= size_cut)

    pos_sm_sf = return_xyz_formatted_array(
            mock['x'], mock['y'], mock['z'], velocity=mock['vz'],
            velocity_distortion_dimension='z', period=period, mask=mask_sm_ssfr)
    pos_sm_sf_small = return_xyz_formatted_array(
            mock['x'], mock['y'], mock['z'], velocity=mock['vz'],
            velocity_distortion_dimension='z', period=period, mask=mask_sm_ssfr_small)
    pos_sm_sf_large = return_xyz_formatted_array(
            mock['x'], mock['y'], mock['z'], velocity=mock['vz'],
            velocity_distortion_dimension='z', period=period, mask=mask_sm_ssfr_large)

    wp_sm = wp(pos_sm_sf, rp_bins, pi_max, period=period)
    wp_sm_small = wp(pos_sm_sf_small, rp_bins, pi_max, period=period)
    wp_sm_large = wp(pos_sm_sf_large, rp_bins, pi_max, period=period)

    return rp_mids, wp_sm, wp_sm_small, wp_sm_large
