"""
"""
import numpy as np
from halotools.mock_observables import return_xyz_formatted_array, wp


__all__ = ('wp_size_ratios_mock', 'single_component_ratios')


def wp_size_ratios_mock(mock, logsm_min, sample_cut, size_key='r50_magr_kpc_meert15',
        rp_bins=np.logspace(-1, 1.25, 20), pi_max=20., period=250.):
    """
    """
    rp_mids = 10**(0.5*(np.log10(rp_bins[:-1]) + np.log10(rp_bins[1:])))

    mask_sm_thresh = mock['obs_sm'] > 10**logsm_min

    if sample_cut == 'sf':
        sample_mask = mask_sm_thresh & (mock['ssfr'] >= -10.75)
    elif sample_cut == 'q':
        sample_mask = mask_sm_thresh & (mock['ssfr'] < -11.25)
    elif sample_cut == 'bulge':
        sample_mask = mask_sm_thresh & (mock['bt_meert15_random'] > 0.75)
    elif sample_cut == 'disk':
        sample_mask = mask_sm_thresh & (mock['bt_meert15_random'] < 0.25)
    elif sample_cut == 'mixed':
        sample_mask = mask_sm_thresh & (mock['bt_meert15_random'] >= 0.25) & (mock['bt_meert15_random'] < 0.75)
    else:
        raise ValueError("sample_cut = {0} not recognized")

    size_cut = np.median(mock[size_key][sample_mask])
    sample_mask_small = sample_mask & (mock[size_key] < size_cut)
    sample_mask_large = sample_mask & (mock[size_key] >= size_cut)

    pos_sm_sf = return_xyz_formatted_array(
            mock['x'], mock['y'], mock['z'], velocity=mock['vz'],
            velocity_distortion_dimension='z', period=period, mask=sample_mask)
    pos_sm_sf_small = return_xyz_formatted_array(
            mock['x'], mock['y'], mock['z'], velocity=mock['vz'],
            velocity_distortion_dimension='z', period=period, mask=sample_mask_small)
    pos_sm_sf_large = return_xyz_formatted_array(
            mock['x'], mock['y'], mock['z'], velocity=mock['vz'],
            velocity_distortion_dimension='z', period=period, mask=sample_mask_large)

    wp_sm = wp(pos_sm_sf, rp_bins, pi_max, period=period)
    wp_sm_small = wp(pos_sm_sf_small, rp_bins, pi_max, period=period)
    wp_sm_large = wp(pos_sm_sf_large, rp_bins, pi_max, period=period)

    return rp_mids, wp_sm, wp_sm_small, wp_sm_large


def single_component_ratios(mock, rp_bins=np.logspace(-1, 1.25, 20),
            pi_max=20., period=250., size_key='r50', num_gals_max=int(1e5)):
    """
    """
    result_collector = []
    rp_mids = 10**(0.5*(np.log10(rp_bins[:-1]) + np.log10(rp_bins[1:])))
    result_collector.append(rp_mids)

    for logsm_cut in (9.75, 10.25, 10.75, 11.25):
        sample = mock[mock['mstar'] > 10**logsm_cut]
        if len(sample) > num_gals_max:
            downsampling_mask = np.random.choice(np.arange(len(sample)), num_gals_max, replace=False)
            sample = sample[downsampling_mask]

        pos_all = return_xyz_formatted_array(sample['x'], sample['y'], sample['z'],
                        velocity=sample['vz'], velocity_distortion_dimension='z', period=period)

        pos_small = return_xyz_formatted_array(sample['x'], sample['y'], sample['z'],
                        velocity=sample['vz'], velocity_distortion_dimension='z', period=period,
                        mask=(sample[size_key] < sample[size_key+'_median']))
        pos_large = return_xyz_formatted_array(sample['x'], sample['y'], sample['z'],
                        velocity=sample['vz'], velocity_distortion_dimension='z', period=period,
                        mask=(sample[size_key] >= sample[size_key+'_median']))

        wp_all = wp(pos_all, rp_bins, pi_max, period=period)
        wp_small = wp(pos_small, rp_bins, pi_max, period=period)
        wp_large = wp(pos_large, rp_bins, pi_max, period=period)
        fracdiff = (wp_large-wp_small)/wp_all
        result_collector.append(fracdiff)

    return result_collector
