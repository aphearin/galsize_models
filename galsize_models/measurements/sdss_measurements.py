"""
"""
import numpy as np
import os

repo_dirname = "/Users/aphearin/work/repositories/python/galsize_models"
output_subdirname = "galsize_models/measurements/data/two_point_functions"
output_dirname = os.path.join(repo_dirname, output_subdirname)

cutoff_err = 0.1

rp = np.load(os.path.join(output_dirname, 'sdss_sm9p75_smbell_rp.npy'))
wp_sdss_sm9p75_smbell = np.load(os.path.join(output_dirname, 'sdss_sm9p75_smbell_wp.npy'))
wperr_sdss_sm9p75_smbell = np.load(os.path.join(output_dirname, 'sdss_sm9p75_smbell_wperr.npy'))
wp_sdss_sm9p75_smbell_small = np.load(os.path.join(output_dirname, 'sdss_sm9p75_smbell_small_wp.npy'))
wp_sdss_sm9p75_smbell_large = np.load(os.path.join(output_dirname, 'sdss_sm9p75_smbell_large_wp.npy'))
fracdiff_sm9p75_smbell = (wp_sdss_sm9p75_smbell_large-wp_sdss_sm9p75_smbell_small)/wp_sdss_sm9p75_smbell
fracdiff_sm9p75_smbell_err = np.maximum(np.sqrt(wperr_sdss_sm9p75_smbell)/wp_sdss_sm9p75_smbell, cutoff_err)

wp_sdss_sm10p25_smbell = np.load(os.path.join(output_dirname, 'sdss_sm10p25_smbell_wp.npy'))
wperr_sdss_sm10p25_smbell = np.load(os.path.join(output_dirname, 'sdss_sm10p25_smbell_wperr.npy'))
wp_sdss_sm10p25_smbell_small = np.load(os.path.join(output_dirname, 'sdss_sm10p25_smbell_small_wp.npy'))
wp_sdss_sm10p25_smbell_large = np.load(os.path.join(output_dirname, 'sdss_sm10p25_smbell_large_wp.npy'))
fracdiff_sm10p25_smbell = (wp_sdss_sm10p25_smbell_large-wp_sdss_sm10p25_smbell_small)/wp_sdss_sm10p25_smbell
fracdiff_sm10p25_smbell_err = np.maximum(np.sqrt(wperr_sdss_sm10p25_smbell)/wp_sdss_sm10p25_smbell, cutoff_err)

wp_sdss_sm10p75_smbell = np.load(os.path.join(output_dirname, 'sdss_sm10p75_smbell_wp.npy'))
wperr_sdss_sm10p75_smbell = np.load(os.path.join(output_dirname, 'sdss_sm10p75_smbell_wperr.npy'))
wp_sdss_sm10p75_smbell_small = np.load(os.path.join(output_dirname, 'sdss_sm10p75_smbell_small_wp.npy'))
wp_sdss_sm10p75_smbell_large = np.load(os.path.join(output_dirname, 'sdss_sm10p75_smbell_large_wp.npy'))
fracdiff_sm10p75_smbell = (wp_sdss_sm10p75_smbell_large-wp_sdss_sm10p75_smbell_small)/wp_sdss_sm10p75_smbell
fracdiff_sm10p75_smbell_err = np.maximum(np.sqrt(wperr_sdss_sm10p75_smbell)/wp_sdss_sm10p75_smbell, cutoff_err)

wp_sdss_sm11p25_smbell = np.load(os.path.join(output_dirname, 'sdss_sm11p25_smbell_wp.npy'))
wperr_sdss_sm11p25_smbell = np.load(os.path.join(output_dirname, 'sdss_sm11p25_smbell_wperr.npy'))
wp_sdss_sm11p25_smbell_small = np.load(os.path.join(output_dirname, 'sdss_sm11p25_smbell_small_wp.npy'))
wp_sdss_sm11p25_smbell_large = np.load(os.path.join(output_dirname, 'sdss_sm11p25_smbell_large_wp.npy'))
fracdiff_sm11p25_smbell = (wp_sdss_sm11p25_smbell_large-wp_sdss_sm11p25_smbell_small)/wp_sdss_sm11p25_smbell
fracdiff_sm11p25_smbell_err = np.maximum(np.sqrt(wperr_sdss_sm11p25_smbell)/wp_sdss_sm11p25_smbell, cutoff_err)


wp_sdss_mpajhu_sm9p75 = np.load(os.path.join(output_dirname, 'sdss_mpajhu_sm9p75_wp.npy'))
wperr_sdss_mpajhu_sm9p75 = np.load(os.path.join(output_dirname, 'sdss_mpajhu_sm9p75_wperr.npy'))
wp_sdss_mpajhu_sm9p75_small = np.load(os.path.join(output_dirname, 'sdss_mpajhu_sm9p75_small_wp.npy'))
wp_sdss_mpajhu_sm9p75_large = np.load(os.path.join(output_dirname, 'sdss_mpajhu_sm9p75_large_wp.npy'))
fracdiff_sm9p75_mpajhu = (wp_sdss_mpajhu_sm9p75_large-wp_sdss_mpajhu_sm9p75_small)/wp_sdss_mpajhu_sm9p75
fracdiff_sm9p75_mpajhu_err = np.maximum(np.sqrt(wperr_sdss_mpajhu_sm9p75)/wp_sdss_mpajhu_sm9p75, cutoff_err)

wp_sdss_mpajhu_sm10p0 = np.load(os.path.join(output_dirname, 'sdss_mpajhu_sm10p0_wp.npy'))
wperr_sdss_mpajhu_sm10p0 = np.load(os.path.join(output_dirname, 'sdss_mpajhu_sm10p0_wperr.npy'))
wp_sdss_mpajhu_sm10p0_small = np.load(os.path.join(output_dirname, 'sdss_mpajhu_sm10p0_small_wp.npy'))
wp_sdss_mpajhu_sm10p0_large = np.load(os.path.join(output_dirname, 'sdss_mpajhu_sm10p0_large_wp.npy'))
fracdiff_sm10p0_mpajhu = (wp_sdss_mpajhu_sm10p0_large-wp_sdss_mpajhu_sm10p0_small)/wp_sdss_mpajhu_sm10p0
fracdiff_sm10p0_mpajhu_err = np.maximum(np.sqrt(wperr_sdss_mpajhu_sm10p0)/wp_sdss_mpajhu_sm10p0, cutoff_err)

wp_sdss_mpajhu_sm10p25 = np.load(os.path.join(output_dirname, 'sdss_mpajhu_sm10p25_wp.npy'))
wperr_sdss_mpajhu_sm10p25 = np.load(os.path.join(output_dirname, 'sdss_mpajhu_sm10p25_wperr.npy'))
wp_sdss_mpajhu_sm10p25_small = np.load(os.path.join(output_dirname, 'sdss_mpajhu_sm10p25_small_wp.npy'))
wp_sdss_mpajhu_sm10p25_large = np.load(os.path.join(output_dirname, 'sdss_mpajhu_sm10p25_large_wp.npy'))
fracdiff_sm10p25_mpajhu = (wp_sdss_mpajhu_sm10p25_large-wp_sdss_mpajhu_sm10p25_small)/wp_sdss_mpajhu_sm10p25
fracdiff_sm10p25_mpajhu_err = np.maximum(np.sqrt(wperr_sdss_mpajhu_sm10p25)/wp_sdss_mpajhu_sm10p25, cutoff_err)

wp_sdss_mpajhu_sm10p5 = np.load(os.path.join(output_dirname, 'sdss_mpajhu_sm10p5_wp.npy'))
wperr_sdss_mpajhu_sm10p5 = np.load(os.path.join(output_dirname, 'sdss_mpajhu_sm10p5_wperr.npy'))
wp_sdss_mpajhu_sm10p5_small = np.load(os.path.join(output_dirname, 'sdss_mpajhu_sm10p5_small_wp.npy'))
wp_sdss_mpajhu_sm10p5_large = np.load(os.path.join(output_dirname, 'sdss_mpajhu_sm10p5_large_wp.npy'))
fracdiff_sm10p5_mpajhu = (wp_sdss_mpajhu_sm10p5_large-wp_sdss_mpajhu_sm10p5_small)/wp_sdss_mpajhu_sm10p5
fracdiff_sm10p5_mpajhu_err = np.maximum(np.sqrt(wperr_sdss_mpajhu_sm10p5)/wp_sdss_mpajhu_sm10p5, cutoff_err)

wp_sdss_mpajhu_sm10p75 = np.load(os.path.join(output_dirname, 'sdss_mpajhu_sm10p75_wp.npy'))
wperr_sdss_mpajhu_sm10p75 = np.load(os.path.join(output_dirname, 'sdss_mpajhu_sm10p75_wperr.npy'))
wp_sdss_mpajhu_sm10p75_small = np.load(os.path.join(output_dirname, 'sdss_mpajhu_sm10p75_small_wp.npy'))
wp_sdss_mpajhu_sm10p75_large = np.load(os.path.join(output_dirname, 'sdss_mpajhu_sm10p75_large_wp.npy'))
fracdiff_sm10p75_mpajhu = (wp_sdss_mpajhu_sm10p75_large-wp_sdss_mpajhu_sm10p75_small)/wp_sdss_mpajhu_sm10p75
fracdiff_sm10p75_mpajhu_err = np.maximum(np.sqrt(wperr_sdss_mpajhu_sm10p75)/wp_sdss_mpajhu_sm10p75, cutoff_err)

wp_sdss_mpajhu_sm11p0 = np.load(os.path.join(output_dirname, 'sdss_mpajhu_sm11p0_wp.npy'))
wperr_sdss_mpajhu_sm11p0 = np.load(os.path.join(output_dirname, 'sdss_mpajhu_sm11p0_wperr.npy'))
wp_sdss_mpajhu_sm11p0_small = np.load(os.path.join(output_dirname, 'sdss_mpajhu_sm11p0_small_wp.npy'))
wp_sdss_mpajhu_sm11p0_large = np.load(os.path.join(output_dirname, 'sdss_mpajhu_sm11p0_large_wp.npy'))
fracdiff_sm11p0_mpajhu = (wp_sdss_mpajhu_sm11p0_large-wp_sdss_mpajhu_sm11p0_small)/wp_sdss_mpajhu_sm11p0
fracdiff_sm11p0_mpajhu_err = np.maximum(np.sqrt(wperr_sdss_mpajhu_sm11p0)/wp_sdss_mpajhu_sm11p0, cutoff_err)

wp_sdss_mpajhu_sm11p25 = np.load(os.path.join(output_dirname, 'sdss_mpajhu_sm11p25_wp.npy'))
wperr_sdss_mpajhu_sm11p25 = np.load(os.path.join(output_dirname, 'sdss_mpajhu_sm11p25_wperr.npy'))
wp_sdss_mpajhu_sm11p25_small = np.load(os.path.join(output_dirname, 'sdss_mpajhu_sm11p25_small_wp.npy'))
wp_sdss_mpajhu_sm11p25_large = np.load(os.path.join(output_dirname, 'sdss_mpajhu_sm11p25_large_wp.npy'))
fracdiff_sm11p25_mpajhu = (wp_sdss_mpajhu_sm11p25_large-wp_sdss_mpajhu_sm11p25_small)/wp_sdss_mpajhu_sm11p25
fracdiff_sm11p25_mpajhu_err = np.maximum(np.sqrt(wperr_sdss_mpajhu_sm11p25)/wp_sdss_mpajhu_sm11p25, cutoff_err)







