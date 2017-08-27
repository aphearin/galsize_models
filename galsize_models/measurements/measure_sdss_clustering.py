"""
"""
import os
import subprocess
import numpy as np
from scipy.stats import binned_statistic
from astropy.table import Table


umachine_sdss_fname = "/Users/aphearin/Dropbox/UniverseMachine/data/sdss/dr10_sm_sfr_complete3.dat"
_orig_umachine_sdss = Table.read(umachine_sdss_fname, format='ascii.commented_header')


__all__ = ('write_umachine_ascii', 'measure_wp', 'below_median_size_mask')


def _copy_auto_regions_data_file(dirname_to_sample_ascii):
    auto_regions_basename = 'auto_regions.txt'
    auto_regions_dirname = "/Users/aphearin/Dropbox/UniverseMachine/data/sdss"

    auto_regions_fname1 = os.path.join(dirname_to_sample_ascii, auto_regions_basename)
    auto_regions_fname2 = os.path.join(auto_regions_dirname, auto_regions_basename)

    if os.path.isfile(auto_regions_fname1):
        pass
    else:
        if os.path.isfile(auto_regions_fname2):
            command = "cp {0} {1}"
            os.system(command.format(auto_regions_fname2, auto_regions_fname1))
        else:
            raise IOError("The ``auto_regions.txt`` file could not be found in your current directory,\n"
                "nor in the Dropbox location {1}".format(auto_regions_fname2))


def write_umachine_ascii(gals, fname, orig_keys=list(_orig_umachine_sdss.keys()), overwrite=False):
    output_table = Table()
    for key in orig_keys:
        output_table[key] = gals[key.lower()]
    output_table.write(fname, format='ascii.commented_header', overwrite=overwrite)


def get_wp_measurements(wp_sample_fname, sm_low, sm_high, pi_max, num_randoms=int(2e5)):

    _copy_auto_regions_data_file(os.path.dirname(wp_sample_fname))

    num_randoms_string = str(int(num_randoms))

    code_basename = "/Users/aphearin/work/UniverseMachine/code/UniverseMachine/correl/correl"

    command_prefix = code_basename + " " + wp_sample_fname + " "
    command_suffix = str(sm_low) + " " + str(sm_high) + " " + str(pi_max) + " 25 0.02 0.1 " + num_randoms_string
    command = command_prefix + command_suffix

    raw_result = subprocess.check_output(command, shell=True)

    result = raw_result.split("\n")

    while True:
        try:
            result.remove('')
        except:
            break

    data_strings = [line.split() for line in result if line[0] != '#']

    result = np.array(data_strings, dtype='f4')
    # rp, wp, wperr = result[:, 0], result[:, 1], result[:, 2]
    return result


def measure_wp(sample, sm_low, sm_high, dirname, rp_bins=np.logspace(-1, 1.25, 25), pi_max=20,
            auto_regions_dirname="/Users/aphearin/Dropbox/UniverseMachine/data/sdss"):
    temp_fname = os.path.join(dirname, 'temp.dat')

    _copy_auto_regions_data_file(dirname)

    __ = write_umachine_ascii(sample, temp_fname, overwrite=True)
    result = get_wp_measurements(temp_fname, sm_low, sm_high, pi_max)
    rp, wp, wperr = result[:, 0], result[:, 1], result[:, 2]
    one_plus_wp_interp = np.exp(np.interp(np.log(rp_bins), np.log(rp), np.log(1. + wp)))
    wperr_interp = np.exp(np.interp(np.log(rp_bins), np.log(rp), np.log(wperr)))
    return rp_bins, one_plus_wp_interp - 1., wperr_interp


def below_median_size_mask(log10sm, rhalf):
    median_log10_size, log10sm_bins, __ = binned_statistic(log10sm, np.log10(rhalf), statistic='median')
    logsm_mids = 0.5*(log10sm_bins[:-1] + log10sm_bins[1:])
    return rhalf < 10**np.interp(log10sm, logsm_mids, median_log10_size)


def save_wp_measurement(sample, output_dirname, output_basename, sm_low, sm_high, pi_max=20.):
    """
    """
    assert len(sample) > 900, "Sample size = {0}.\nCan only measure clustering for at least 900 galaxies".format(len(sample))

    msg = "Measuring wp for {0} sample of {1} galaxies"
    print(msg.format(output_basename, len(sample)))

    temp_ascii_fname = os.path.join(output_dirname, 'tmp.dat')
    write_umachine_ascii(sample, temp_ascii_fname, overwrite=True)

    _results = get_wp_measurements(temp_ascii_fname, sm_low, sm_high, pi_max)
    rp, wp, _wperr = _results[:, 0], _results[:, 1], _results[:, 2]
    rp_fname = os.path.join(output_dirname, output_basename+'_rp')
    wp_fname = os.path.join(output_dirname, output_basename+'_wp')
    np.save(wp_fname, wp)
    np.save(rp_fname, rp)

