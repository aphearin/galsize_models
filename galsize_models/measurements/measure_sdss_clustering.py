"""
"""
import subprocess
import numpy as np
from astropy.table import Table


umachine_sdss_fname = "/Users/aphearin/Dropbox/UniverseMachine/data/sdss/dr10_sm_sfr_complete3.dat"
_orig_umachine_sdss = Table.read(umachine_sdss_fname, format='ascii.commented_header')


__all__ = ('write_umachine_ascii', )


def write_umachine_ascii(gals, fname, orig_keys=list(_orig_umachine_sdss.keys()), overwrite=False):
    output_table = Table()
    for key in orig_keys:
        output_table[key] = gals[key.lower()]
    output_table.write(fname, format='ascii.commented_header', overwrite=overwrite)


def get_wp_measurements(wp_sample_fname, sm_low, sm_high, pi_max):

    code_basename = "/Users/aphearin/work/UniverseMachine/code/UniverseMachine/correl/correl"

    command_prefix = code_basename + " " + wp_sample_fname + " "
    command_suffix = str(sm_low) + " " + str(sm_high) + " " + str(pi_max) + " 20 0.02 0.1 500000"
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
