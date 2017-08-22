"""
"""
from astropy.table import Table
from time import time
from emcee import EnsembleSampler
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("num_iteration", type=int,
    help="Approximate number of mcmc iterations")
parser.add_argument("-num_burnin", type=int, default=3,
    help="Number of steps per walker for the burn-in")

args = parser.parse_args()

from galsize_models.models import mcmc_bulge_disk_double_power_law as mcmc


ndim = 5

nwalkers = 2*ndim
p0 = np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))
sampler = EnsembleSampler(nwalkers, ndim, mcmc.lnprob, args=mcmc.lnprob_args)

print("...Running burn-in phase of {0} total likelihood evaluations ".format(
    args.num_burnin*nwalkers))
start = time()
pos0, prob, state = sampler.run_mcmc(p0, args.num_burnin)
sampler.reset()
end = time()
print("Total runtime for burn-in = {0:.2f} seconds".format(end-start))

outname = "bulge_disk_power_law_chain.dat"


sep = "  "
formatter = sep.join("{"+str(i)+":.4f}" for i in range(pos0.shape[-1])) + "  " + "{"+str(pos0.shape[-1])+":.4f}\n"
header = "norm1, norm2, alpha1, alpha2, scatter, lnprob\n"

start = time()

print("...Running MCMC with {0} chain elements".format(args.num_iteration*nwalkers))
with open(outname, "wb") as f:
    f.write(header)
    for result in sampler.sample(pos0, iterations=args.num_iteration, storechain=False):
        pos, prob, state = result
        for a, b in zip(pos, prob):
            newline = formatter.format(*np.append(a, b))
            f.write(newline)
end = time()
print("Runtime for MCMC = {0:.2f} minutes".format((end-start)/60.))
print("\a\a\a")

chain = Table.read(outname, format='ascii')
print("Successfully loaded chain with {0} elements from disk after completion of MCMC".format(len(chain)))
