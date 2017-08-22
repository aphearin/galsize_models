"""
"""
import argparse
from time import time
from emcee import EnsembleSampler
import numpy as np
from galsize_models.models import mcmc_bulge_disk_double_power_law as mcmc


parser = argparse.ArgumentParser()
parser.add_argument("num_iteration", type=int,
    help="Approximate number of mcmc iterations")
parser.add_argument("-num_burnin", type=int, default=3,
    help="Number of steps per walker for the burn-in")

args = parser.parse_args()

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

outname = "ssfr_dependent_chain.dat"
