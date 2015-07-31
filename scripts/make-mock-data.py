# coding: utf-8

""" Generate mock data for testing KinematicConsensus """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# Third-party
from astropy import log as logger
import matplotlib.pyplot as pl
import numpy as np

# Project
project_path = "/Users/adrian/projects/kinematic-consensus/"
if project_path not in sys.path:
    sys.path.append(project_path)
from stroemefinden import potential, radial_periods
from stroemefinden.util import random_quaternion, quaternion_to_rotation_matrix

data_path = os.path.join(project_path, "data", "mock")
if not os.path.exists(data_path):
    os.makedirs(data_path)

def main(seed, n=1):
    np.random.seed(seed)
    logger.info("Seed: {0}".format(seed))
    logger.debug("Generating {0} mock streams".format(n))

    dt = 0.01 # the timestep for integration
    nperiods = 1.5 # number of periods to integrate for
    nstars = 16 # number of points to sample from each orbit / number of "stars" per "stream"

    # sample a random initial velocity
    v = np.random.uniform(0., 1., size=n)

    # create array of 6D initial conditions (3D position, 3D velocity)
    w0 = np.zeros((n,6))
    w0[:,0] = 1.
    w0[:,4] = v

    # sample a random rotation matrix
    q = random_quaternion(size=n)
    R = quaternion_to_rotation_matrix(q)

    # compute radial periods for all orbits
    periods = radial_periods(potential, w0)

    # container for all "streams"
    stream_w = np.zeros((n,nstars,6))
    for i in range(n):
        # integrate the orbits for some number of radial periods, starting from apocenter
        t,w = potential.integrate_orbit(w0[i], dt=dt, nsteps=int(nperiods*periods[i]/dt))
        ix = np.random.randint(w.shape[0], size=nstars)
        w = w[ix,0]
        # w = w[:,0]
        rotated_w = np.vstack((R[i].dot(w[:,:3].T), R[i].dot(w[:,3:].T))).T
        stream_w[i] = rotated_w

if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        default=False, help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                        default=False, help="Be quiet! (default = False)")

    parser.add_argument("--seed", dest="seed", type=int, default=100000,
                        help="Seed for the random number generator.")
    parser.add_argument("-n", dest="n", type=int, default=128,
                        help="Number of mock 'streams' to make.")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    main(seed=args.seed, n=args.n)
