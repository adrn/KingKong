# coding: utf-8

""" Generate mock data """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
from astropy import log as logger

# Project
from .core import potential, radial_periods
from .util import Quaternion

__all__ = ['mock_stream_generator']

class MockStream(object):

    def __init__(self, potential, w0):
        pass

    @property
    def quaternion


def mock_stream_generator(n, nstars_per_stream=16, nperiods=1.5, seed=42):
    """
    Generate a set of ``n`` mock streams.

    Parameters
    ----------
    n : int
    nstars_per_stream : int (optional)
        Number of points to sample from each orbit / number of "stars" per "stream"
    nperiods : numeric (optional)
        Number of periods to integrate for to generate a stream segment.
    seed : int (optional)
    """

    np.random.seed(seed)
    logger.info("Seed: {0}".format(seed))
    logger.debug("Generating {0} mock streams".format(n))

    nsteps_per_period = 128 # to determine the timestep for integration

    # sample a random initial velocity
    v = np.random.uniform(0., 1., size=n)

    # radial scale
    scale = np.random.uniform(0.1, 30., size=n)

    # create array of 6D initial conditions (3D position, 3D velocity)
    w0 = np.zeros((n,6))
    w0[:,0] = 1. * scale
    w0[:,4] = v

    # sample a random rotation matrix
    q = Quaternion.random(size=n)
    R = q.rotation_matrix

    # compute radial periods for all orbits
    periods = radial_periods(potential, w0)

    # container for all "streams"
    stream_w = np.zeros((n,nstars_per_stream,6))
    for i in range(n):
        # integrate the orbits for some number of radial periods, starting from apocenter
        dt = periods[i] / nsteps_per_period
        t,w = potential.integrate_orbit(w0[i], dt=dt, nsteps=int(nperiods*periods[i]/dt))
        ix = np.random.randint(w.shape[0], size=nstars_per_stream)
        w = w[ix,0]
        rotated_w = np.vstack((R[i].dot(w[:,:3].T), R[i].dot(w[:,3:].T))).T
        stream_w[i] = rotated_w

    return stream_w

def sample_isothermal():
    pass
