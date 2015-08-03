# coding: utf-8

""" Generate mock data """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
import gary.dynamics as gd

# Project
from .core import potential, radial_periods
from .util import Quaternion

__all__ = ['MockStream']

class MockStream(object):

    """
    A trivial mock tidal stream.

    Parameters
    ----------
    r0 : numeric
        Initial radial distance.
    v0 : numeric (optional)
        Velocity, expressed as a fraction of the circular velocity.
    quaternion : Quaternion (optional)
    nperiods : numeric (optional)
    t0 : numeric (optional)
    nsteps_per_period : int (optional)

    """
    def __init__(self, r0, v0=None,
                 quaternion=None,
                 nperiods=1.5, t0=0., nsteps_per_period=128):

        self.r0 = float(r0)
        self.v0 = float(v0)
        self._w0 = np.array([self.r0,0,0, 0,self.v0,0])

        if quaternion is None:
            # sample a random rotation matrix using Quaternions, if none provided
            quaternion = Quaternion.random()
        self.quaternion = quaternion

        if t0 != 0.:
            raise NotImplementedError()

        # compute radial periods for orbit
        periods = radial_periods(potential, self._w0)
        dt = periods[0] / nsteps_per_period

        # integrate the orbit
        t,w = potential.integrate_orbit(self._w0, dt=dt, nsteps=int(nperiods*periods[0]/dt))
        w = w[:,0]

        # get rotation matrix from quaternion
        R = self.quaternion.rotation_matrix
        self.X = np.vstack((R.dot(w[:,:3].T), R.dot(w[:,3:].T))).T

    def plot(self, **kwargs):
        """
        TODO:
        """
        if 'ls' not in kwargs and 'linestyle' not in kwargs:
            kwargs['linestyle'] = 'none'

        if 'marker' not in kwargs:
            kwargs['marker'] = '.'

        fig = gd.plot_orbits(self.X, **kwargs)
        return fig

def sample_isothermal():
    pass
