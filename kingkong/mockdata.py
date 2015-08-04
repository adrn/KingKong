# coding: utf-8

""" Generate mock data """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
import gary.dynamics as gd
from scipy.optimize.slsqp import approx_jacobian

# Project
from .core import potential, radial_periods
from .util import Quaternion
from .coordinates import galactocentric_to_heliocentric, cartesian_to_spherical

__all__ = ['MockStream', 'Data']

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
                 phi0=0., nsteps_per_period=128):

        self.r0 = float(r0)
        self.v0 = float(v0)
        self._w0 = np.array([self.r0,0,0, 0,self.v0,0])

        if quaternion is None:
            # sample a random rotation matrix using Quaternions, if none provided
            quaternion = Quaternion.random()
        self.quaternion = quaternion

        # compute radial periods for orbit
        period = radial_periods(potential, self._w0)[0]
        dt = period / nsteps_per_period

        # integrate the orbit
        nperiods = 2.
        t,w = potential.integrate_orbit(self._w0, dt=dt, nsteps=int(nperiods*period/dt))

        t1 = phi0*period / (2*np.pi)
        t2 = t1 + period
        ix1 = np.abs(t-t1).argmin()
        ix2 = np.abs(t-t2).argmin()
        w = w[ix1:ix2,0]

        # get rotation matrix from quaternion
        R = self.quaternion.rotation_matrix
        self.X = np.vstack((R.dot(w[:,:3].T), R.dot(w[:,3:].T))).T
        self.Y = cartesian_to_spherical(galactocentric_to_heliocentric(self.X))
        self.K = len(self.X)

    def observed_variances(self, x_spread=0.01, v_spread=0.001):
        """
        Spread in position = 10 pc, spread in velocity = 1 km/s.
        """
        VX = np.array([x_spread,x_spread,x_spread,
                       v_spread,v_spread,v_spread])

        func = lambda x: cartesian_to_spherical(galactocentric_to_heliocentric(x))

        VY = np.zeros_like(self.Y)
        for k in range(self.K):
            J = approx_jacobian(self.X[k], func, 1E-4)
            cov = np.diag(VX)
            VY[k] = np.diag(J.dot(cov).dot(J.T))

        return VY

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

    def compute_statistic(self, streamdata, **kwargs):
        """
        TODO:

        kwargs passed through to observed_variances()
        """

        Y = self.Y
        Y_obs = streamdata.Y

        VY = self.observed_variances(**kwargs)
        VY_obs = streamdata.VY

        chisq_nk = np.sum(((Y[None] - Y_obs[:,None])**2.) / (VY[None] + VY_obs[:,None]), axis=-1)

        return chisq_nk

class Data(object):

    def __init__(self, Y, VY):
        self.Y = Y
        self.VY = VY
