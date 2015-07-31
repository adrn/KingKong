# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
import scipy.optimize as so
import scipy.integrate as si

# Project

__all__ = ['radial_periods']

def radial_periods(potential, w0):
    """
    Compute the radial periods for a given set of initial conditions, ``w0``,
    in the specified gravitational potential.

    Parameters
    ----------
    potential : :class:`gary.potential.PotentialBase`
        The gravitational potential.
    w0 : array_like
        A set of initial conditions to compute the radial periods for. If
        input is 1D, assumes it is a single set of initial conditions. If
        2D, ``axis=0`` is assumed to be the different sets of initial conditions,
        and ``axis=1`` are the phase-space dimensions.

    Returns
    -------
    periods : :class:`numpy.ndarray`
        An array of radial periods.
    """

    w0 = np.atleast_2d(w0)

    E = potential.total_energy(w0[:,:3], w0[:,3:])
    L = np.sqrt(np.sum(np.cross(w0[:,:3], w0[:,3:])**2, axis=-1))

    def func(p,E,L):
        r = p[0]
        return 2*(E-potential.value(np.array([r,0,0]))) - L**2/r**2

    pericenters = np.array([so.fsolve(func, x0=1E-3, args=(EE,LL))[0] for EE,LL in zip(E,L)])

    def integrand(r,E,L):
        r = np.atleast_1d(r)
        xyz = np.zeros((len(r),3))
        xyz[:,0] = r
        return 1/np.sqrt(2*(E-potential.value(xyz)) - L**2/r**2)

    periods = 2*np.array([si.quad(integrand, peri, 1., args=(EE, LL))[0]
                          for peri,EE,LL in zip(pericenters,E,L)])

    return periods
