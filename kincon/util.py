# coding: utf-8

""" Misc. utilities """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# Third-party
import numpy as np
from astropy import log as logger
import astropy.units as u

# Project
# ...

__all__ = ['']

def quaternion_to_rotation_matrix(q):
    """
    Convert a Quaternion, ``q``, to a rotation matrix.

    http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/

    Parameters
    ----------
    q : array_like
        A quaternion or array of quaternions. Assumes q=(w,x,y,z).

    Returns
    -------
    R : :class:`numpy.ndarray`
        A 3x3 rotation matrix, or an array of such matrices.
    """

    q = np.array(q)
    _initial_ndim = q.ndim
    q = np.atleast_2d(q)
    qw,qx,qy,qz = q.T

    # number of input quaternions
    n = q.shape[0]

    R = np.zeros((n,3,3))

    R[:,0,0] = 1 - 2*qy**2 - 2*qz**2
    R[:,0,1] = 2*qx*qy - 2*qz*qw
    R[:,0,2] = 2*qx*qz + 2*qy*qw

    R[:,1,0] = 2*qx*qy + 2*qz*qw
    R[:,1,1] = 1 - 2*qx**2 - 2*qz**2
    R[:,1,2] = 2*qy*qz - 2*qx*qw

    R[:,2,0] = 2*qx*qz - 2*qy*qw
    R[:,2,1] = 2*qy*qz + 2*qx*qw
    R[:,2,2] = 1 - 2*qx**2 - 2*qy**2

    if _initial_ndim == 1:
        return R[0]
    else:
        return R

def random_quaternion(size=1):
    """
    Randomly sample a Quaternion from a distribution uniform in
    3D rotation angles.

    https://www-preview.ri.cmu.edu/pub_files/pub4/kuffner_james_2004_1/kuffner_james_2004_1.pdf

    Parameters
    ----------
    size : int
        Number of quaternions to randomly sample.

    """

    s = np.random.uniform(size=size)
    s1 = np.sqrt(1 - s)
    s2 = np.sqrt(s)
    t1 = np.random.uniform(0, 2*np.pi, size=size)
    t2 = np.random.uniform(0, 2*np.pi, size=size)

    w = np.cos(t2)*s2
    x = np.sin(t1)*s1
    y = np.cos(t1)*s1
    z = np.sin(t2)*s2

    return np.vstack((w,x,y,z))
