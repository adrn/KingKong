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

    q = np.atleast_2d(q)

    # number of dimensions
    k = q.shape[-1]
