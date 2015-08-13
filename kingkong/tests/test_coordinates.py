# coding: utf-8

""" Test coordinates stuff  """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import matplotlib.pyplot as pl
import numpy as np

# Project
from ..coordinates import cartesian_to_spherical, spherical_to_cartesian

n = 128

def test_round_trip_car_sph():

    X = np.random.uniform(-10,10,size=(n,6))
    sph = cartesian_to_spherical(X)
    X2 = spherical_to_cartesian(sph)
    np.testing.assert_allclose(X, X2)

def test_round_trip_sph_car():
    sph = np.zeros((n,6))
    sph[:,0] = np.random.uniform(0,2*np.pi,size=n)
    sph[:,1] = np.random.uniform(-np.pi/2.,np.pi/2.,size=n)
    sph[:,2] = np.random.uniform(0,30,size=n)
    sph[:,3:5] = np.random.normal(1.,1.,size=(n,2)) / 1000.
    sph[:,5] = np.random.normal(0.,0.1,size=n)
    X = spherical_to_cartesian(sph)
    sph2 = cartesian_to_spherical(X)
    np.testing.assert_allclose(sph, sph2)
