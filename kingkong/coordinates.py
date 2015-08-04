# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
import astropy.units as u

__all__ = ['galactocentric_to_heliocentric', 'heliocentric_to_galactocentric',
           'cartesian_to_spherical', 'spherical_to_cartesian']

def galactocentric_to_heliocentric(X,
                                   rsun=8.*u.kpc,
                                   vcirc=220.*u.km/u.s,
                                   vlsr=[10., 5.25, 7.17]*u.km/u.s): # TODO: cite
    """
    """

    Xframe = np.zeros_like(X)
    Xframe[...,0] = rsun.to(u.kpc).value

    vlsr_kpc_Myr = vlsr.to(u.kpc/u.Myr).value
    Xframe[...,3] = -vlsr_kpc_Myr[0]
    Xframe[...,4] = -vlsr_kpc_Myr[1] - vcirc.to(u.kpc/u.Myr).value
    Xframe[...,5] = -vlsr_kpc_Myr[2]

    return X + Xframe

def heliocentric_to_galactocentric(X,
                                   rsun=8.*u.kpc,
                                   vcirc=220.*u.km/u.s,
                                   vlsr=[10., 5.25, 7.17]*u.km/u.s): # TODO: cite
    """
    """

    Xframe = np.zeros_like(X)
    Xframe[...,0] = rsun.to(u.kpc).value

    vlsr_kpc_Myr = vlsr.to(u.kpc/u.Myr).value
    Xframe[...,3] = -vlsr_kpc_Myr[0]
    Xframe[...,4] = -vlsr_kpc_Myr[1] - vcirc.to(u.kpc/u.Myr).value
    Xframe[...,5] = -vlsr_kpc_Myr[2]

    return X - Xframe

def cartesian_to_spherical(X):
    """
    """

    X = np.array(X)
    xyz = X[...,:3].T
    vxyz = X[...,3:].T

    # get out spherical components
    d = np.sqrt(np.sum(xyz**2, axis=0))
    dxy = np.sqrt(np.sum(xyz[:2]**2, axis=0))

    # sky position
    l = np.arctan2(xyz[1], xyz[0]) % (2*np.pi)
    b = np.arctan2(xyz[2], dxy)

    # radial velocity
    vr = np.sum(xyz * vxyz, axis=0) / d

    mu_l_cosb = (xyz[0]*vxyz[1] - vxyz[0]*xyz[1]) / dxy**2 * np.cos(b)
    mu_b = (xyz[2]*(xyz[0]*vxyz[0] + xyz[1]*vxyz[1]) - dxy**2*vxyz[2]) / d**2 / dxy

    Y = np.zeros_like(X)
    Y[...,0] = l.T
    Y[...,1] = b.T
    Y[...,2] = d.T
    Y[...,3] = mu_l_cosb.T
    Y[...,4] = mu_b.T
    Y[...,5] = vr.T

    return Y

def spherical_to_cartesian(Y):
    """
    """

    Y = np.array(Y)
    l,b,d = Y[...,:3].T
    mu_l_cosb,mu_b,vr = Y[...,3:].T

    vl = d*mu_l_cosb
    vb = d*mu_b

    # transform from spherical to cartesian
    x = d*np.cos(b)*np.cos(l)
    y = d*np.cos(b)*np.sin(l)
    z = d*np.sin(b)

    vx = vr*np.cos(l)*np.cos(b) - np.sin(l)*vl - np.cos(l)*np.sin(b)*vb
    vy = vr*np.sin(l)*np.cos(b) + np.cos(l)*vl - np.sin(l)*np.sin(b)*vb
    vz = vr*np.sin(b) + np.cos(b)*vb

    X = np.zeros_like(Y)
    X[...,0] = x
    X[...,1] = y
    X[...,2] = z
    X[...,3] = vx
    X[...,4] = vy
    X[...,5] = vz

    return X
