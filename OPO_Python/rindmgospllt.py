#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 14:40:11 2021

@author: alfredo
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants
c           = 299792458*1e6/1e12 # speed of light in vac. [μm/ps]
pi          = np.pi

def n(L,T):
    """ This function returns the MgO:sPPLT extraordinary refractive index 
    from the Sellmeier Equation.
    Reference: Bruner et. al. Temperature-dependent Sellmeier equation for
    the refractive index of stoichiometric lithium tantalate
    
    INPUTS:
    L: wavelenght in um
    T: temperature in degrees
    
    OUTPUT:
    ne: refractive index as a funcion of wavelength"""
    

    A =  4.502483
    B =  0.007294
    C =  0.185087
    D =  -0.02357
    E =  0.073423
    F =  0.199595
    G =  0.001
    H =  7.99724
    b =  3.483933e-8 * (T + 273.15)**2
    c =  1.607839e-8 * (T + 273.15)**2
    return np.sqrt( A + (B+b)/(L**2-(C+c)**2) + E/(L**2-F**2) + G/(L**2-H**2)
                   + D*L**2)

def dndl(L, T):
    """Returns the first-order derivative of the refractive index respect to
    the wavelength dn/dλ."""
    A =  4.502483
    B =  0.007294
    C =  0.185087
    D =  -0.02357
    E =  0.073423
    F =  0.199595
    G =  0.001
    H =  7.99724
    b =  3.483933e-8 * (T + 273.15)**2
    c =  1.607839e-8 * (T + 273.15)**2
    return -L/n(L, T)*( (B+b)/(L**2-(C+c)**2)**2 + E/(L**2-F**2)**2 +
                       G/(L**2-H**2)**2 - D )

def d2ndl2(L, T):
    """Returns the second-order derivative of the refractive index respect to
    the wavelength d²n/dλ²."""
    A =  4.502483
    B =  0.007294
    C =  0.185087
    D =  -0.02357
    E =  0.073423
    F =  0.199595
    G =  0.001
    H =  7.99724
    b =  3.483933e-8 * (T + 273.15)**2
    c =  1.607839e-8 * (T + 273.15)**2
    S1 = dndl(L, T)/L
    S2 = 4*L**2/n(L,T)*((B+b)/(L**2-(C+c)**2)**3 + E/(L**2-F**2)**3 +
                       G/(L**2-H**2)**3  )
    return S1+S2

def group_vel(L, T):
    """ Returns the group-velocity vg(λ) = c/(n(λ)-λdn/dλ). """
    c  = 299792458*1e6/1e12 # speed of light in vac. [μm/ps]
    return c/(n(L,T)-L*dndl(L,T))

def GVD(L, T):
    """Returns the group-velocity β(λ)=λ^3/(2πc²)(d²n/dλ²)."""
    c  = 299792458*1e6/1e12 # speed of light in vac. [μm/ps]
    return L**3*d2ndl2(L, T)/(2*pi*c**2)

def plots(lmin, lmax, x, T):
    """ This funtion plots the refractive index, the group velocity and the 
    GVD as a function of the wavelength. 
    
    INTPUS:
    
    lmin, lmax: minimun and maximun wavelengths in microns (float)
    x         : list of wavelength to label in each plot (list)
    T         : temperature in Celsius degrees (float)
    
    OUTPUTS:
    
    Three plots for each quantity.    
    """
    c  = 299792458*1e6/1e12 # speed of light in vac. [μm/ps]
    L=np.linspace(lmin,lmax,1000)
    yn = n(np.array(x), T)
    yv = group_vel(np.array(x),T)/c
    yb = GVD(np.array(x), T)*1e9
    nn = yn.astype(float).tolist()
    nv = yv.astype(float).tolist()
    nb = yb.astype(int).tolist()
   
    fig, ax = plt.subplots()
    plt.title("Refractive index for MgO:sPPLT at T={:1d} °C".format(T))
    ax.plot(L,n(L, T))
    ax.scatter(x, yn,color="red")
    plt.xlim([lmin,lmax])
    plt.xlabel(r"$\lambda$ ($\mu$m)")
    plt.ylabel(r"n($\lambda$)")
    plt.grid()
    for i, txt in enumerate(nn):
        ax.annotate("{:.2f}".format(txt), (x[i]*1.03, yn[i]))
        
    fig, ax = plt.subplots()
    plt.title("Group-velocity for MgO:sPPLT at T={:1d} °C".format(T))
    ax.plot(L,group_vel(L, T)/c)
    ax.scatter(x, yv,color="red")
    plt.xlim([lmin,lmax])
    # plt.ylim([0.37,0.4615])
    # plt.yticks([0.37,0.38,0.39,0.40,0.41,0.42,0.43,0.44,0.45,0.46])
    plt.xlabel(r"$\lambda$ ($\mu$m)")
    plt.ylabel(r"$\nu_g$/c")
    plt.grid()
    for i, txt in enumerate(nv):
        ax.annotate("{:.3f}".format(txt), (x[i]*1.03, yv[i]*0.99))
    
    fig, ax = plt.subplots()
    ax.plot(L,GVD(L, T)*1e9)
    ax.scatter(x, yb,color="red")
    plt.title("GVD for MgO:sPPLT at T={:1d} °C".format(T))
    plt.xlabel(r"$\lambda$ ($\mu$m)")
    plt.ylabel(r"$\beta_2$ (fs$^2$/mm)")
    plt.xlim([lmin,lmax])
    # plt.ylim([-4000,1000])
    plt.grid()
    for i, txt in enumerate(nb):
        ax.annotate(txt, (x[i]*1.03, yb[i]))
        
        
plt.close("all")        
lmin,lmax = .4, 3.9
x  = np.array([1.064,2*1.064])
plots(lmin, lmax, x, 21)        