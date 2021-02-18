#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 19:03:27 2021
@author: Alfredo Sanchez
@email: Alfredo.Sanchez@icfo.eu

"""

import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt

# Functions declarations 

# Coupled equations for pump and signal waves
def dAsdz(kappa_s,dk,z,As,Ap):
    """ This functions is just the z-evolution for signal wave.
    INPUTS:
    Awj    :  field in frequency domain (j=pump, signal)
    dz     :  half-step size un z direction
    dk     :  mismatch
    kappa_s:  coupling defined as κj=ωj x deff / nj / c	 
    
    OUTPUT:
    dAs/dz = j x κs x Ap x Ai* x exp^{j x Δk x z}
    """
    return 1j*kappa_s*Ap*np.conjugate(As)*np.exp(1j*dk*z)

def dApdz(kappa_p,dk,z,As,Ap):
    """ This functions is just the z-evolution for pump wave.
    INPUTS:
    Awj    :  field in frequency domain (j=pump, signal)
    dz     :  half-step size un z direction
    dk     :  mismatch
    kappa_p:  coupling defined as κj=ωj x deff / nj / c	 
    
    OUTPUT:
    dAp/dz = j x κp x As x As x exp^{-j x Δk x z}
    """
    return 1j*kappa_p*As*As*np.exp(-1j*dk*z)

def linear_operator(Awj,z,w,vj,vm,beta_j):
    """ Operator for linear propagagion
    INPUTS
    Awj   :  field in frequency domain (j=pump, signal)
    z     :  propagated distance in z direction
    w     :  angular frequency detuning (Δω)
    vm    :  the fastest group velocity (GV) for the transformed system 
             t'=t-z/vm.
    vj    :  the GV for the field j
    beta_j:  the group-velocity dispersion for the field j (βj)
    
    OUTPUT
    this function returns the field with an accumlated linear phase in a 
    distance z: 
    φ = [-Δω(1/vj-1/vm)+Δω²βj] x z
    Aj -> Aj x exp{iφ}
    """
    return Awj*np.exp(1j*z*(w*(1/vj-1/vm)+0.5*w**2*beta_j))
    

def rk4(As, Ap, kappa_s, kappa_p, dk, z, dz):
    """This function return the nonlinear half-step using the Runge-Kutta
    method.
    INPUTS
    As,Ai,Ap: signal, idler, and pump vectors.
    κs,κi,κp: signal, idler, and pump coupling
    Δk: mismatch factor (usually equal to zero)
    z, dz: lenght and half-step size
    
    OUTPUTS
    As,Ai,Ap after passing a half-step size.
    """
    # Fourth-order Runge-Kutta for the nonlinear propagation in a half-step dz
    k1 = dz*dAsdz(kappa_s,dk,z,As,Ap)
    m1 = dz*dApdz(kappa_p,dk,z,As,Ap)
    k2 = dz*dAsdz(kappa_s,dk,z+dz/2,As+k1/2,Ap+m1/2)
    m2 = dz*dApdz(kappa_p,dk,z+dz/2,As+k1/2,Ap+m1/2)
    k3 = dz*dAsdz(kappa_s,dk,z+dz/2,As+k2/2,Ap+m2/2)
    m3 = dz*dApdz(kappa_p,dk,z+dz/2,As+k2/2,Ap+m2/2)
    k4 = dz*dAsdz(kappa_s,dk,z+dz,As+k3,Ap+m3)
    m4 = dz*dApdz(kappa_p,dk,z+dz,As+k3,Ap+m3)
    As = As+(k1+2*k2+2*k3+k4)/6
    Ap = Ap+(m1+2*m2+2*m3+m4)/6
    return As, Ap

def evol_in_crystal(As, Ap, kappa_s, kappa_p, dk, z, dz, Lc, w, vp, vs,
                    beta_p, beta_s):
    """ This function performs the SSFM in order to get the evolution across
    nonlinear crystal.
    
    The employed scheme for fields at z+dz is:
    A(z+dz) = exp(N*dz/2) * exp(L*dz) * exp(N*dz/2) * A(z)
               Nonlinear  -  Linear   -  Nonlinear
                   
    RK4 method is applied in for a half-step in exp(N*dz/2)*A(z).
    In the frequency domain, the fields evolve as exp(L*dz)*A(z) along the 
    step dz. 
    Proccess finishes when z=Lc (crystal lenght).
    """    
    # Main loop inside crystal
    # i=0
    vm = np.max([ vp, vs]) # sets the maximum group-velocity
    while(z <= Lc):
        # Nonlinear propagation over a step dz: exp(Ndz)*A 
        As, Ap = rk4(As, Ap, kappa_s, kappa_p, dk, z, dz/2)
        # Linear propagation over a half-step dz/2: IFFT(exp(Ldz/2)*FFT(A)) 
        Asw = linear_operator(fft.ifft(As), dz, w, vs, vm, beta_s)
        Apw = linear_operator(fft.ifft(Ap), dz, w, vp, vm, beta_p)
        # Nonlinear propagation over a step dz: exp(Ndz)*A 
        As, Ap = rk4(fft.fft(Asw), fft.fft(Apw), kappa_s, kappa_p, dk, z, dz/2)
        z+=dz # step increment  
    return As, Ap

def filtro_triang(wn, win_size, spectrum):
    """This is a window function. It returns the filtered spectrum in order to
    simulate a 5-cm^{-1}-resolution spectrum analyzer.
    INPUTS
    wn:   wavenumber in cm^{-1}
    win_size: size of window in in cm^{-1}
    spectrum: spectrum vector
    
    OUTPUT
    Filtered spectrum"""
    
    inds = np.where((wn>=-0.5*win_size)&(wn<=0.5*win_size))
    wns = wn[inds]
    N = len(wns)
    M = N-1
    triang_win = np.zeros([N])
    n = np.arange(N)
    triang_win[0:int(M/2)+1] = 2*n[0:int(M/2)+1]/M
    triang_win[int(M/2)+1:-1] = 2-2*n[int(M/2)+1:-1]/M
    triang_win/=np.sum(triang_win)
    convol = np.convolve(triang_win,spectrum)
    dif = convol.size - spectrum.size
    return convol[int(dif/2):convol.size-int(dif/2)-1]

def freq2cm1(f,c):
    """This function converts frecuency in THz and returns the corresponding
    wavenumber in cm^{-1}"""
    return f/c*1e4

def cm12freq(wn,c):
    """This function converts wavenumber in cm^{-1} and returns the
    corresponding frecuency in THz.
    """
    return c*1e-4*wn

def w2la(w,c):
    """This function converts angular frequency in THz and returns the
    corresponding wavelength in um.
    """
    return 2*np.pi*c/w

def la2w(L,c):
    """This function converts wavelength in um and returns the
    corresponding angular frequency in THz.
    """
    return 2*np.pi*c/L
