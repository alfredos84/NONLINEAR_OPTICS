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
def dAsdz(Ap,As,z,dk,kappa_s):
    """ This functions is just the z-evolution for signal wave.
    INPUTS:
    Awj    :  field in frequency domain (j=pump, signal)
    dz     :  half-step size un z direction
    dk     :  mismatch
    kappa_s:  coupling defined as κj=ωj x deff / nj / c	 
    
    OUTPUT:
    dAs/dz = j x κs x Ap x As* x exp^{j x Δk x z}
    """
    return 1j*kappa_s*Ap*np.conjugate(As)*np.exp(1j*dk*z)

def dApdz(Ap,As,z,dk,kappa_p):
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
    return Awj*np.exp(1j*z*(-w*(1/vj-1/vm)+w**2*beta_j))

def rk4(As, Ap, kappa_s, kappa_p, dk, z, dz):
    # Fourth-order Runge-Kutta for the nonlinear propagation in a half-step dz
        s1 = dz * dAsdz(Ap     ,As     ,z     ,dk,kappa_s)
        p1 = dz * dApdz(Ap     ,As     ,z     ,dk,kappa_p)       
        s2 = dz * dAsdz(Ap+p1/2,As+s1/2,z+dz/2,dk,kappa_s)
        p2 = dz * dApdz(Ap+p1/2,As+s1/2,z+dz/2,dk,kappa_p)       
        s3 = dz * dAsdz(Ap+p2/2,As+s2/2,z+dz/2,dk,kappa_s)
        p3 = dz * dApdz(Ap+p2/2,As+s2/2,z+dz/2,dk,kappa_p)       
        s4 = dz * dAsdz(Ap+p3  ,As+s3  ,z+dz  ,dk,kappa_s)
        p4 = dz * dApdz(Ap+p3  ,As+s3  ,z+dz  ,dk,kappa_p)   
        As = As + (s1 + 2*s2 + 2*s3 + s4 )/6
        Ap = Ap + (p1 + 2*p2 + 2*p3 + p4 )/6
        return As, Ap
    
def evol_in_crystal(As, Ap, kappa_s, kappa_p, dk, z, dz, Lc, w, vp, vs,
                    beta_p, beta_s):
    """ This function performs the SSFM in order to get the evolution across
    nonlinear crystal.
    
    Firsly, field in frequecy domain evolve by adquiring the linear phase 
    along the half-step dz. Finally, RK4 method is applied in the next half-
    step. Proccess occurs until z=Lc (crystal lenght).
    """    
    # Main loop inside crystal
    # i=0
    while(z <= Lc):
        # Linear propagation
        # print("Iteracion #  {:1}".format(i))
        Asw = fft.fft(As)
        Asw = linear_operator(Asw, dz, w, vs, vs, beta_s)
        Apw = fft.fft(Ap)
        Apw = linear_operator(Apw, dz, w, vp, vs, beta_p)
        z+=dz # half-step increment    
        # Nonlinear propagation
        As = fft.ifft(Asw)
        Ap = fft.ifft(Apw)
        As, Ap = rk4(As, Ap, kappa_s, kappa_p, dk, z, dz)
        z+=dz # half-step increment
        # i+=1
    return As, Ap 

def filtrado(signal, n_new, extra_points):
    m,n = signal.shape
    sig_output = np.zeros([m,n_new], dtype='complex')
    x = np.arange(0,n)
    win_centrals = np.tanh(0.3*(x-extra_points)-np.tanh(0.3*(x-(n-extra_points))))+1
    win_centrals = win_centrals/np.max(win_centrals)
    win_left = 0.5*(np.tanh(-0.3*(x-n_new))+1)
    win_right= 0.5*(np.tanh(0.3*(x-20))+1)
    for i in range(0,m):
        if(i==0):
            signal[i,:]*=win_left
            sig_output[i,:]=signal[i,0:n_new]
        if(i==m-1):
            signal[i,:]*=win_right
            sig_output[i,:]=signal[i,extra_points-1:-1]
        else:
            signal[i,:]*=win_centrals
            sig_output[i,:] = signal[i,extra_points//2:n-extra_points//2]
    return sig_output

def freq2cm1(f,c):
    return f/c*1e4

def filtro_triang(wn, win_size, spectrum):
    inds = np.where((wn>=-0.5*win_size)&(wn<=0.5*win_size))
    wns = wn[inds]
    N = len(wns)
    M = N-1
    triang_win = np.zeros([N])
    n = np.arange(N)
    triang_win[0:int(M/2)+1] = 2*n[0:int(M/2)+1]/M
    triang_win[int(M/2)+1:-1] = 2-2*n[int(M/2)+1:-1]/M
    triang_win/=np.sum(triang_win)
    # plt.figure()
    # plt.plot(wns,triang_win)
    return np.convolve(triang_win,spectrum)