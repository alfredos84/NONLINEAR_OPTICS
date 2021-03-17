#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 19:03:27 2021
@author: Alfredo Sanchez
@email: Alfredo.Sanchez@icfo.eu

"""

import numpy as np
import numpy.fft as fft
# import matplotlib.pyplot as plt

# Constants
c           = 299792458*1e6/1e12 # speed of light in vac. [μm/ps]
eps0        = 8.8541878128e-12*1e12/1e6 # vacuum pertivity [W.ps/V²μm]
pi          = np.pi

# Functions declarations 



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

def filtro_triang_freq(f, win_size, spectrum):
    """This is a window function. It returns the filtered spectrum in order to
    simulate a win_size-THz-resolution spectrum analyzer.
    INPUTS
    f:   frequency in THz
    win_size: size of window in THz
    spectrum: spectrum vector
    
    OUTPUT
    Filtered spectrum"""
    
    inds = np.where((f>=-0.5*win_size)&(f<=0.5*win_size))
    fs = f[inds]
    N = len(fs)
    M = N-1
    triang_win = np.zeros([N])
    n = np.arange(N)
    triang_win[0:int(M/2)+1] = 2*n[0:int(M/2)+1]/M
    triang_win[int(M/2)+1:-1] = 2-2*n[int(M/2)+1:-1]/M
    triang_win/=np.sum(triang_win)
    convol = np.convolve(triang_win,spectrum)
    dif = convol.size - spectrum.size
    return convol[int(dif/2):convol.size-int(dif/2)-1]

def freq2cm1(f):
    """This function converts frecuency in THz and returns the corresponding
    wavenumber in cm^{-1}"""
    c = 299792458*1e6/1e12 # speed of light in vac. [μm/ps]
    return f/c*1e4

def cm12freq(wn):
    """This function converts wavenumber in cm^{-1} and returns the
    corresponding frecuency in THz.
    """
    c = 299792458*1e6/1e12 # speed of light in vac. [μm/ps]
    return c*1e-4*wn

def w2la(w):
    """This function converts angular frequency in THz and returns the
    corresponding wavelength in um.
    """
    c = 299792458*1e6/1e12 # speed of light in vac. [μm/ps]
    return 2*np.pi*c/w

def la2w(L):
    """This function converts wavelength in um and returns the
    corresponding angular frequency in THz.
    """
    c = 299792458*1e6/1e12 # speed of light in vac. [μm/ps]
    return 2*np.pi*c/L

def SPL(signal, idler, pump):
    """Evaluate the self-phase locking condition.
    ϕs+ϕi-ϕp-π/2=2πm
    OUTPUT:
    m = (ϕs+ϕi-ϕp-π/2)/2/π
    """
    return (np.angle(signal)+np.angle(idler)-np.angle(pump)-0.5*np.pi)/2/np.pi

def SIP(signal, idler):
    """Evaluate the difference of phase between signal and idler.
    OUTPUT:
    m = (ϕs-ϕi)/2/π
    """
    return (np.angle(signal)-np.angle(idler))/2/np.pi

def measure_interval(phase, low, high):
    av_phase = np.mean(phase)
    for i in range(len(phase)):
        if(phase[i]>= av_phase):
            phase[i] = +1
        else:
            phase[i] = -1
    i = 0
    while(i<len(phase)-1):
        # print("i={}".format(i))
        c = 0    
        while((phase[i]*phase[i+1]>=0) and (i<len(phase)-2)):
            c+=1
            i+=1
        if(phase[i]== 1):
            high.append(c)
        else:
            low.append(c)
        if(i<=len(phase)-2):
            i+=1
    return low, high


class Crystal:
    # Instance method
    def __init__(self, deff, length, refindx, group_vel, gvd, kind):
        self.deff = deff # eff. nonlinear suscept. [μm/V]
        self.length = length # crystal length [μm]
        self.refindx = refindx # refractive index
        self.group_vel = group_vel # GV [μm/ps]
        self.gvd = gvd # GVD in [ps²/μm]
        self.kind = kind # kind of nonlinear crystal

    # description
    def __str__(self):
        return "Nonlinear crystal used here: "+ str(self.kind) + ".\n eff. nonlinear susceptibility"+ str(self.deff) + " [μm/V]\n crystal length " + str(self.length) + " [μm]\n ref. index for p,s,i " + str(self.refindx) + "\n group vel. for p,s,i " + str(self.group_vel) + " [μm/ps]\n GVD for p,s,i" + str(self.gvd) + "[ps²/μm]\n\n"

class Cavity:
    def __init__(self, length, R, trt):
        self.length = length # cavity lengh [μm]
        self.R = R # reflectivity
        self.trt = trt # round-trip time [ps]
    # description
    def __str__(self):
        return "Cavity length of {} um.".format(self.length*1e-6)        

class Fields:
    def __init__(self):
        pass
    
    # Coupled equations for pump and signal waves
    def dAdz(self,kappas, dk, z, A):
        if(A.shape[0]==2):
            aux = np.zeros_like(A)
            aux[0,:]=kappas[0]*A[1,:]*A[1,:]*np.exp(-1j*dk*z)
            aux[1,:]=kappas[1]*A[0,:]*np.conjugate(A[1,:])*np.exp(1j*dk*z)
            return 1j*aux
        if(A.shape[0]==3):
            aux = np.zeros_like(A)
            aux[0,:]=kappas[0]*A[1,:]*A[2,:]*np.exp(-1j*dk*z)
            aux[1,:]=kappas[1]*A[0,:]*np.conjugate(A[2,:])*np.exp(1j*dk*z)
            aux[2,:]=kappas[2]*A[0,:]*np.conjugate(A[1,:])*np.exp(1j*dk*z)
            return 1j*aux
    
    def linear_operator(self,Aw ,z ,w ,vg ,gvd):
        vm = np.max(vg) # sets the maximum group-velocity
        if(Aw.shape[0]==2):
            aux = np.zeros_like(Aw)
            aux[0,:] = Aw[0,:]*np.exp(1j*z*(w*(1/vg[0]-1/vm)+0.5*w**2*gvd[0]))
            aux[1,:] = Aw[1,:]*np.exp(1j*z*(w*(1/vg[1]-1/vm)+0.5*w**2*gvd[1]))
        if(Aw.shape[0]==3):
            aux = np.zeros_like(Aw)
            aux[0,:] = Aw[0,:]*np.exp(1j*z*(w*(1/vg[0]-1/vm)+0.5*w**2*gvd[0]))
            aux[1,:] = Aw[1,:]*np.exp(1j*z*(w*(1/vg[1]-1/vm)+0.5*w**2*gvd[1]))
            aux[2,:] = Aw[2,:]*np.exp(1j*z*(w*(1/vg[2]-1/vm)+0.5*w**2*gvd[2]))            
        return aux
        
    
    def rk4(self,A, kappas, dk, z, dz):
        # Fourth-order Runge-Kutta for the nonlinear propagation in a step dz
        k1 = dz*self.dAdz(kappas,dk,z,A)
        k2 = dz*self.dAdz(kappas,dk,z+dz/2,A+k1/2)
        k3 = dz*self.dAdz(kappas,dk,z+dz/2,A+k2/2)
        k4 = dz*self.dAdz(kappas,dk,z+dz,A+k3)
        A = A+(k1+2*k2+2*k3+k4)/6        
        return A
    
    def evol_in_crystal(self, A, kappas, dk, dz, z, crystallength, w, vg, gvd):
        # Main loop
        while(z <= crystallength):
            # Nonlinear propagation over a step dz: exp(Ndz)*A 
            A = self.rk4(A, kappas, dk, z, dz/2)
            # Linear propagation over a half-step dz/2: IFFT(exp(Ldz/2)*FFT(A)) 
            Aw = self.linear_operator(fft.ifft(A,axis=1), dz, w, vg, gvd)
            # Nonlinear propagation over a step dz: exp(Ndz)*A 
            A = self.rk4(fft.fft(Aw,axis=1), kappas, dk, z, dz/2)
            z+=dz # step increment  
        return A    