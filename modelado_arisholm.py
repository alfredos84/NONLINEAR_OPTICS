#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 19:03:27 2021
@author: Alfredo Sanchez
@email: Alfredo.Sanchez@icfo.eu

This code models the light propagation in a degenerate type-I nanosecond OPO,
based on the results published in "Degenerate type I nanosecond optical
parametric oscillators", Smith et al. (2003).
Essentially, this code solves the well-known coupled equations in a nonlinear 
crystal in order to obtain both temporal and spectral outputs.

This code calls the file "evolution.py" where are some functions used here.
"""

 # Imports
import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
import evolution as ev
# import pickle


plt.close("all")

# Constants
c           = 299792458*1e6/1e12 # speed of light in vac. [μm/ps]
h           = 6.626e-34*1e12 # plank constant [J.ps]
pi          = np.pi
# Define the time- and frequency-grid
lp          = .532   # pump wavelenght [μm]
ls          = 2*lp   # signal wavelenght [μm]
fp, wp      = c/lp, 2*pi*c/lp # pump frequency [THz]
fs, ws      = c/ls, 2*pi*c/ls # signal frequency [THz]
# Crystal
n_signal    = 1.654   # refractive ind. at signal freq
n_pump      = 1.654   # refractive ind. at pump freq
deff        = 2.0e-6  # effective nonlinear suscept. [μm/V]
Lc          = 5e3 # crystal length [μm]
Lcav        = 20e3 # cavity lengh [μm]
DZ          = Lc/50 # step size in z-crystal
dz          = 0.5*DZ  # half-step size in z-crystal
kappa_p     = 2*pi*deff/n_pump/lp # [1/V]
kappa_s     = 2*pi*deff/n_signal/ls # [1/V]
dk          = 0 # mismatch
vs,vp       = c/1.67, c/1.70  # signal and pump GV [μm/ps]
beta_s      = 2*2.07e-7 # signal GVD in [ps²/μm]
beta_p      = 2*3.22e-7 # pump GVD in [ps²/μm]
dw_bw       = np.sqrt(pi/Lc/beta_s) # bandwidth acceptance [THz]
t_rt        = (Lcav-Lc)/c + Lc/np.max([vs,vp]) # round-trip time [ps]
N_rt        = 100 # number of round trips to cover the input pulse
T_width     = N_rt*t_rt # total time for input ns-pulse
N_ps        = 2**12 # number of points per time slice
extra_win   = 16 # extra points for short-time slices
dT          = t_rt/N_ps # time step in [ps]
dF          = 1/t_rt # frequency step in [THz]
T           = (np.linspace(-N_ps/2, N_ps/2-1, N_ps))*dT # temp. grid for slice
Tp          = (np.linspace(-N_ps*N_rt/2, N_ps*N_rt/2-1,
                           N_ps*N_rt))*dT # temp. grid for pulse
dF_p        = 1/T_width
f_p         = (np.linspace(-Tp.size/2,Tp.size/2-1,
                           Tp.size))*dF_p # freq. grid [THz]
w_p         = fft.fftshift(2*pi*f_p)  # angular frequency in [2*pi*THz]
f           = (np.linspace(-N_ps/2,N_ps/2-1,N_ps))*dF # freq. grid [THz]
w           = fft.fftshift(2*pi*f)  # angular frequency in [2*pi*THz]
f_ext       = (np.linspace(-(N_ps+extra_win)/2,(N_ps+extra_win)/2-1,
                           (N_ps+extra_win)))*dF # extended freq. grid [THz]
w_ext       = fft.fftshift(2*pi*f_ext)  # ext. angular frequency in [2*pi*THz]


for p in [7,8]:
    for delta in [-0.1,0,0.1]:

        # Input fields
        # Pump
        FWHM        = 2000 # input pump with fwhm [ps]
        Ener_pulse  = 0.4 # energy pulse [J]
        Peaw_P      = 1*Ener_pulse/FWHM*10**p # peak power [W]
        Ap0         = np.sqrt(Peaw_P) # input field amplitud [W^1/2] 
        sigmap      = FWHM/2/np.sqrt(2*np.log(2)) # standar deviation [ps]
        Ap_total    = Ap0*np.exp(-(Tp/2/sigmap)**2)*np.exp(-2*1j*pi*0*Tp) # pump field
        
        # Signal
        Signal = np.zeros([N_rt,N_ps+extra_win],dtype='complex')
        mu, sigma = 0, 1e-15 # for generating a signal noisy vector
        
        nr = 40 # number of realizations
        spectrum = np.zeros([nr,len(Tp)], dtype="complex")
        
        Signal = np.zeros([N_rt,N_ps+extra_win],dtype='complex')
        Pump   = np.zeros([N_rt,N_ps+extra_win],dtype='complex')
        
        # This is the main loop for nr noise realizations and for the N_rt 
        # round trips in which the input pump pulse was splitted
        for j in range(0,nr):
            As_in = np.random.normal(mu,sigma,N_ps+extra_win)+1j*np.random.normal(mu,sigma, N_ps+extra_win)
            # As_in = ev.noise(f_ext, h, fs)
            As    = As_in
            for i in range(0,N_rt):
                if(i%10==0):
                    print("Pow={:1} - delta={:2} - Realization {:1} - Round trip #: {:1}".format(p,delta, j,i))
                # here the time-short slice is selected with extra points
                if(i==0):
                    Ap = Ap_total[0:N_ps+extra_win]        
                elif(i>0 and i<N_rt-2):
                    Ap = Ap_total[i*N_ps-extra_win//2:(i+1)*N_ps+extra_win//2]
                if(i==N_rt-1):
                    Ap = Ap_total[i*N_ps-extra_win:(i+1)*N_ps]
                
                # Propagation in the nonlinear crystal
                z=0 # is the starting point of crystal
                As, Ap = ev.evol_in_crystal(As, Ap, kappa_s, kappa_p, dk, z, dz, Lc, 
                                        w_ext, vp, vs, beta_p, beta_s)
                # intra-cavity phase accumulation exp(jmπ+/-jδ)
                As=np.sqrt(0.7)*As*np.exp(1j*pi*(i+delta))
                Signal[i,:] = As # save the signal nth-round-trip
                
            Signal_output = Signal[:,0:N_ps]
            spectrum[j,:] = fft.fft(Signal_output.reshape(Signal_output.size))
        
        wn = ev.freq2cm1(f_p,c)
        
        # Post-processing and plotting
        spectrum_av = np.mean(spectrum,axis=0)    
        irrad_av = np.abs(fft.fftshift(spectrum_av))**2
        
        irrad = np.zeros(spectrum.shape)
        
        win_size = 5 # window size in cm^-1
        irrad_av_filt = ev.filtro_triang(wn, win_size, irrad_av)
            
        irrad_filt = np.zeros([nr,len(irrad_av_filt)])
        for i in range(nr):
            irrad[i,:] = np.abs(spectrum[i,:])**2
            irrad_filt[i,:] = ev.filtro_triang(wn, win_size, fft.fftshift(irrad[i,:]))
        
        irrad_av = np.mean(fft.fftshift(irrad,axes=1),axis=0)    
        irrad_filt_av = np.mean(irrad_filt,axis=0)
        
        inds = np.where((wn>-200)&(wn<200))    
        plt.figure()
        plt.plot(wn[inds],irrad_av[inds]/np.max(irrad_av[inds]),':r', linewidth=1)
        plt.plot(wn[inds],irrad_filt_av[inds]/np.max(irrad_filt_av[inds]), 'b', linewidth=2)
        plt.hlines(1,-200,200,colors="blue",linestyles="dashed")
        plt.xlabel(r"$\nu$ (cm$^{-1}$)")
        plt.title(r"Filtered, and then averaged.  Pow=10$^{:1}$ - delta={:2}".format(p,delta))
        
        for be in range(10,91,15):
            # be= N_rt//2+sl
            fig, ax1 = plt.subplots()
            color = 'tab:red'
            ax1.set_xlabel('time (ps)')
            ax1.set_ylabel('Phase (rad)', color=color)
            ax1.plot(T,np.angle(Signal_output[be,:]),color,linewidth=0.5)
            ax1.yaxis.set_ticks([-pi,-3*pi/4,0,pi/4,pi])
            ax1.yaxis.set_ticklabels(["$\pi$", "$-3\pi/4$","$0$","$\pi/4$","$\pi$",])
            plt.hlines(pi/4,-t_rt/2,t_rt/2,colors=color,linestyles="dotted",linewidth=0.5)
            plt.hlines(-3*pi/4,-t_rt/2,t_rt/2,colors=color,linestyles="dotted",linewidth=0.5)
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            color = 'tab:blue'
            ax2.set_ylabel('Norm. Int.', color=color)  # we already handled the x-label with ax1
            ax2.plot(T,np.abs(Signal_output[be,:])**2/np.max(
                abs(Signal_output[be,:])**2),'--',linewidth=1.0)
            plt.hlines(0,-t_rt/2,t_rt/2,colors=color,linestyles="dotted",linewidth=0.5)
            plt.xlim([0,10])
            plt.title(r"Short-time slice in R.T. #{:1} - delta={:2}".format(be,delta))