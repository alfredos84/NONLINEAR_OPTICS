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

This code calls the file "evolution2eq.py" where are the functions used here.
"""

 # Imports
import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
import evolution2eq as ev
import timeit

start = timeit.default_timer()

plt.close("all")
save_files = True
nr = 1 # number of realizations
N_rt = 1000 # number of round trips to cover the input pulse

# Constants
c           = 299792458*1e6/1e12 # speed of light in vac. [μm/ps]
eps0        = 8.8541878128e-12*1e12/1e6 # vacuum pertivity [W.ps/V²μm]
pi          = np.pi
# Define the time- and frequency-grid
lp          = .532   # pump wavelenght [μm]
ls          = 2*lp # signal wavelenght [μm]
fp, wp      = c/lp, 2*pi*c/lp # pump frequency [THz]
fs, ws      = c/ls, 2*pi*c/ls # signal frequency [THz]
# Crystal
n_pump      = 1.654   # refractive ind. at pump freq
n_signal    = 1.654   # refractive ind. at signal freq
deff        = 2.0e-6  # effective nonlinear suscept. [μm/V]
Lc          = 5e3 # crystal length [μm]
Lcav        = 20e3 # cavity lengh [μm]
R           = 0.7 # mirror reflectivity
dz          = Lc/50 # step size in z-crystal
# dz          = DZ  # half-step size in z-crystal
kappa_p     = 2*pi*deff/n_pump/lp # [1/V]
kappa_s     = 2*pi*deff/n_signal/ls # [1/V]

dk          = 0 # mismatch
vs,vp       = c/1.67, c/1.70  # signal, idler and pump GV [μm/ps]
beta_s      = 2*2.07e-7 # signal GVD in [ps²/μm]
beta_p      = 2*3.22e-7 # pump GVD in [ps²/μm]
dw_bw       = np.sqrt(pi/Lc/beta_s) # bandwidth acceptance [THz]
t_rt        = (Lcav-Lc)/c + Lc/np.max([vs,vp]) # round-trip time [ps]
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

# Pump field
FWHM        = 2000 # input pump with fwhm [ps]
sigmap      = FWHM/2/np.sqrt(2*np.log(2)) # standar deviation [ps]
waist       = 55 # beam waist radius [μm]
spot        = pi*waist**2 # spot area [μm²]
P_threshold = 50 #spot*n_pump*eps0*c*(ls*n_signal/2/pi/deff/Lc)**2*((1-0.3)/0.3)
rep_rate    = 1e3*1e-12 # repetition rate in THz
Peak_P      = 2*P_threshold/FWHM/rep_rate
Intensity   = P_threshold/spot # intensity [W/μm²]
Ap0         = np.sqrt(Intensity*2/c/n_pump/eps0) # input field amplitud [V/μm] 
# Ap_total    = Ap0*np.exp(-(Tp/2/sigmap)**2) # pump field ns-pulsed
Ap_total    = Ap0*np.ones(Tp.shape) # pump field CW

for delta in [-0.1,0.0,+0.1]:
    spectrum_signal  = np.zeros([nr,len(Tp)], dtype="complex")
    Signal = np.zeros([N_rt,N_ps+extra_win],dtype='complex')
    
    spectrum_pump    = np.zeros([nr,len(Tp)], dtype="complex")
    Pump   = np.zeros([N_rt,N_ps+extra_win],dtype='complex')

    # This is the main loop for nr noise realizations and for the N_rt 
    # round-trips in which the input pump pulse was splitted
    for j in range(0,nr):
        # Signal field
        mu, sigma = 0, 1e-20 # for generating a signal noisy vector
        As_in     = np.random.normal(mu,sigma,N_ps+extra_win)+1j*np.random.normal(mu,sigma, N_ps+extra_win)
        As        = As_in
        print("Pow=2Pth - δ={:2} - Realiz #: {:1}/{:1}".format(delta, j+1,nr))
        for i in range(0,N_rt):
            if(i%(N_rt/10)==0):
                print("Pow=2Pth - δ={:2} - Realiz #: {:1} - Rnd.trip#: {:1}/{:1}".format(delta, j+1, i, N_rt))
            # here the time-short slice is selected with extra points
            if(i==0):
                Ap = Ap_total[0:N_ps+extra_win]        
            elif(i>0 and i<N_rt-2):
                Ap = Ap_total[i*N_ps-extra_win//2:(i+1)*N_ps+extra_win//2]
            if(i==N_rt-1):
                Ap = Ap_total[i*N_ps-extra_win:(i+1)*N_ps]
            # Propagation in the nonlinear crystal
            z=0 # is the starting point of crystal
            As, Ap = ev.evol_in_crystal(As, Ap, kappa_s, kappa_p, dk, z, dz,
                                        Lc, w_ext, vp, vs, beta_p, beta_s)
            # intra-cavity phase accumulation exp(jmπ+/-jδ)
            As=np.sqrt(R)*As*np.exp(1j*pi*(i+delta))
            Signal[i,:] = As # save the signal nth-round-trip
            Pump[i,:] = Ap # save the signal nth-round-trip

        Signal_output = Signal[:,0:N_ps]
        spectrum_signal[j,:] = fft.ifft(Signal_output.reshape(Signal_output.size))
        Pump_output = Pump[:,0:N_ps]
        spectrum_pump[j,:] = fft.ifft(Pump_output.reshape(Pump_output.size))

    wn = ev.freq2cm1(f_p,c)
    
    # Post-processing and plotting
    spectrum_signal_av = np.mean(spectrum_signal,axis=0) # averaged signal spectrum
    SDP_signal_av      = np.abs(fft.ifftshift(spectrum_signal_av))**2 # averaged SDP.
    SDP_signal         = np.zeros(spectrum_signal.shape)

    win_size = 10 # window size in cm^-1
    SDP_signal_av_filt = ev.filtro_triang(wn, win_size, SDP_signal_av)
    SDP_signal_filt = np.zeros([nr,len(SDP_signal_av_filt)])
    
    for j in range(nr):
        SDP_signal[j,:]      = np.abs(spectrum_signal[j,:])**2
        SDP_signal_filt[j,:] = ev.filtro_triang(wn, win_size, fft.ifftshift(SDP_signal[j,:]))

    SDP_signal_filt_av = np.mean(SDP_signal_filt,axis=0)

    # Plot results
    inds = np.where((wn>-200)&(wn<200))    
    plt.figure(figsize=(6,4))
    plt.plot(wn[inds],SDP_signal_av[inds]/np.max(SDP_signal_av[inds]),':r', linewidth=1, label='Signal')
    plt.plot(wn[inds],SDP_signal_av_filt[inds]/np.max(SDP_signal_av_filt[inds]),'b', linewidth=1.5, label='Filtered')
    plt.xlabel(r"$\nu$ (cm$^{-1}$)")
    plt.ylabel(r"Norm. spectral density (a.u.)")
    plt.title(r"Signal spectrum $\delta$ = {:1.1f}$\pi$".format(delta))
    plt.legend(loc='upper right')
    plt.xlim([-200,200])
    if(save_files is True):
        plt.savefig("CW_OPO_paper/spectrum_delta_{:1.1f}_pow_2Pth.png".format(delta))
    
    for be in [int(0.4*N_rt), int(0.8*N_rt)]:
        fig, ax1 = plt.subplots(figsize=(8,5))
        color = 'tab:red'
        ax1.set_xlabel('time (ps)')
        ax1.set_ylabel('Phase (rad)', color=color)
        ax1.plot(T,np.angle(Signal_output[be,:]),color,linewidth=1.5)
        ax1.yaxis.set_ticks([-pi,-3*pi/4,0,pi/4,pi])
        ax1.yaxis.set_ticklabels(["$\pi$", "$-3\pi/4$","$0$","$\pi/4$","$\pi$",])
        plt.hlines(pi/4,-t_rt/2,t_rt/2,colors=color,linestyles="dotted",linewidth=0.5)
        plt.hlines(-3*pi/4,-t_rt/2,t_rt/2,colors=color,linestyles="dotted",linewidth=0.5)
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        ax2.set_ylabel('Norm. Int.', color=color)  # we already handled the x-label with ax1
        ax2.plot(T,np.abs(Signal_output[be,:])**2/np.max(
            abs(Signal_output[be,:])**2),'--',linewidth=1.25)
        plt.hlines(0,-t_rt/2,t_rt/2,colors=color,linestyles="dotted",linewidth=0.5)
        plt.xlim([25,35])
        plt.title(r"Short-time slice in R.T. #{:1}/{:1} - delta={:2}".format(be,N_rt,delta))
        if(save_files is True):
            plt.savefig("CW_OPO_paper/phase_delta_{:1.1f}_rt_{:1d}pow_2Pth.png".format(delta,be))

stop = timeit.default_timer()

print('Runtime: {:1.2f} seconds'.format( stop - start)) 