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

# Constants
c                  = 299792458*1e6/1e12 # speed of light in vac. [nm/ps]
pi                 = np.pi
N                  = 2**14 #  # of points in time and frequency vectors

# Define the time- and frequency-grid
lp                 = .532   # pump wavelenght [nm]
ls                 = 2*lp   # signal wavelenght [nm]
fp, wp             = c/lp, 2*pi*c/lp # pump frequency [THz]
fs, ws             = c/ls, 2*pi*c/ls # signal frequency [THz]

# Crystal
n_signal           = 1.654   # refractive ind. at signal freq
n_pump             = 1.654   # refractive ind. at pump freq
deff               = 2.0e-6  # effective nonlinear suscept. [nm/V]
Lc                 = 5e3 # crystal length [nm]
Lcav               = 20e3 # cavity lengh [nm]
DZ                 = Lc/50 # step size in z-crystal
dz                 = 0.5*DZ  # half-step size in z-crystal
kappa_p            = 2*pi*deff/n_pump/lp # [1/V]
kappa_s            = 2*pi*deff/n_signal/ls # [1/V]
dk                 = 0 # mismatch
vs,vp              = c/1.67, c/1.70  # signal and pump GV [nm/ps]
beta_s, beta_p     = 2.07e-25*1e24/1e6, 3.22e-25*1e24/1e6 # signal and pump GVD in [s²/m]
dw_bw              = np.sqrt(pi/Lc/beta_s) # bandwidth acceptance [THz]
t_rt               = (Lcav-Lc)/c + Lc/np.max([vs,vp]) # round-trip time [ps]
N_rt               = 80 # number of round trips to cover the input pulse
T_width            = N_rt*t_rt # total time for input ns-pulse
N_ps               = 2**12 # number of points per time slice
extra_win          = 20
dT                 = t_rt/N_ps
dF                 = 1/t_rt # frequency step in [THz]
T                  = np.arange(-t_rt/2,t_rt/2,dT) # temp. grid for slice

Tp                 = np.arange(-t_rt*N_rt/2,t_rt*N_rt/2,dT) # temp. grid for pulse
dF_p               = 1/T_width
f_p                = np.arange(-len(Tp)*dF_p/2,len(Tp)*dF_p/2,dF_p) # freq. grid [THz]
f_p                += dF_p 
w_p                = fft.fftshift(2*pi*f_p)  # angular frequency in [2*pi*THz]

f                  = np.arange(-N_ps*dF/2,N_ps*dF/2,dF) # freq. grid [THz]
f+=dF
w                  = fft.fftshift(2*pi*f)  # angular frequency in [2*pi*THz]
f_ext              = np.arange(-(N_ps+extra_win)*dF/2,(N_ps+extra_win)*dF/2,dF) # freq. grid [THz]
f_ext+=dF
w_ext              = fft.fftshift(2*pi*f_ext)  # angular frequency in [2*pi*THz]

# tau = Lc/c*(1/vp-1/vs) # walk-off time

# Input fields
# Pump
FWHM        = 2000 # input pump with fwhm [ps]
Ener_pulse  = 0.4 # energy pulse [J]
Peaw_P      = Ener_pulse/FWHM*1e7 # peak power [W]
Ap0         = np.sqrt(Peaw_P) # input field amplitud [W^1/2] 
sigmap      = FWHM/2/np.sqrt(2*np.log(2)) # standar deviation [ps]
Ap_total    = Ap0*np.exp(-(Tp/2/sigmap)**2)*np.exp(-2*1j*pi*0*Tp) # pump field

# Signal

Signal = np.zeros([N_rt,N_ps+extra_win],dtype='complex')
mu, sigma   = 0,0.00001 # for generating a signal noisy vector

nr = 2 # number of realizations
spectrum = np.zeros([nr,len(Tp)], dtype="complex")

Signal = np.zeros([N_rt,N_ps+extra_win],dtype='complex')
for j in range(0,nr):
    As_in = np.random.normal(mu,sigma,N_ps+extra_win)+1j*np.random.normal(mu,sigma, N_ps+extra_win)
    As    = As_in
    for i in range(0,N_rt):
        print("Realization {:1} - Round trip #: {:1}".format(j,i))
        if(i==0):
            Ap = Ap_total[0:N_ps+extra_win]        
        elif(i>0 and i<N_rt-2):
            Ap = Ap_total[i*N_ps-extra_win//2:(i+1)*N_ps+extra_win//2]
        if(i==N_rt-1):
            Ap = Ap_total[i*N_ps-extra_win:(i+1)*N_ps]
        # Propagation in the nonlinear crystal
        z=0
        As, Ap = ev.evol_in_crystal(As, Ap, kappa_s, kappa_p, dk, z, dz, Lc, 
                                w_ext, vp, vs, beta_p, beta_s)
        # intra-cavity phase exp(jmπ)
        As=np.sqrt(0.7)*As*np.exp(1j*(i*pi-0.1*pi))
        Signal[i,:] = As
    Signal_output = ev.filtrado(Signal, N_ps, extra_win)
    spectrum[j,:] = fft.fft(Signal_output.reshape(Signal_output.size))

wn = ev.freq2cm1(f_p,c)
spectrum_av = np.mean(spectrum,axis=0)    
irrad_av = np.abs(fft.fftshift(spectrum_av))**2

irrad = np.zeros(spectrum.shape)

win_size = 10 # window size in cm^-1
irrad_av_filt = ev.filtro_triang(wn, win_size, irrad_av)

irrad_filt = np.zeros([nr,len(irrad_av_filt)])
for i in range(nr):
    irrad[i,:] = np.abs(spectrum[i,:])**2
    irrad_filt[i,:] = ev.filtro_triang(wn, win_size, fft.fftshift(irrad[i,:]))
   
irrad_filt_av = np.mean(irrad_filt,axis=0)
inds = np.where((wn>-200)&(wn<200))    
plt.figure()
plt.plot(wn[inds],irrad_filt_av[inds]/np.max(irrad_filt_av[inds]))
plt.xlabel(r"$\nu$ (cm$^{-1}$)")
plt.title("Filtered, and then averaged")

af= 33
be= 33
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('time (ps)')
ax1.set_ylabel('Phase (rad)', color=color)
ax1.plot(T,np.angle(Signal_output[be,:]),color)
plt.hlines(pi/4,-6,0,colors=color,linestyles="dotted")
plt.hlines(-3*pi/4,-6,0,colors=color,linestyles="dotted")
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Norm. Int.', color=color)  # we already handled the x-label with ax1
ax2.plot(T,np.abs(Signal_output[be,:])**2/np.max(
    abs(Signal_output[be,:])**2))
plt.hlines(0,-6,0,colors=color,linestyles="dotted")
plt.xlim([-6,0])

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('time (ps)')
ax1.set_ylabel('Phase (rad)', color=color)
ax1.plot(T,np.angle(Signal_output[af,:]),color)
plt.hlines(pi/4,0,6,colors=color,linestyles="dotted")
plt.hlines(-3*pi/4,0,6,colors=color,linestyles="dotted")
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Norm. Int.', color=color)  # we already handled the x-label with ax1
ax2.plot(T,np.abs(Signal_output[af,:])**2/np.max(
    abs(Signal_output[af,:])**2))
plt.hlines(0,0,6,colors=color,linestyles="dotted")
plt.xlim([0,6])





# nr = 1 # number of realizations
# for j in range(0,nr):
#     for i in range(0,N_rt):
#         print("Realization {:1} Round trip #: {:1}".format(j,i))
#         if(i==0):
#             Ap = Ap_total[0:N_ps+extra_win]        
#         elif(i>0 and i<N_rt-2):
#             Ap = Ap_total[i*N_ps-extra_win//2:(i+1)*N_ps+extra_win//2]
#         if(i==N_rt-1):
#             Ap = Ap_total[i*N_ps-extra_win:(i+1)*N_ps]
#         # Propagation in the nonlinear crystal
#         z=0
#         As, Ap = ev.evol_in_crystal(As, Ap, kappa_s, kappa_p, dk, z, dz, Lc, 
#                                 w_ext, vp, vs, beta_p, beta_s)
#         # intra-cavity phase exp(jmπ)
#         As=np.sqrt(0.7)*As*np.exp(1j*i*pi)
#         Signal[i,:] += As

# Signal/=nr    
# Signal_output = ev.filtrado(Signal, N_ps, extra_win)        
    
# af= 33
# be= 33
# fig, ax1 = plt.subplots()
# color = 'tab:red'
# ax1.set_xlabel('time (ps)')
# ax1.set_ylabel('Phase (rad)', color=color)
# ax1.plot(T,np.angle(Signal_output[be,:]),color)
# plt.hlines(pi/4,-6,0,colors=color,linestyles="dotted")
# plt.hlines(-3*pi/4,-6,0,colors=color,linestyles="dotted")
# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
# color = 'tab:blue'
# ax2.set_ylabel('Norm. Int.', color=color)  # we already handled the x-label with ax1
# ax2.plot(T,np.abs(Signal_output[be,:])**2/np.max(
#     abs(Signal_output[be,:])**2))
# plt.hlines(0,-6,0,colors=color,linestyles="dotted")
# plt.xlim([-6,0])

# fig, ax1 = plt.subplots()
# color = 'tab:red'
# ax1.set_xlabel('time (ps)')
# ax1.set_ylabel('Phase (rad)', color=color)
# ax1.plot(T,np.angle(Signal_output[af,:]),color)
# plt.hlines(pi/4,0,6,colors=color,linestyles="dotted")
# plt.hlines(-3*pi/4,0,6,colors=color,linestyles="dotted")
# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
# color = 'tab:blue'
# ax2.set_ylabel('Norm. Int.', color=color)  # we already handled the x-label with ax1
# ax2.plot(T,np.abs(Signal_output[af,:])**2/np.max(
#     abs(Signal_output[af,:])**2))
# plt.xlim([0,6])

# # aux = Signal_output.reshape(Signal_output.size)
# # spectrum = fft.fft((aux))
# scp = Signal_output[af,:]    
# aux = np.abs(fft.fftshift(fft.fft(scp)))**2
# plt.figure()
# wn = ev.freq2cm1(f, c)
# plt.plot(wn[1::10],aux[1::10])
# plt.xlabel(r"Frequency (cm$^{-1}$)")
# plt.xlim([-200,200])