#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 15:18:06 2021
@author: Alfredo Sanchez
@email: Alfredo.Sanchez@icfo.eu

This code models the light propagation in a OPO.
Essentially, this code solves the well-known coupled equations in a nonlinear 
crystal in order to obtain both temporal and spectral outputs.

This code calls the file "evolution.py" where are some functions used here.
It also calls the file "rindmgoplln.py" to bring the refractive index.

The convention for three field is PSI (pump-signal-idler)
"""

 # Imports
import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
import timeit
import os
import evolution as opo
import rindmgoplln as ri

# The screen clear function
def screen_clear():
   # for mac and linux(here, os.name is 'posix')
   if os.name == 'posix':
      _ = os.system('clear')
   else:
      # for windows platfrom
      _ = os.system('cls')
      

plt.close("all")
## Number of equations and regime (CW or pulsed)
eq, reg = 3, "cw"

print("Number of equations {}".format(eq))
save_files = True
if (save_files is True):
    # path  
    path = reg+"_regime_"+str(eq)+"eq"
    path_fig =  path+"/figures"
    # Create the directory to save all the created files
    try:
        os.mkdir(path)
        os.mkdir(path_fig)
    except OSError as error:
        print(error) 

#####################################################################
## Set parameters and constants
nr          = 1 # number of realizations
N_rt        = 500 # number of round trips to cover the input pulse
c           = 299792458*1e6/1e12 # speed of light in vac. [μm/ps]
eps0        = 8.8541878128e-12*1e12/1e6 # vacuum pertivity [W.ps/V²μm]
pi          = np.pi

## Define wavelngths
lp          = 0.532            # pump wavelenght [μm]
ls          = 2*lp          # signal wavelenght [μm]
li          = 1/(1/lp-1/ls)   # idler wavelenght [μm]
wl          = np.array([lp,ls,li ])
fp, wp      = c/lp, 2*pi*c/lp # pump frequency [THz]
fs, ws      = c/ls, 2*pi*c/ls # signal frequency [THz]
fi, wi      = c/li, 2*pi*c/li # signal frequency [THz]

## Crystal
Temp = 21
cr = opo.Crystal(deff=15.0e-6,
                 length=10e3,
                 refindx=ri.n(wl,Temp),
                 group_vel=ri.group_vel(wl, Temp),
                 gvd=ri.GVD(wl,Temp),
                 kind="Mgo:PPLN")
print(cr)
dz = cr.length/50 # step size in z-crystal
kappas = 4*pi*cr.deff/(cr.refindx*wl) # kappa [1/V]
dk = 0 # mismatch

## Cavity
Lcav = 1.235e6
cavity = opo.Cavity(length=Lcav,
                    R=0.95, 
                    trt=(Lcav-cr.length)/c+cr.length/np.max(cr.group_vel))

## Time and frequency grid
T_width     = N_rt*cavity.trt # total time for input ns-pulse
N_ps        = 2**15 # #points per time slice
extra_win   = 16 # extra pts for short-time slices
dT          = cavity.trt/N_ps # time step in [ps]
dF          = 1/cavity.trt # frequency step in [THz]
T           = (np.linspace(-N_ps/2, N_ps/2-1, N_ps))*dT # temp. grid for slice
Tp          = (np.linspace(-N_ps*N_rt/2, N_ps*N_rt/2-1,
                           N_ps*N_rt))*dT # temp. grid for pump field
dF_p        = 1/T_width
f_p         = (np.linspace(-Tp.size/2,Tp.size/2-1,
                           Tp.size))*dF_p # freq. grid [THz]
w_p         = fft.fftshift(2*pi*f_p) # ang freq [2*pi*THz]
f           = (np.linspace(-N_ps/2,N_ps/2-1,N_ps))*dF # freq grid [THz]
w           = fft.fftshift(2*pi*f) # ang freq [2*pi*THz]
f_ext       = (np.linspace(-(N_ps+extra_win)/2,(N_ps+extra_win)/2-1,
                           (N_ps+extra_win)))*dF # extended freq. grid [THz]
w_ext       = fft.fftshift(2*pi*f_ext) # extended ang freq [2*pi*THz]


## Fields 
# Pump field
FWHM        = 2000 # input pump with fwhm [ps]
sigmap      = FWHM/2/np.sqrt(2*np.log(2)) # standar deviation [ps]
waist       = 55 # beam waist radius [μm]
spot        = pi*waist**2 # spot area [μm²]
P_threshold = 25000 #spot*n_pump*eps0*c*(ls*n_signal/2/pi/deff/Lc)**2*((1-0.3)/0.3)
rep_rate    = 1e3*1e-12 # repetition rate in THz
Peak_P      = 2*P_threshold/FWHM/rep_rate
Intensity   = 2/spot # intensity [W/μm²]
Ap0         = np.sqrt(Intensity*2/c/cr.refindx[0]/eps0) # input field amplitud [V/μm] 
Ap_total    = Ap0*np.exp(-(Tp/2/sigmap)**2) if reg=="ns" else Ap0*np.ones(Tp.shape) # pump field CW

A = np.zeros([eq,N_ps+extra_win],dtype="complex")
fields = opo.Fields()

spectrum_signal = np.zeros([nr,len(Tp)], dtype="complex")
Signal          = np.zeros([N_rt,N_ps+extra_win],dtype='complex')
spectrum_pump   = np.zeros([nr,len(Tp)], dtype="complex")
Pump            = np.zeros([N_rt,N_ps+extra_win],dtype='complex')
spectrum_idler  = np.zeros([nr,len(Tp)], dtype="complex")
Idler           = np.zeros([N_rt,N_ps+extra_win],dtype='complex')

# This is the main loop for nr noise realizations and for the N_rt 
# round-trips in which the input pump pulse was splitted
delta_crit = cr.length*cr.refindx[1]/c/cavity.trt
print(delta_crit)

deltas = list(delta_crit*np.arange(-7.5,5.5,0.5))

for delta in deltas:
    # plt.close("all")
    start = timeit.default_timer()
    for j in range(0,nr):
        # Signal and Idler fields
        mu, sigma = 0, 1e-20 # for generating a signal noisy vector
        A[1,:] = np.random.normal(mu,sigma,N_ps+extra_win)+1j*np.random.normal(
            mu,sigma, N_ps+extra_win)
        if(eq == 3):
            A[2,:] = np.random.normal(mu,sigma,N_ps+extra_win)+1j*np.random.normal(
                mu,sigma, N_ps+extra_win)
        for i in range(0,N_rt):
            if(i%(N_rt/5)==0):
                print("δ={:+1.4f} - Realiz #: {:1} - Rnd.trip#: {:1}/{:1}"
                      .format(delta, j+1, i, N_rt))
            # here the time-short slice is selected with extra points
            if(i==0):
                A[0,:] = Ap_total[0:N_ps+extra_win]        
            elif(i>0 and i<N_rt-2):
                A[0,:] = Ap_total[i*N_ps-extra_win//2:(i+1)*N_ps+extra_win//2]
            if(i==N_rt-1):
                A[0,:] = Ap_total[i*N_ps-extra_win:(i+1)*N_ps]
            
            # Propagation in the nonlinear crystal
            z = 0
            A = fields.evol_in_crystal(A,kappas,dk,dz,z,cr.length,w_ext,
                                       cr.group_vel,cr.gvd)
            # intra-cavity phase accumulation exp(jmπ+/-jδ)
            A[1,:]=np.sqrt(cavity.R)*A[1,:]*np.exp(1j*pi*(i+delta))
            if(eq == 3):
                A[2,:]=np.sqrt(cavity.R)*A[2,:]*np.exp(1j*pi*(i+delta))
            Pump[i,:]   = A[0,:] # save the signal jth-round-trip    
            Signal[i,:] = A[1,:] # save the signal jth-round-trip
            if(eq == 3):
                Idler[i,:]  = A[2,:] # save the signal jth-round-trip
    
        Signal_output = Signal[:,0:N_ps]
        spectrum_signal[j,:] = fft.fft(Signal_output.reshape(Signal_output.size))
        Pump_output   = Pump[:,0:N_ps]
        spectrum_pump[j,:] = fft.fft(Pump_output.reshape(Pump_output.size))
        Idler_output  = Idler[:,0:N_ps]
        spectrum_idler[j,:] = fft.fft(Idler_output.reshape(Idler_output.size))
    
    # Post-processing and plotting
    spectrum_signal_av = np.mean(spectrum_signal,axis=0) # averaged signal spectrum
    SDP_signal_av      = np.abs(fft.ifftshift(spectrum_signal_av))**2 # averaged SDP.
    spectrum_idler_av = np.mean(spectrum_idler,axis=0) # averaged signal spectrum
    SDP_idler_av      = np.abs(fft.ifftshift(spectrum_idler_av))**2 # averaged SDP.
    spectrum_pump_av = np.mean(spectrum_pump,axis=0) # averaged signal spectrum
    SDP_pump_av      = np.abs(fft.ifftshift(spectrum_pump_av))**2 # averaged SDP.
        
    # Plot results
    SDP_AV_NS = SDP_signal_av
    SDP_AV_NI = SDP_idler_av
    SDP_AV_NP = SDP_pump_av
    
    if(save_files is True):
        freq_file = f_p 
        freq_file = np.append(freq_file, SDP_AV_NS, axis=0)
        freq_file = np.append(freq_file, SDP_AV_NI, axis=0)
        freq_file = np.append(freq_file, SDP_AV_NP, axis=0)
        freq_file = np.transpose(freq_file.reshape(4,len(freq_file)//4))
        np.savetxt(path+"/"+reg+"_spect_delta_{:+1.4f}.dat".format(delta), 
                    freq_file, delimiter=',')
        
    for be in [int(0.2*N_rt),int(0.4*N_rt), int(0.6*N_rt),int(0.8*N_rt)]:
        PS = np.angle(Signal_output[be,:])
        PI = np.angle(Idler_output[be,:])
        PP = np.angle(Pump_output[be,:])
        IS = np.abs(Signal_output[be,:])**2
        II = np.abs(Idler_output[be,:])**2
        IP = np.abs(Pump_output[be,:])**2
        if(save_files is True):
            time_filep = T
            time_filep = np.append(time_filep, PS, axis=0)
            time_filep = np.append(time_filep, PI, axis=0)
            time_filep = np.append(time_filep, PP, axis=0)
            time_filep = np.transpose(time_filep.reshape(4,len(time_filep)//4))
            np.savetxt(path+"/"+reg+"_phases_rt_{:1}_delta_{:+1.4f}.dat".format(be,delta), time_filep, delimiter=',')
            time_filei= T
            time_filei = np.append(time_filei, IS, axis=0)
            time_filei = np.append(time_filei, II, axis=0)
            time_filei = np.append(time_filei, IP, axis=0)
            time_filei = np.transpose(time_filei.reshape(4,len(time_filei)//4))
            np.savetxt(path+"/"+reg+"_intens_rt_{:1}_delta_{:+1.4f}.dat".format(be,delta), time_filei, delimiter=',')
    stop = timeit.default_timer()
    print('Runtime per delta: {:1.2f} minutes'.format((stop - start)/60)) 
