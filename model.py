#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 19:03:27 2021
@author: Alfredo Sanchez
@email: Alfredo.Sanchez@icfo.eu

This code models the light propagation in a OPO.
Essentially, this code solves the well-known coupled equations in a nonlinear 
crystal in order to obtain both temporal and spectral outputs.

This code calls the file "evolution.py" where are some functions used here.
"""

 # Imports
import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
import timeit
import os
import evolution as opo
from matplotlib import cm

# The screen clear function
def screen_clear():
   # for mac and linux(here, os.name is 'posix')
   if os.name == 'posix':
      _ = os.system('clear')
   else:
      # for windows platfrom
      _ = os.system('cls')
      

plt.close("all")
for equa in range(3,4):
    print("Number of equations {}".format(equa))
    eq = equa # number of equations
    save_files = True
    if (save_files is True):
        # path  
        path = "CW_regime_"+str(eq)+"eq"
        # Create the directory to save all the created files
        try:
            os.mkdir(path)
        except OSError as error:
            print(error) 
    
    #####################################################################
    ## Set parameters and constants
    nr          = 1 # number of realizations
    N_rt        = 1000 # number of round trips to cover the input pulse
    c           = 299792458*1e6/1e12 # speed of light in vac. [μm/ps]
    eps0        = 8.8541878128e-12*1e12/1e6 # vacuum pertivity [W.ps/V²μm]
    pi          = np.pi
    
    # Define the time- and frequency-grid
    lp          = 0.532            # pump wavelenght [μm]
    ls          = 2*lp          # signal wavelenght [μm]
    li          = 1/(1/lp-1/ls)   # idler wavelenght [μm]
    wl          = np.array([lp,ls,li ])
    fp, wp      = c/lp, 2*pi*c/lp # pump frequency [THz]
    fs, ws      = c/ls, 2*pi*c/ls # signal frequency [THz]
    fi, wi      = c/li, 2*pi*c/li # signal frequency [THz]
    
    # Crystal
    cr = opo.Crystal(deff=2.0e-6,
                     length=5e3,
                     refindx=np.array([ 1.654,1.654,1.654]),
                     group_vel=np.array([c/1.70, c/1.67, c/1.67]),
                     gvd=np.array([2*3.22e-7,2*2.07e-7, 2*2.07e-7]),
                     kind="BBO")
    print(cr)
    dz = cr.length/50 # step size in z-crystal
    kappas = 2*pi*cr.deff/(cr.refindx*wl) # kappa [1/V]
    dk = 0 # mismatch
    
    # Cavity
    Lcav = 20e3
    cavity = opo.Cavity(length=Lcav,
                        R=0.3, 
                        trt=(Lcav-cr.length)/c+cr.length/np.max(cr.group_vel))
    
    ## Time and frequency grid
    T_width     = N_rt*cavity.trt # total time for input ns-pulse
    N_ps        = 2**12 # #points per time slice
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
    Intensity   = 2*P_threshold/spot # intensity [W/μm²]
    Ap0         = np.sqrt(Intensity*2/c/cr.refindx[0]/eps0) # input field amplitud [V/μm] 
    # Ap_total    = Ap0*np.exp(-(Tp/2/sigmap)**2) # pump field ns-pulsed
    Ap_total    = Ap0*np.ones(Tp.shape) # pump field CW
    
    A = np.zeros([eq,N_ps+extra_win],dtype="complex")
    fields = opo.Fields()
    
    spectrum_signal = np.zeros([nr,len(Tp)], dtype="complex")
    Signal          = np.zeros([N_rt,N_ps+extra_win],dtype='complex')
    spectrum_pump   = np.zeros([nr,len(Tp)], dtype="complex")
    Pump            = np.zeros([N_rt,N_ps+extra_win],dtype='complex')
    if(eq == 3):
        spectrum_idler  = np.zeros([nr,len(Tp)], dtype="complex")
        Idler           = np.zeros([N_rt,N_ps+extra_win],dtype='complex')
    
    # This is the main loop for nr noise realizations and for the N_rt 
    # round-trips in which the input pump pulse was splitted
    delta_crit = cr.length*cr.refindx[1]/c/cavity.trt
    
    deltas = list([-1.0,-0.9,-0.8,-0.7,-0.65,(delta_crit-1),-0.645,-0.6350,
                       -0.63,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.0,+0.1,+0.2,+0.3,
                       +0.35,delta_crit,0.355,0.365,0.37,0.4,0.5,0.6,0.7,0.8,0.9,
                       1.0])
    
    for delta in deltas:
        plt.close("all")
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
                if(eq == 2):
                    A[1,:]=np.sqrt(cavity.R)*A[1,:]*np.exp(1j*pi*(i+delta))
                if(eq == 3):
                    A[1,:]=np.sqrt(cavity.R)*A[1,:]*np.exp(1j*pi*(i+delta))
                    A[2,:]=np.sqrt(cavity.R)*A[2,:]*np.exp(1j*pi*(i+delta))
                Pump[i,:]   = A[0,:] # save the signal jth-round-trip    
                Signal[i,:] = A[1,:] # save the signal jth-round-trip
                if(eq == 3):
                    Idler[i,:]  = A[2,:] # save the signal jth-round-trip
        
            Signal_output = Signal[:,0:N_ps]
            spectrum_signal[j,:] = fft.fft(Signal_output.reshape(Signal_output.size))
            Pump_output   = Pump[:,0:N_ps]
            spectrum_pump[j,:] = fft.fft(Pump_output.reshape(Pump_output.size))
            if(eq == 3):
                Idler_output  = Idler[:,0:N_ps]
                spectrum_idler[j,:] = fft.fft(Idler_output.reshape(Idler_output.size))
        
        wn = opo.freq2cm1(f_p)
        
        # Post-processing and plotting
        spectrum_signal_av = np.mean(spectrum_signal,axis=0) # averaged signal spectrum
        SDP_signal_av      = np.abs(fft.ifftshift(spectrum_signal_av))**2 # averaged SDP.
        SDP_signal         = np.zeros(spectrum_signal.shape)
        
        win_size = 10 # window size in cm^-1
        SDP_signal_av_filt = opo.filtro_triang(wn, win_size, SDP_signal_av)
        SDP_signal_filt = np.zeros([nr,len(SDP_signal_av_filt)])
        
        for j in range(nr):
            SDP_signal[j,:]      = np.abs(spectrum_signal[j,:])**2
            SDP_signal_filt[j,:] = opo.filtro_triang(wn, win_size, fft.ifftshift(SDP_signal[j,:]))
        
        SDP_signal_filt_av = np.mean(SDP_signal_filt,axis=0)
        limite = 400
        # Plot results
        inds = np.where((f_p>opo.cm12freq(-limite))&(f_p<opo.cm12freq(limite)))    
        frequency = f_p[inds]
        SDP_AV_N = SDP_signal_av[inds]/np.max(SDP_signal_av[inds])
        SDP_AV_FILT_N = SDP_signal_av_filt[inds]/np.max(SDP_signal_av_filt[inds])
        plt.figure(figsize=(6,4))
        plt.plot(frequency ,SDP_AV_N,':r', linewidth=1, label='Signal')
        plt.plot(frequency ,SDP_AV_FILT_N,'b', linewidth=1.5, label='Filtered')
        plt.xlabel(r"Frequency (THz)")
        plt.ylabel(r"Norm. spectral density (a.u.)")
        plt.title(r"Signal spectrum $\delta$ = {:+1.4f}$\pi$".format(delta))
        plt.legend(loc='upper right')
        plt.xlim([opo.cm12freq(-limite),opo.cm12freq(limite)])
        if(save_files is True):
            freq_file = frequency 
            freq_file = np.append(freq_file, SDP_AV_N, axis=0)
            freq_file = np.append(freq_file, SDP_AV_FILT_N, axis=0)
            freq_file = np.transpose(freq_file.reshape(3,len(freq_file)//3))
            np.savetxt(path+"/cw_spect_delta_{:+1.4f}.dat".format(delta), 
                       freq_file, delimiter=',')
            plt.savefig(path+"/spectrum_delta_{:+1.4f}.png".format(
                delta))
        
        for be in [int(0.4*N_rt), int(0.8*N_rt)]:
            fig, ax1 = plt.subplots(figsize=(8,5))
            color = 'tab:red'
            ax1.set_xlabel('time (ps)')
            ax1.set_ylabel('Phase (rad)', color=color)
            if(eq == 2):
                PS = np.angle(Signal_output[be,:])
                ax1.plot(T,PS,color,linewidth=1.5)
                ax1.yaxis.set_ticks([-pi,-3*pi/4,0,pi/4,pi])
                ax1.yaxis.set_ticklabels(["$\pi$", "$-3\pi/4$","$0$","$\pi/4$","$\pi$",])
                plt.axhline(pi/4,color=color,ls=":",linewidth=0.5)
                plt.axhline(-3*pi/4,color=color,ls=":",linewidth=0.5)
            if(eq == 3):
                PS = np.angle(Signal_output[be,:])
                PI = np.angle(Idler_output[be,:])
                ax1.plot(T,(PS+PI)/2,color,linewidth=1.5)
                ax1.yaxis.set_ticks([-pi,-3*pi/4,0,pi/4,pi])
                ax1.yaxis.set_ticklabels(["$\pi$", "$-3\pi/4$","$0$","$\pi/4$","$\pi$",])
                plt.axhline(pi/4,color=color,ls=":",linewidth=0.5)
                plt.axhline(-3*pi/4,color=color,ls=":",linewidth=0.5)            
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            color = 'tab:blue'
            ax2.set_ylabel('Norm. Int.', color=color)  # we already handled the x-label with ax1
            I = np.abs(Signal_output[be,:])**2/np.max(abs(Signal_output[be,:])**2)
            ax2.plot(T,I,'--',linewidth=1.25)
            plt.axhline(0,color=color,linewidth=0.5)
            plt.xlim([-10,10])
            plt.title(r"Short-time slice in R.T. #{:1}/{:1} - delta={:+1.4f}".format(be,N_rt,delta))
            if(save_files is True):
                time_file = T
                time_file = np.append(time_file, PS, axis=0)
                if(eq==3):
                    time_file = np.append(time_file, PI, axis=0)
                time_file = np.transpose(time_file.reshape(eq,len(time_file)//eq))
                np.savetxt(path+"/cw_rt_{:1}_delta_{:+1.4f}.dat".format(be,delta), time_file, delimiter=',')
                plt.savefig(path+"/phase_delta_{:+1.4f}_rt_{}.png".format(delta,be))

        stop = timeit.default_timer()
        print('Runtime per delta: {:1.2f} minutes'.format((stop - start)/60)) 