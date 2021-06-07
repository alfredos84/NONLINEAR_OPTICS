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

# The screen clear function
def screen_clear():
   # for mac and linux(here, os.name is 'posix')
   if os.name == 'posix':
      _ = os.system('clear')
   else:
      # for windows platfrom
      _ = os.system('cls')
      

plt.close("all")
reg, eq = "ns", 2

#####################################################################
## Set parameters and constants
nr          = 10 # number of realizations
N_rt        = 400 # number of round trips to cover the input pulse
c           = 299792458*1e6/1e12 # speed of light in vac. [μm/ps]
eps0        = 8.8541878128e-12*1e12/1e6 # vacuum pertivity [W.ps/V²μm]
pi          = np.pi

for ll in [2]:
    for rr in [.3]:
        ## Define wavelngths
        lp          = ll*0.532            # pump wavelenght [μm]
        ls          = 2*lp          # signal wavelenght [μm]
        li          = 1/(1/lp-1/ls)   # idler wavelenght [μm]
        wl          = np.array([lp,ls,li ])
        fp, wp      = c/lp, 2*pi*c/lp # pump frequency [THz]
        fs, ws      = c/ls, 2*pi*c/ls # signal frequency [THz]
        fi, wi      = c/li, 2*pi*c/li # signal frequency [THz]
        
        
        ## Crystal
        if( lp == 0.532):
            cr = opo.Crystal(deff=2.01e-6,
                             length=5e3,
                             refindx=np.array([ 1.656035,1.654208, 1.654208]),
                             group_vel=np.array([c/1.70, c/1.674, c/1.674]),
                             gvd=np.array([ 1.283e-7, 4.4e-8, 4.4e-8 ]),
                             kind="BBO")
        else:
            cr = opo.Crystal(deff=1.83e-6,
                             length=5e3,
                             refindx=np.array([ 1.636908,1.636851, 1.636851]),
                             group_vel=np.array([c/1.656, c/1.676, c/1.676]),
                             gvd=np.array([ 4.05e-8, -1.715e-7, -1.715e-7 ]),
                             kind="BBO")
        
        print(cr)
        dz = cr.length/50 # step size in z-crystal
        kappas = 2*pi*cr.deff/(cr.refindx*wl) # kappa [1/V]
        dk = 0 # mismatch
        
        ## Cavity
        Lcav = 20e3
        cavity = opo.Cavity(length=Lcav,
                            R=rr, 
                            trt=(Lcav-cr.length)/c+cr.length/np.max(cr.group_vel))
        
        save_files = True
        if (save_files is True):
            # path  
            path = reg+"_regime_"+str(int(lp*1e3))+"nm_R_"+str(cavity.R)
            # path_fig = path+"/figures"
            # Create the directory to save all the created files
            try:
                os.mkdir(path)
                # os.mkdir(path_fig)
            except OSError as error:
                print(error) 
        
        
        ## Time and frequency grid
        T_width     = N_rt*cavity.trt # total time for input ns-pulse
        N_ps        = 2**13 # #points per time slice
        extra_win   = 32 # extra pts for short-time slices
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
        FWHM        = 8000 # input pump with fwhm [ps]
        sigmap      = FWHM/2/np.sqrt(2*np.log(2)) # standar deviation [ps]
        waist       = 55 # beam waist radius [μm]
        spot        = pi*waist**2 # spot area [μm²]
        P_threshold = spot*cr.refindx[0]*eps0*c/8*(
            ls*cr.refindx[1]*(1-np.sqrt(cavity.R))/pi/cr.deff/cr.length)**2
        print("Power threshold = "+str(P_threshold)+" W")
        rep_rate    = 20e3*1e-12 # repetition rate in THz
        if(cavity.R == 0.3):
            pp = 22 
        else:
            pp = 22 
        print("pp = {}".format(pp))
        Peak_P      = 150e3
        Intensity   = Peak_P/spot # intensity [W/μm²]
        Ap0         = np.sqrt(Intensity*2/c/cr.refindx[0]/eps0) # input field amplitud [V/μm] 
        Ap_total    = Ap0*np.exp(-(Tp/(sigmap*np.sqrt(2)))**2/2) if reg=="ns" else Ap0*np.ones(Tp.shape) # pump field CW
        # pump_file = f_p 
        # pump_file = np.append(pump_file, np.abs(fft.ifftshift(fft.ifft(Ap_total)))**2, axis=0)
        # pump_file = np.transpose(pump_file.reshape(2,len(pump_file)//2))
        # np.savetxt(path+"/input_pump_spect_"+reg+".dat", pump_file, delimiter=',')
        
        
        A = np.zeros([eq,N_ps+extra_win],dtype="complex")
        fields = opo.Fields()
        
        spectrum_signal = np.zeros([nr,len(Tp)], dtype="complex")
        Signal          = np.zeros([N_rt,N_ps+extra_win],dtype='complex')
        spectrum_pump   = np.zeros([nr,len(Tp)], dtype="complex")
        Pump            = np.zeros([N_rt,N_ps+extra_win],dtype='complex')
        
        # This is the main loop for nr noise realizations and for the N_rt 
        # round-trips in which the input pump pulse was splitted
        
        if(ll == 1):
            l0= list(.01*np.arange(-100, -80, 1,dtype="int16"))
            l0+= list(.01*np.arange(-80, -70, 10,dtype="int16"))
            l0+= list(.01*np.arange(-70, -60, 1,dtype="int16"))
            l0+= list(.01*np.arange(-60, 0, 10,dtype="int16"))
            l0+= list(.01*np.arange(0, 20, 1,dtype="int16"))
            l0+= list(.01*np.arange(20, 30, 10,dtype="int16"))
            l0+= list(.01*np.arange(30, 40, 1,dtype="int16"))            
            l0+= list(.01*np.arange(40, 101, 10,dtype="int16"))            
        if(ll == 2):
            l0= list(.01*np.arange(-100, -40, 10,dtype="int16"))
            l0+= list(.01*np.arange(-40, -30, 1,dtype="int16"))
            l0+= list(.01*np.arange(-30, -20, 10,dtype="int16"))
            l0+= list(.01*np.arange(-20, 0, 1,dtype="int16"))
            l0+= list(.01*np.arange(0, 60, 10,dtype="int16"))
            l0+= list(.01*np.arange(60, 70, 1,dtype="int16"))
            l0+= list(.01*np.arange(70, 80, 10,dtype="int16"))            
            l0+= list(.01*np.arange(80, 101, 1,dtype="int16"))
        
        # deltas = list(.001*np.arange(-100, 101, 10,dtype="int16"))
        # deltas = list(.01*np.arange(-100, 101, 5,dtype="int16"))
        deltas = list(.01*np.arange(-30, 10, 5,dtype="int16"))
        deltas = [0, -.1, -.2, -.3]
        for delta in deltas:
            for gdd in [0, 0.25, 0.5, 0.75, 1.00]:
                # plt.close("all")
                start = timeit.default_timer()
                for j in range(0,nr):
                    # Signal and Idler fields
                    mu, sigma = 0, 1e-20 # for generating a signal noisy vector
                    A[1,:] = np.random.normal(mu,sigma,N_ps+extra_win)+1j*np.random.normal(
                        mu,sigma, N_ps+extra_win)
                    if(eq == 3):
                        A[2,:] =  A[1,:]
                        # A[2,:] = np.random.normal(mu,sigma,N_ps+extra_win)+1j*np.random.normal(
                            # mu,sigma, N_ps+extra_win)
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
                                    # intra-cavity phase accumulation due to GDD exp(j*0.5*GDD*Δω²)
                        GDD = -gdd * cr.gvd[1]*cr.length # GDD [ps²]
                        if(GDD != 0):
                            A[1,:]= fields.add_GDD(A[1,:], w_ext, GDD)
                        if(eq == 3):
                            A[2,:]=np.sqrt(cavity.R)*A[2,:]*np.exp(1j*pi*(i+delta))
                        Pump[i,:]   = A[0,:] # save the signal jth-round-trip    
                        Signal[i,:] = A[1,:] # save the signal jth-round-trip
                
                    Signal_output = Signal[:,0:N_ps]
                    spectrum_signal[j,:] = fft.ifft(Signal_output.reshape(Signal_output.size))
                    Pump_output   = Pump[:,0:N_ps]
                    spectrum_pump[j,:] = fft.ifft(Pump_output.reshape(Pump_output.size))
        
                   # Post-processing and plotting
                spectrum_signal_av = np.mean(spectrum_signal,axis=0) # averaged signal spectrum
                SDP_signal_av      = np.abs(fft.ifftshift(spectrum_signal_av))**2 # averaged SDP.
                spectrum_pump_av = np.mean(spectrum_pump,axis=0) # averaged signal spectrum
                SDP_pump_av      = np.abs(fft.ifftshift(spectrum_pump_av))**2 # averaged SDP.
                    
                # plt.figure()
                # plt.plot(f_p,SDP_signal_av)
                # plt.title("\delta = {:.2f}".format(delta))
                # plt.xlim([-15,15])
                # figname = "delta_{:.3f}.png".format(delta)
                # plt.savefig(figname,dpi=300)
                
                # Plot results
                SDP_AV_NS = SDP_signal_av
                SDP_AV_NP = SDP_pump_av
                
                if(save_files is True):
                    freq_file = f_p 
                    freq_file = np.append(freq_file, SDP_AV_NS, axis=0)
                    freq_file = np.append(freq_file, SDP_AV_NP, axis=0)
                    freq_file = np.transpose(freq_file.reshape(3,len(freq_file)//3))
                    np.savetxt(path+"/eq_"+str(eq)+"_"+reg+"_spect_delta_{:+1.4f}_gdd_{:1.2f}.dat".format(delta,gdd), 
                                freq_file, delimiter=',')
                    
            # for be in [int(0.4*N_rt),int(0.8*N_rt)]:
            #     PS = np.angle(Signal_output[be,:])
            #     #PI = np.angle(Idler_output[be,:])
            #     PP = np.angle(Pump_output[be,:])
            #     IS = np.abs(Signal_output[be,:])**2
            #     #II = np.abs(Idler_output[be,:])**2
            #     IP = np.abs(Pump_output[be,:])**2
            #     if(save_files is True):
            #         time_filep = T
            #         time_filep = np.append(time_filep, PS, axis=0)
            #         #time_filep = np.append(time_filep, PI, axis=0)
            #         time_filep = np.append(time_filep, PP, axis=0)
            #         time_filep = np.transpose(time_filep.reshape(3,len(time_filep)//3))
            #         np.savetxt(path+"/eq_"+str(eq)+"_"+reg+"_phases_rt_{:1}_delta_{:+1.4f}.dat".format(be,delta), time_filep, delimiter=',')
            #         time_filei= T
            #         time_filei = np.append(time_filei, IS, axis=0)
            #         #time_filei = np.append(time_filei, II, axis=0)
            #         time_filei = np.append(time_filei, IP, axis=0)
            #         time_filei = np.transpose(time_filei.reshape(3,len(time_filei)//3))
            #         np.savetxt(path+"/eq_"+str(eq)+"_"+reg+"_intens_rt_{:1}_delta_{:+1.4f}.dat".format(be,delta), time_filei, delimiter=',')
            stop = timeit.default_timer()
            print('Runtime per delta: {:1.2f} minutes'.format((stop - start)/60))