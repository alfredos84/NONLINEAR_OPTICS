#ifndef _FUNCTIONSCUH
#define _FUNCTIONSCUH



/**
 * This file contains a set of functions used in single-pass
 * as well as another important calculations.
 * 
 * */

#include <iostream>
#include <string>
#include <random>
#include <fstream>
#include <iomanip>
#include <typeinfo>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <stdio.h>
#include <sys/time.h>
#include <chrono>

#include <cuda_runtime.h>
#include <cufft.h>
#include <thrust/host_vector.h>
#include <thrust/complex.h>
#include <cuComplex.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/generate.h>
#include <thrust/transform.h>
#include <thrust/random.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>




// Complex data type
using CC_t = cuFloatComplex;
using typefl_t = float;

using rVech_t = thrust::host_vector<typefl_t>;
using rVecd_t = thrust::device_vector<typefl_t>;
using cVech_t = thrust::host_vector<CC_t>;
using cVecd_t = thrust::device_vector<CC_t>;	


/* FUNCTIONS */

void NoiseGeneratorCPU ( cVech_t &As ) // set the signal field as a noisy complex vector
{
	unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::normal_distribution<typefl_t> distribution(0.0, 1.0e-15);

	for (int i=0; i < As.size(); ++i) {
		As[i].x = distribution(generator);
		As[i].y = distribution(generator);
	}
	
	return ;
	
}

void InputField(cVech_t &Ap, typefl_t Ap0, std::string regime) // set the input cw pump field
{
	if(regime == "cw"){
		std::cout << "Wave:                   Continuous Wave\n" << std::endl;
		for (int i = 0; i < Ap.size(); i++){
			Ap[i].x = Ap0;
			Ap[i].y = 0;
		}
	}
	else{
		std::cout << "Wave:                   Error\n" << std::endl;
	}
	
	return ;
}


void linspace( rVech_t &V, typefl_t xmin, typefl_t xmax)// fills the a vector
{
	for (int i = 0; i < V.size(); i++)
		V[i] = xmin + i * (xmax - xmin)/(V.size()-1);
	
	return ;
}


void linspace( rVecd_t &V, typefl_t xmin, typefl_t xmax)  // fills the a vector
{
	for (int i = 0; i < V.size(); i++)
		V[i] = xmin + i * (xmax - xmin)/(V.size()-1);

	return ;
}


void inic_vector_F(rVech_t &V, typefl_t DF)  // fills the frequency vector
{
    for (int i = 0; i < V.size(); i++){
        V[i] = i * DF - V.size() * DF/2.0;
    }
    
    return ;
}


void inic_vector_F(rVecd_t &V, typefl_t DF) // fills the frequency vector
{
    for (int i = 0; i < V.size(); i++){
        V[i] = i * DF - V.size() * DF/2.0;
    }
    
    return ;
}


void fftshift( rVech_t &V_ss, rVech_t v ) // flips a vector
{
	int c = v.size()/2;
    
	for ( int i = 0; i < v.size()/2; i++ ){
		V_ss[i+c] = v[i];
		V_ss[i]   = v[i+c];
	}
    
    return ;
}


void dAdz( cVecd_t &dAp, cVecd_t &dAs, cVecd_t &Ap, cVecd_t &As, typefl_t kp, typefl_t ks, typefl_t dk, typefl_t z )
{
	/**
	 * This functions accounts returns the electric fields variation along z, and it is used in the RK4 method.
	 * 
	 * Inputs:
	 * 
	 * dAp, dAs:			pump and signal fields in the time domain after passing dz
	 * Ap, As:			     	pump and signal fields in the time domain
	 * kp, ks, auxp and auxs:	coupling factors at the corresponding wavelength
	 * dk: 				mismatch factor
	 * z:					distance propagation
	 */
	
	CC_t const_p =  make_float2(-kp*sinf(dk*z), +kp*cosf(dk*z));
	CC_t const_s =  make_float2(+ks*sinf(dk*z), +ks*cosf(dk*z));
		
	thrust::transform(As.begin(), As.end(), As.begin(), dAp.begin(), ComplexMultbyComplexCoef(const_p) );
	
	cVecd_t Asconj(As.size()); thrust::transform(As.begin(), As.end(), Asconj.begin(), Conjugate());
	thrust::transform(Ap.begin(), Ap.end(), Asconj.begin(), dAs.begin(), ComplexMultbyComplexCoef(const_s) );
	
	
	return ;
	
}


void RK4(cVecd_t &Ap, cVecd_t &As, cVecd_t &k1p, cVecd_t &k1s, cVecd_t &k2p, cVecd_t &k2s, cVecd_t &k3p, cVecd_t &k3s, cVecd_t &k4p, cVecd_t &k4s, typefl_t dz)
{
	/**
	 * This functions accounts returns the electric fields after solving the fourth-order Runge-Kutta
	 * method.
	 * 
	 * Inputs:
	 * 
	 * Ap, As:			     	pump and signal fields in the time domain
	 * kip and kis (i=p,s):     	auxiliar vectors for calculations
	 * dz:				step size in z
	 */
		
	cVecd_t Aux1(Ap.size()); cVecd_t Aux2(Ap.size());
	
	thrust::transform( k1p.begin(), k1p.end(), k4p.begin(), Aux1.begin(), ComplexLinearCombRealCoef(1.,1.));    // k1 + k4
	thrust::transform( k2p.begin(), k2p.end(), k3p.begin(), Aux2.begin(), ComplexLinearCombRealCoef(2.,2.));    // 2.k2 + 2.k3
	thrust::transform( Aux2.begin(), Aux2.end(), Aux1.begin(), Aux1.begin(), ComplexSum());                     // k1 + 2.k2 + 2.k3 + k4
	thrust::transform( Aux1.begin(), Aux1.end(), Ap.begin(), Ap.begin(), ComplexLinearCombRealCoef(dz/6.0,1.));	// Ap -> Ap + ( k1+2k2+2k3 +k4)dz/6

	thrust::transform( k1s.begin(), k1s.end(), k4s.begin(), Aux1.begin(), ComplexLinearCombRealCoef(1.,1.));    // k1 + k4
	thrust::transform( k2s.begin(), k2s.end(), k3s.begin(), Aux2.begin(), ComplexLinearCombRealCoef(2.,2.));    // 2.k2 + 2.k3
	thrust::transform( Aux2.begin(), Aux2.end(), Aux1.begin(), Aux1.begin(), ComplexSum());				// k1 + 2.k2 + 2.k3 + k4
	thrust::transform( Aux1.begin(), Aux1.end(), As.begin(), As.begin(), ComplexLinearCombRealCoef(dz/6.0,1.));	//  As -> As + ( k1+2k2+2k3 +k4)dz/6

	return ;
}


void LinearOperator(cVecd_t &Apw, cVecd_t &Asw, cVecd_t &w_GVDp, cVecd_t &w_GVDs, typefl_t alphap, typefl_t alphas, typefl_t dz)
{
	/**
	 * This functions accounts for the linear effects on the electric fields
	 * 
	 * Inputs:
	 * 
	 * Apw and Asw:           pump and signal fields in the frequency domain
	 * w_GVDp and w_GVDs:     e^(i.dz.((1/vs-1/vp).Ω + ½.β.Lcr.Ω²)) and e^(i.½.β.Lcr.Ω²))
	 * alphap and alphas:     linear absorption at the corresponding wavelength
	 * dz:                    step size in z
	 */
	
	typefl_t attenp = expf(-0.5*alphap*dz);
	typefl_t attens = expf(-0.5*alphas*dz);
		
	thrust::transform ( Apw.begin(), Apw.end(), w_GVDp.begin(), Apw.begin(), ComplexMultbyRealCoef(attenp));
	thrust::transform ( Asw.begin(), Asw.end(), w_GVDs.begin(), Asw.begin(), ComplexMultbyRealCoef(attens));
	
	return ;
}


void SinglePass( cufftHandle plan, cVecd_t &w_GVDp, cVecd_t &w_GVDs, cVecd_t &Ap, cVecd_t &As, cVecd_t &Apw, cVecd_t &Asw, cVecd_t &k1p, cVecd_t &k1s, cVecd_t &k2p, cVecd_t &k2s, cVecd_t &k3p, cVecd_t &k3s, cVecd_t &k4p, cVecd_t &k4s, cVecd_t &auxp, cVecd_t &auxs, typefl_t dk, typefl_t alphap, typefl_t alphas, typefl_t kp, typefl_t ks, typefl_t dz, int steps_z )
{
	/**
	 * This functions accounts for a single-pass on the nonlinear crystal. 
	 * It uses the Split-Step Fourier Method by solving the nonlinear part
	 * in the time doming using a standard fourth-order Runge-Kutta method,
	 * where as and the linear part is solved in the frequency domain. 
	 * 1- Firsly, it solves the time domain for a half-step size, dz/2.
	 * 2- Secondly, computes the linear part in the frequency domain for a 
	 * full-step size, dz. 
	 * 3- Finally, computes again the nonlinear part in the time domain for
	 * a half-step size, dz/2. 
	 * 
	 * The process is repeated until the fields leave the nonlinear crystal.
	 * 
	 * Inputs:
	 * 
	 * plan:				plan used for FFT/IFFT calculations
	 * w_GVDp and w_GVDs:     	e^(i.dz.((1/vs-1/vp).Ω + ½.β.Lcr.Ω²)) and e^(i.½.β.Lcr.Ω²))
	 * Ap, As, Apw and Asw:     	pump and signal fields in the frequency and time domains
	 * kip and kis (i=p,s):     	auxiliar vectors for calculations
	 * alphap and alphas:       	linear absorption at the corresponding wavelength
	 * kp, ks, auxp and auxs:	coupling factors at the corresponding wavelength
	 * dz:				step size in z
	 * steps_z:                 	number of slices in the nonlinear crystal
	 */

	typefl_t z = 0;
	for (int s = 0; s < steps_z; s++){
		
		/* RK4 for dz */
		//k1 = dAdz(kappas,dk,z,A)
		dAdz( k1p, k1s, Ap, As, kp, ks, dk, z );		
		//k2 = dAdz(kappas,dk,z+dz/2,A+k1/2) -> aux = A+k1/2
		thrust::transform ( Ap.begin(), Ap.end(), k1p.begin(), auxp.begin(), ComplexLinearCombRealCoef(1.,0.5) );
		thrust::transform ( As.begin(), As.end(), k1s.begin(), auxs.begin(), ComplexLinearCombRealCoef(1.,0.5) );
		dAdz( k2p, k2s, auxp, auxs, kp, ks, dk, z+dz/4.0 );
		// k3 = dAdz(kappas,dk,z+dz/2,A+k2/2) -> aux = A+k2/2
		thrust::transform ( Ap.begin(), Ap.end(), k2p.begin(), auxp.begin(), ComplexLinearCombRealCoef(1.,0.5) );
		thrust::transform ( As.begin(), As.end(), k2s.begin(), auxs.begin(), ComplexLinearCombRealCoef(1.,0.5) );
		dAdz( k3p, k3s, auxp, auxs, kp, ks, dk, z+dz/4.0 );
		// k4 = dAdz(kappas,dk,z+dz,A+k3)
		thrust::transform ( Ap.begin(), Ap.end(), k3p.begin(), auxp.begin(), ComplexLinearCombRealCoef(1.,1.) );
		thrust::transform ( As.begin(), As.end(), k3s.begin(), auxs.begin(), ComplexLinearCombRealCoef(1.,1.) );
		dAdz( k4p, k4s, auxp, auxs, kp, ks, dk, z+dz/2.0 );
		// A = A+(k1+2*k2+2*k3+k4)*dz/6
		RK4( Ap, As, k1p, k1s, k2p, k2s, k3p, k3s, k4p, k4s, dz*0.5);
		/**************************/
		
		/* Linear operator for dz/2 */
		ifft( Ap, Apw, plan);	ifft( As, Asw, plan);
		LinearOperator( Apw, Asw, w_GVDp, w_GVDs, alphap, alphas, dz );
		fft( Apw, Ap, plan );	fft( Asw, As, plan );
		/**************************/
		
		/* RK4 for dz */
		//k1 = dAdz(kappas,dk,z,A)
		dAdz( k1p, k1s, Ap, As, kp, ks, dk, z );		
		//k2 = dAdz(kappas,dk,z+dz/2,A+k1/2) -> aux = A+k1/2
		thrust::transform ( Ap.begin(), Ap.end(), k1p.begin(), auxp.begin(), ComplexLinearCombRealCoef(1.,0.5) );
		thrust::transform ( As.begin(), As.end(), k1s.begin(), auxs.begin(), ComplexLinearCombRealCoef(1.,0.5) );
		dAdz( k2p, k2s, auxp, auxs, kp, ks, dk, z+dz/4.0 );
		// k3 = dAdz(kappas,dk,z+dz/2,A+k2/2) -> aux = A+k2/2
		thrust::transform ( Ap.begin(), Ap.end(), k2p.begin(), auxp.begin(), ComplexLinearCombRealCoef(1.,0.5) );
		thrust::transform ( As.begin(), As.end(), k2s.begin(), auxs.begin(), ComplexLinearCombRealCoef(1.,0.5) );
		dAdz( k3p, k3s, auxp, auxs, kp, ks, dk, z+dz/4.0 );
		// k4 = dAdz(kappas,dk,z+dz,A+k3)
		thrust::transform ( Ap.begin(), Ap.end(), k3p.begin(), auxp.begin(), ComplexLinearCombRealCoef(1.,1.) );
		thrust::transform ( As.begin(), As.end(), k3s.begin(), auxs.begin(), ComplexLinearCombRealCoef(1.,1.) );
		dAdz( k4p, k4s, auxp, auxs, kp, ks, dk, z+dz/2.0 );
		// A = A+(k1+2*k2+2*k3+k4)*dz/6
		RK4( Ap, As, k1p, k1s, k2p, k2s, k3p, k3s, k4p, k4s, dz*0.5);
		/**************************/

		
		z+=dz;  // Next crystal slice
	}
}


void AddPhase( cVecd_t &As, typefl_t R, typefl_t delta, int rtn )
{
	/**
	 * This functions accounts for the accumulated phase and reflectivity losses 
	 * per round trip
	 * 
	 * Inputs:
	 * 
	 * As:        signal fields in the time domain
	 * R:         intensity reflectivity
	 * delta:     detuning (note that in the function this number is multiplied by PI)
	 * rtn:       is the number of round trip
	 */
		
	CC_t const_s =  make_float2( sqrtf(R)*cosf(PI*(rtn+delta)), sqrtf(R)*sinf(PI*(rtn+delta)) );
	As *= const_s;
	
	return ;
}


#endif // -> #ifdef _FUNCTIONSCUH
