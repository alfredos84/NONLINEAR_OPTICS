#ifndef _REFINDEX
#define _REFINDEX



/**
 * This file contains a database with the Sellmeier equations for some
 * nonlinear crystals typically used in OPOs.
 * 
 * Inputs:
 * 
 * L:   wavelength in microns
 * T:   temperature in Celsius degrees
 * 
 * Funcions: the sufix XXXX is the used nonlinear crystal
 * 
 * n_XXXX:               returns the refractive index at the wavelength L and the temperature T
 * dndl_XXXX:            returns the first derivative of refractive index respect to wavelength (dn/dλ)
 * d2ndl2_XXXX:          returns the second derivative of refractive index respect to wavelength (d²n/dλ²)
 * group_vel_XXXX:       returns the group velocity at the wavelength L and the temperature T in um/ps (vg(λ) = c/(n(λ)-λdn/dλ))
 * gvd_XXXX:             returns the GVD at the wavelength L and the temperature T in ps²/um (β(λ)=λ³/(2πc²)(d²n/dλ²))
 * 
 * */



#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>
#include <iostream>
#include <string>



// Complex data type
using complex_t = cuFloatComplex;
using real_t = float;

using rVech_t = thrust::host_vector<real_t>;
using rVecd_t = thrust::device_vector<real_t>;
using cVech_t = thrust::host_vector<complex_t>;
using cVecd_t = thrust::device_vector<complex_t>;	
    
const real_t C     = 299792458*1e6/1e12; // speed of light in vac. [μm/ps]
const real_t PI     = 3.141592653589793238462643383279502884;





/****************** MgO:PPLN ******************/

__host__ __device__ 
real_t n_PPLN(real_t L,real_t T){
	
	/**	
	 * This function returns the MgO:PPLN extraordinary refractive index 
	from the Sellmeier Equation.
	Reference: O. Gayer_Temperature and wavelength dependent refractive index 
	equations for MgO-doped congruent and stoichiometric LiNbO3

	OUTPUT:
	ne: refractive index as a funcion of wavelength
    
    */
	real_t f = (T - 24.5) * (T + 570.82);
	real_t a1 = 5.756;
	real_t a2 = 0.0983;
	real_t a3 = 0.2020;
	real_t a4 = 189.32;
	real_t a5 = 12.52;
	real_t a6 =  1.32e-2;
	real_t b1 =  2.860e-6;
	real_t b2 =  4.700e-8;
	real_t b3 =  6.113e-8;
	real_t b4 =  1.516e-4;
	real_t G1 = a1 + b1*f;
	real_t G2 = a2 + b2*f;
	real_t G3 = a3 + b3*f;
	real_t G4 = a4 + b4*f;

	return sqrtf(G1+G2/(powf(L,2) - powf(G3,2))+G4/(powf(L,2) - powf(a5,2))-a6*L*L);
	
}


/** Returns the first-order derivative of the 
 * refractive index respect to the wavelength dn/dλ. */
__host__ __device__ 
real_t dndl_PPLN(real_t L,real_t T){

	real_t f = (T - 24.5) * (T + 570.82);
	real_t a2 = 0.0983;
	real_t a3 = 0.2020;
	real_t a4 = 189.32;
	real_t a5 = 12.52;
	real_t a6 =  1.32e-2;
	real_t b2 =  4.700e-8;
	real_t b3 =  6.113e-8;
	real_t b4 =  1.516e-4;
	real_t G2 = a2 + b2*f;
	real_t G3 = a3 + b3*f;
	real_t G4 = a4 + b4*f;
	
	return -L*(G2/powf((pow(L,2)-powf(G3,2)),2)+G4/powf((pow(L,2)-powf(a5,2)),2) + a6)/n_PPLN(L, T);

}


/** Returns the second-order derivative of the
 * refractive index respect to the wavelength d²n/dλ². */
__host__ __device__ 
real_t d2ndl2_PPLN(real_t L,real_t T){

	real_t f = (T - 24.5) * (T + 570.82);
	real_t a2 = 0.0983;
	real_t a3 = 0.2020;
	real_t a4 = 189.32;
	real_t a5 = 12.52;
	real_t b2 =  4.700e-8;
	real_t b3 =  6.113e-8;
	real_t b4 =  1.516e-4;
	real_t G2 = a2+b2*f;
	real_t G3 = a3+b3*f;
	real_t G4 = a4+b4*f;

	real_t A =((n_PPLN(L,T)-L*dndl_PPLN(L,T))/n_PPLN(L,T))*dndl_PPLN(L,T)/L;
	real_t B = (G2/powf(powf(L,2)-powf(G3,2),3) + G4/powf(powf(L,2)-powf(a5,2),3))*4*L*L/n_PPLN(L,T);
	return A+B;

}


/** Returns the third-order derivative of the
 * refractive index respect to the wavelength d³n/dλ³. */
__host__ __device__ real_t d3ndl3_PPLN(real_t L,real_t T){

	real_t f = (T - 24.5) * (T + 570.82);
	real_t a2 = 0.0983;
	real_t a3 = 0.2020;
	real_t a4 = 189.32;
	real_t a5 = 12.52;
	real_t b2 = 4.700e-8;
	real_t b3 = 6.113e-8;
	real_t b4 = 1.516e-4;
	real_t G2 = a2+b2*f;
	real_t G3 = a3+b3*f;
	real_t G4 = a4+b4*f;
	real_t G  = G2/powf(powf(L,2)-powf(G3,2),3) + G4/powf(powf(L,2)-powf(a5,2),3);
	real_t dG = -6*L*(G2/powf(powf(L,2)-powf(G3,2),4) + G4/powf(powf(L,2)-powf(a5,2),4));

	return d2ndl2_PPLN(L,T)*(1-2*dndl_PPLN(L,T)/n_PPLN(L,T)) + powf(dndl_PPLN(L,T),3)/powf(n_PPLN(L,T),2) + 4*L*L/n_PPLN(L,T)*dG + G*(8*L/n_PPLN(L,T)-powf(L/dndl_PPLN(L,T),2));
	
}


/** Returns the group-velocity vg(λ) = c/(n(λ)-λdn/dλ). */
__host__ __device__ 
real_t group_vel_PPLN(real_t L,real_t T){
	
	return C/(n_PPLN(L,T)-L*dndl_PPLN(L,T));
	
}


/** Returns the group-velocity β2(λ)=λ^3/(2πc²)(d²n/dλ²). */
__host__ __device__ 
real_t gvd_PPLN(real_t L,real_t T){
	
	return powf(L,3)*d2ndl2_PPLN(L, T)/(2*PI*C*C);
	
}


/** Returns the TOD β3(λ)=-λ^4/(4π²c³)[3.d²n/dλ² + λ.d³n/dλ³]. */
__host__ __device__ real_t TOD_PPLN(real_t L,real_t T){
	return -powf(L,4)/(4*PI*PI*C*C*C)*(3*d2ndl2_PPLN(L, T)+L*d3ndl3_PPLN(L, T));
}




/****************** MgO:sPPLT ******************/

__host__ __device__
real_t n_sPPLT(real_t L,real_t T){

	/**
	 * This function returns the MgO:sPPLT extraordinary refractive index from the Sellmeier Equation.
	Reference: Bruner et. al. Temperature-dependent Sellmeier equation for
	the refractive index of stoichiometric lithium tantalate

	OUTPUT:
	ne: refractive index as a funcion of wavelength
	
	*/
    
    real_t A =  4.502483;
    real_t B =  0.007294;
    real_t C =  0.185087;
    real_t D =  -0.02357;
    real_t E =  0.073423;
    real_t F =  0.199595;
    real_t G =  0.001;
    real_t H =  7.99724;
    real_t b =  3.483933e-8 * pow(T + 273.15,2);
    real_t c =  1.607839e-8 * pow(T + 273.15,2);
    
    return sqrt( A + (B+b)/(pow(L,2)-pow((C+c),2)) + E/(pow(L,2)-pow(F,2)) + G/(pow(L,2)-pow(H,2))+ D*pow(L,2));
    
}


__host__ __device__
real_t dndl_sPPLT(real_t L,real_t T){
	
    real_t B =  0.007294;
    real_t C =  0.185087;
    real_t D =  -0.02357;
    real_t E =  0.073423;
    real_t F =  0.199595;
    real_t G =  0.001;
    real_t H =  7.99724;
    real_t b =  3.483933e-8 * pow(T + 273.15,2);
    real_t c =  1.607839e-8 * pow(T + 273.15,2);
    
    return -L/n_sPPLT(L, T)*( (B+b)/pow(pow(L,2)-pow((C+c),2),2) + E/pow((pow(L,2)-pow(F,2)),2) + G/pow((pow(L,2)-pow(H,2)),2) - D );
    
}


__host__ __device__ 
real_t d2ndl2_sPPLT(real_t L,real_t T){
	
    real_t B =  0.007294;
    real_t C =  0.185087;
    real_t E =  0.073423;
    real_t F =  0.199595;
    real_t G =  0.001;
    real_t H =  7.99724;
    real_t b =  3.483933e-8 * pow(T + 273.15,2);
    real_t c =  1.607839e-8 * pow(T + 273.15,2);
    real_t S1 = dndl_sPPLT(L, T)/L;
    real_t S2 = 4*pow(L,2)/n_sPPLT(L,T)*((B+b)/pow(pow(L,2)-pow((C+c),2),3)+E/pow((pow(L,2)-pow(F,2)),3)+G/pow((pow(L,2)-pow(H,2)),3));
    
    return S1+S2;
    
}


__host__ __device__
real_t group_vel_sPPLT(real_t L,real_t T){
	
    return C/(n_sPPLT(L,T)-L*dndl_sPPLT(L,T));
    
}


__host__ __device__ 
real_t gvd_sPPLT(real_t L,real_t T){
	
	return pow(L,3)*d2ndl2_sPPLT(L, T)/(2*PI*C*C);
	
}



/******************    GaP    ******************/

__host__ __device__
real_t n_GaP(real_t L, real_t T){
	
    /**
     * This function returns the GaP extraordinary refractive index from the Sellmeier Equation.
    Reference: Wei et. al. Temperature dependent Sellmeier equation for the refractive index of GaP.
    
    OUTPUT:
    ne: refractive index as a funcion of wavelength.
    
    */
    
    real_t A =  10.926 + 7.0787e-4 * T + 1.8594e-7 * T*T;
    real_t B =  0.53718 + 5.8035e-5 * T + 1.9819e-7 * T*T;
    real_t C =  0.0911014;
    real_t D =  1504 + 0.25935 * T - 0.00023326 * T*T;
    real_t E =  758.048;
    
    return sqrt( A + B/(L*L-C) + D/(L*L-E) );
	
}


__host__ __device__ 
real_t dndl_GaP(real_t L, real_t T){
	
    real_t B =  0.53718 + 5.8035e-5 * T + 1.9819e-7 * T*T;
    real_t C =  0.0911014;
    real_t D =  1504 + 0.25935 * T - 0.00023326 * T*T;
    real_t E =  758.048;
    
    return -L/n_GaP(L,T) * ( B/pow(L*L-C,2) + D/pow(L*L-E,2) );
    
}


__host__ __device__ 
real_t d2ndl2_GaP(real_t L, real_t T){
	
    real_t B =  0.53718 + 5.8035e-5 * T + 1.9819e-7 * T*T;
    real_t C =  0.0911014;
    real_t D =  1504 + 0.25935 * T - 0.00023326 * T*T;
    real_t E =  758.048;
    
    return dndl_GaP(L, T)/L + 4*L*L/n_GaP(L,T)*( B/pow(L*L-C,3) + D/pow(L*L-E,3) );
    
}


__host__ __device__
real_t group_vel_GaP(real_t L,real_t T){
	
	return C/(n_GaP(L,T)-L*dndl_GaP(L,T));
	
}


__host__ __device__ 
real_t gvd_GaP(real_t L,real_t T){
	
    return pow(L,3)*d2ndl2_GaP(L, T)/(2*PI*C*C);
	
}




#endif // -> #ifdef _REfINDEX
