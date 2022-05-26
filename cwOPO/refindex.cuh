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
using CC_t = cuFloatComplex;
using typefl_t = float;

using rVech_t = thrust::host_vector<typefl_t>;
using rVecd_t = thrust::device_vector<typefl_t>;
using cVech_t = thrust::host_vector<CC_t>;
using cVecd_t = thrust::device_vector<CC_t>;	
    
const typefl_t C     = 299792458*1e6/1e12; // speed of light in vac. [μm/ps]
const typefl_t PI     = 3.141592653589793238462643383279502884;





/****************** MgO:PPLN ******************/

__host__ __device__ 
typefl_t n_PPLN(typefl_t L,typefl_t T){
	
	/**	
	 * This function returns the MgO:PPLN extraordinary refractive index 
	from the Sellmeier Equation.
	Reference: O. Gayer_Temperature and wavelength dependent refractive index 
	equations for MgO-doped congruent and stoichiometric LiNbO3

	OUTPUT:
	ne: refractive index as a funcion of wavelength
    
    */
	typefl_t f = (T - 24.5) * (T + 570.82);
	typefl_t a1 = 5.756;
	typefl_t a2 = 0.0983;
	typefl_t a3 = 0.2020;
	typefl_t a4 = 189.32;
	typefl_t a5 = 12.52;
	typefl_t a6 =  1.32e-2;
	typefl_t b1 =  2.860e-6;
	typefl_t b2 =  4.700e-8;
	typefl_t b3 =  6.113e-8;
	typefl_t b4 =  1.516e-4;
	typefl_t G1 = a1 + b1*f;
	typefl_t G2 = a2 + b2*f;
	typefl_t G3 = a3 + b3*f;
	typefl_t G4 = a4 + b4*f;

	return sqrtf(G1+G2/(powf(L,2) - powf(G3,2))+G4/(powf(L,2) - powf(a5,2))-a6*L*L);
	
}


__host__ __device__ 
typefl_t dndl_PPLN(typefl_t L,typefl_t T){

	typefl_t f = (T - 24.5) * (T + 570.82);
	typefl_t a2 = 0.0983;
	typefl_t a3 = 0.2020;
	typefl_t a4 = 189.32;
	typefl_t a5 = 12.52;
	typefl_t a6 =  1.32e-2;
	typefl_t b2 =  4.700e-8;
	typefl_t b3 =  6.113e-8;
	typefl_t b4 =  1.516e-4;
	typefl_t G2 = a2 + b2*f;
	typefl_t G3 = a3 + b3*f;
	typefl_t G4 = a4 + b4*f;
	
	return -L*(G2/powf((pow(L,2)-powf(G3,2)),2)+G4/powf((pow(L,2)-powf(a5,2)),2) + a6)/n_PPLN(L, T);

}


__host__ __device__ 
typefl_t d2ndl2_PPLN(typefl_t L,typefl_t T){

	typefl_t f = (T - 24.5) * (T + 570.82);
	typefl_t a2 = 0.0983;
	typefl_t a3 = 0.2020;
	typefl_t a4 = 189.32;
	typefl_t a5 = 12.52;
	typefl_t b2 =  4.700e-8;
	typefl_t b3 =  6.113e-8;
	typefl_t b4 =  1.516e-4;
	typefl_t G2 = a2+b2*f;
	typefl_t G3 = a3+b3*f;
	typefl_t G4 = a4+b4*f;

	typefl_t A =((n_PPLN(L,T)-L*dndl_PPLN(L,T))/n_PPLN(L,T))*dndl_PPLN(L,T)/L;
	typefl_t B = (G2/powf(powf(L,2)-powf(G3,2),3) + G4/powf(powf(L,2)-powf(a5,2),3))*4*L*L/n_PPLN(L,T);
	return A+B;

}


__host__ __device__ 
typefl_t group_vel_PPLN(typefl_t L,typefl_t T){
	
	return C/(n_PPLN(L,T)-L*dndl_PPLN(L,T));
	
}


__host__ __device__ 
typefl_t gvd_PPLN(typefl_t L,typefl_t T){
	
	return powf(L,3)*d2ndl2_PPLN(L, T)/(2*PI*C*C);
	
}



/****************** MgO:sPPLT ******************/

__host__ __device__
typefl_t n_sPPLT(typefl_t L,typefl_t T){

	/**
	 * This function returns the MgO:sPPLT extraordinary refractive index from the Sellmeier Equation.
	Reference: Bruner et. al. Temperature-dependent Sellmeier equation for
	the refractive index of stoichiometric lithium tantalate

	OUTPUT:
	ne: refractive index as a funcion of wavelength
	
	*/
    
    typefl_t A =  4.502483;
    typefl_t B =  0.007294;
    typefl_t C =  0.185087;
    typefl_t D =  -0.02357;
    typefl_t E =  0.073423;
    typefl_t F =  0.199595;
    typefl_t G =  0.001;
    typefl_t H =  7.99724;
    typefl_t b =  3.483933e-8 * pow(T + 273.15,2);
    typefl_t c =  1.607839e-8 * pow(T + 273.15,2);
    
    return sqrt( A + (B+b)/(pow(L,2)-pow((C+c),2)) + E/(pow(L,2)-pow(F,2)) + G/(pow(L,2)-pow(H,2))+ D*pow(L,2));
    
}


__host__ __device__
typefl_t dndl_sPPLT(typefl_t L,typefl_t T){
	
    typefl_t B =  0.007294;
    typefl_t C =  0.185087;
    typefl_t D =  -0.02357;
    typefl_t E =  0.073423;
    typefl_t F =  0.199595;
    typefl_t G =  0.001;
    typefl_t H =  7.99724;
    typefl_t b =  3.483933e-8 * pow(T + 273.15,2);
    typefl_t c =  1.607839e-8 * pow(T + 273.15,2);
    
    return -L/n_sPPLT(L, T)*( (B+b)/pow(pow(L,2)-pow((C+c),2),2) + E/pow((pow(L,2)-pow(F,2)),2) + G/pow((pow(L,2)-pow(H,2)),2) - D );
    
}


__host__ __device__ 
typefl_t d2ndl2_sPPLT(typefl_t L,typefl_t T){
	
    typefl_t B =  0.007294;
    typefl_t C =  0.185087;
    typefl_t E =  0.073423;
    typefl_t F =  0.199595;
    typefl_t G =  0.001;
    typefl_t H =  7.99724;
    typefl_t b =  3.483933e-8 * pow(T + 273.15,2);
    typefl_t c =  1.607839e-8 * pow(T + 273.15,2);
    typefl_t S1 = dndl_sPPLT(L, T)/L;
    typefl_t S2 = 4*pow(L,2)/n_sPPLT(L,T)*((B+b)/pow(pow(L,2)-pow((C+c),2),3)+E/pow((pow(L,2)-pow(F,2)),3)+G/pow((pow(L,2)-pow(H,2)),3));
    
    return S1+S2;
    
}


__host__ __device__
typefl_t group_vel_sPPLT(typefl_t L,typefl_t T){
	
    return C/(n_sPPLT(L,T)-L*dndl_sPPLT(L,T));
    
}


__host__ __device__ 
typefl_t gvd_sPPLT(typefl_t L,typefl_t T){
	
	return pow(L,3)*d2ndl2_sPPLT(L, T)/(2*PI*C*C);
	
}



/******************    GaP    ******************/

__host__ __device__
typefl_t n_GaP(typefl_t L, typefl_t T){
	
    /**
     * This function returns the GaP extraordinary refractive index from the Sellmeier Equation.
    Reference: Wei et. al. Temperature dependent Sellmeier equation for the refractive index of GaP.
    
    OUTPUT:
    ne: refractive index as a funcion of wavelength.
    
    */
    
    typefl_t A =  10.926 + 7.0787e-4 * T + 1.8594e-7 * T*T;
    typefl_t B =  0.53718 + 5.8035e-5 * T + 1.9819e-7 * T*T;
    typefl_t C =  0.0911014;
    typefl_t D =  1504 + 0.25935 * T - 0.00023326 * T*T;
    typefl_t E =  758.048;
    
    return sqrt( A + B/(L*L-C) + D/(L*L-E) );
	
}


__host__ __device__ 
typefl_t dndl_GaP(typefl_t L, typefl_t T){
	
    typefl_t B =  0.53718 + 5.8035e-5 * T + 1.9819e-7 * T*T;
    typefl_t C =  0.0911014;
    typefl_t D =  1504 + 0.25935 * T - 0.00023326 * T*T;
    typefl_t E =  758.048;
    
    return -L/n_GaP(L,T) * ( B/pow(L*L-C,2) + D/pow(L*L-E,2) );
    
}


__host__ __device__ 
typefl_t d2ndl2_GaP(typefl_t L, typefl_t T){
	
    typefl_t B =  0.53718 + 5.8035e-5 * T + 1.9819e-7 * T*T;
    typefl_t C =  0.0911014;
    typefl_t D =  1504 + 0.25935 * T - 0.00023326 * T*T;
    typefl_t E =  758.048;
    
    return dndl_GaP(L, T)/L + 4*L*L/n_GaP(L,T)*( B/pow(L*L-C,3) + D/pow(L*L-E,3) );
    
}


__host__ __device__
typefl_t group_vel_GaP(typefl_t L,typefl_t T){
	
	return C/(n_GaP(L,T)-L*dndl_GaP(L,T));
	
}


__host__ __device__ 
typefl_t gvd_GaP(typefl_t L,typefl_t T){
	
    return pow(L,3)*d2ndl2_GaP(L, T)/(2*PI*C*C);
	
}



#endif // -> #ifdef _REfINDEX
