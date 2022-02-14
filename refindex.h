#pragma once

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>
#include <iostream>
#include <string>

#include "common.h"

// Complex data type
#ifdef DOUBLEPRECISION
    typedef double typefl;
#else
    typedef float typefl;
#endif

    
const typefl C     =   299792458*1e6/1e12; // speed of light in vac. [μm/ps]
const typefl PI    =   3.14159265358979323846;          //PI


//////////////////// MgO:PPLN ////////////////////

__host__ __device__ typefl n_PPLN(typefl L,typefl T){
/*	
    This function returns the MgO:PPLN extraordinary refractive index 
    from the Sellmeier Equation.
    Reference: O. Gayer_Temperature and wavelength dependent refractive index 
    equations for MgO-doped congruent and stoichiometric LiNbO3
    
    INPUTS:
    L: wavelenght in um
    T: temperature in degrees
    
    OUTPUT:
    ne: refractive index as a funcion of wavelength
    */
    typefl f = (T - 24.5) * (T + 570.82);
    typefl a1 = 5.756;
    typefl a2 = 0.0983;
    typefl a3 = 0.2020;
    typefl a4 = 189.32;
    typefl a5 = 12.52;
    typefl a6 =  1.32e-2;
    typefl b1 =  2.860e-6;
    typefl b2 =  4.700e-8;
    typefl b3 =  6.113e-8;
    typefl b4 =  1.516e-4;
    typefl G1 = a1 + b1*f;
    typefl G2 = a2 + b2*f;
    typefl G3 = a3 + b3*f;
    typefl G4 = a4 + b4*f;
	#ifdef DOUBLEPRECISION
		return sqrt(G1+G2/(pow(L,2) - pow(G3,2))+G4/(pow(L,2) - pow(a5,2))-a6*L*L);
	#else
		return sqrtf(G1+G2/(powf(L,2) - powf(G3,2))+G4/(powf(L,2) - powf(a5,2))-a6*L*L);
	#endif
}

__host__ __device__ typefl dndl_PPLN(typefl L,typefl T){
	/*
    Returns the first-order derivative of the refractive index respect to
    the wavelength dn/dλ.
    */
    typefl f = (T - 24.5) * (T + 570.82);
    typefl a2 = 0.0983;
    typefl a3 = 0.2020;
    typefl a4 = 189.32;
    typefl a5 = 12.52;
    typefl a6 =  1.32e-2;
    typefl b2 =  4.700e-8;
    typefl b3 =  6.113e-8;
    typefl b4 =  1.516e-4;
    typefl G2 = a2 + b2*f;
    typefl G3 = a3 + b3*f;
    typefl G4 = a4 + b4*f;
	#ifdef DOUBLEPRECISION
		return -L*(G2/pow((pow(L,2)-pow(G3,2)),2)+G4/pow((pow(L,2)-pow(a5,2)),2) + a6)/n_PPLN(L, T);
	#else
		return -L*(G2/powf((pow(L,2)-powf(G3,2)),2)+G4/powf((pow(L,2)-powf(a5,2)),2) + a6)/n_PPLN(L, T);
	#endif
}

__host__ __device__ typefl d2ndl2_PPLN(typefl L,typefl T){
    /*
    Returns the second-order derivative of the refractive index respect to
    the wavelength d²n/dλ².
    */
    typefl f = (T - 24.5) * (T + 570.82);
    typefl a2 = 0.0983;
    typefl a3 = 0.2020;
    typefl a4 = 189.32;
    typefl a5 = 12.52;
    typefl b2 =  4.700e-8;
    typefl b3 =  6.113e-8;
    typefl b4 =  1.516e-4;
    typefl G2 = a2+b2*f;
    typefl G3 = a3+b3*f;
    typefl G4 = a4+b4*f;
	#ifdef DOUBLEPRECISION
		typefl A =((n_PPLN(L,T)-L*dndl_PPLN(L,T))/n_PPLN(L,T))*dndl_PPLN(L,T)/L;
		typefl B = (G2/pow(pow(L,2)-pow(G3,2),3) + G4/pow(pow(L,2)-pow(a5,2),3))*4*L*L/n_PPLN(L,T);
		return A+B;
	#else
		typefl A =((n_PPLN(L,T)-L*dndl_PPLN(L,T))/n_PPLN(L,T))*dndl_PPLN(L,T)/L;
		typefl B = (G2/powf(powf(L,2)-powf(G3,2),3) + G4/powf(powf(L,2)-powf(a5,2),3))*4*L*L/n_PPLN(L,T);
		return A+B;
	#endif
}

__host__ __device__ typefl group_vel_PPLN(typefl L,typefl T){
//     Returns the group-velocity vg(λ) = c/(n(λ)-λdn/dλ).
    return C/(n_PPLN(L,T)-L*dndl_PPLN(L,T));
}

__host__ __device__ typefl gvd_PPLN(typefl L,typefl T){
    // Returns the group-velocity β(λ)=λ^3/(2πc²)(d²n/dλ²).
	#ifdef DOUBLEPRECISION
		return pow(L,3)*d2ndl2_PPLN(L, T)/(2*PI*C*C);
	#else
		return powf(L,3)*d2ndl2_PPLN(L, T)/(2*PI*C*C);
	#endif
}


//////////////////// MgO:sPPLT ////////////////////

__host__ __device__ typefl n_sPPLT(typefl L,typefl T){
//     This function returns the MgO:sPPLT extraordinary refractive index 
//     from the Sellmeier Equation.
//     Reference: Bruner et. al. Temperature-dependent Sellmeier equation for
//     the refractive index of stoichiometric lithium tantalate
//     
//     INPUTS:
//     L: wavelenght in um
//     T: temperature in degrees
//     
//     OUTPUT:
//     ne: refractive index as a funcion of wavelength
    
    typefl A =  4.502483;
    typefl B =  0.007294;
    typefl C =  0.185087;
    typefl D =  -0.02357;
    typefl E =  0.073423;
    typefl F =  0.199595;
    typefl G =  0.001;
    typefl H =  7.99724;
    typefl b =  3.483933e-8 * pow(T + 273.15,2);
    typefl c =  1.607839e-8 * pow(T + 273.15,2);
    return sqrt( A + (B+b)/(pow(L,2)-pow((C+c),2)) + E/(pow(L,2)-pow(F,2)) + G/(pow(L,2)-pow(H,2))+ D*pow(L,2));
}

__host__ __device__ typefl dndl_sPPLT(typefl L,typefl T){
//     Returns the first-order derivative of the refractive index respect to the wavelength dn/dλ.
    typefl B =  0.007294;
    typefl C =  0.185087;
    typefl D =  -0.02357;
    typefl E =  0.073423;
    typefl F =  0.199595;
    typefl G =  0.001;
    typefl H =  7.99724;
    typefl b =  3.483933e-8 * pow(T + 273.15,2);
    typefl c =  1.607839e-8 * pow(T + 273.15,2);
    return -L/n_sPPLT(L, T)*( (B+b)/pow(pow(L,2)-pow((C+c),2),2) + E/pow((pow(L,2)-pow(F,2)),2) + G/pow((pow(L,2)-pow(H,2)),2) - D );
}

__host__ __device__ typefl d2ndl2_sPPLT(typefl L,typefl T){
//     Returns the second-order derivative of the refractive index respect to the wavelength d²n/dλ².

    typefl B =  0.007294;
    typefl C =  0.185087;
    typefl E =  0.073423;
    typefl F =  0.199595;
    typefl G =  0.001;
    typefl H =  7.99724;
    typefl b =  3.483933e-8 * pow(T + 273.15,2);
    typefl c =  1.607839e-8 * pow(T + 273.15,2);
    typefl S1 = dndl_sPPLT(L, T)/L;
    typefl S2 = 4*pow(L,2)/n_sPPLT(L,T)*((B+b)/pow(pow(L,2)-pow((C+c),2),3)+E/pow((pow(L,2)-pow(F,2)),3)+G/pow((pow(L,2)-pow(H,2)),3));
    return S1+S2;
}

__host__ __device__ typefl group_vel_sPPLT(typefl L,typefl T){
//     Returns the group-velocity vg(λ) = c/(n(λ)-λdn/dλ)
    return C/(n_sPPLT(L,T)-L*dndl_sPPLT(L,T));
}

__host__ __device__ typefl gvd_sPPLT(typefl L,typefl T){
//     Returns the group-velocity β(λ)=λ^3/(2πc²)(d²n/dλ²).
    return pow(L,3)*d2ndl2_sPPLT(L, T)/(2*PI*C*C);
}


//////////////////// GaP ////////////////////

__host__ __device__ typefl n_GaP(typefl L, typefl T){
//     This function returns the GaP extraordinary refractive index from the Sellmeier Equation.
//     Reference: Wei et. al. Temperature dependent Sellmeier equation for the refractive index of GaP.
//     
//     INPUTS:
//     L: wavelenght in um
//     T: temperature in Kelvin
//     
//     OUTPUT:
//     ne: refractive index as a funcion of wavelength.
    
    typefl A =  10.926 + 7.0787e-4 * T + 1.8594e-7 * T*T;
    typefl B =  0.53718 + 5.8035e-5 * T + 1.9819e-7 * T*T;
    typefl C =  0.0911014;
    typefl D =  1504 + 0.25935 * T - 0.00023326 * T*T;
    typefl E =  758.048;
    return sqrt( A + B/(L*L-C) + D/(L*L-E) );
}

__host__ __device__ typefl dndl_GaP(typefl L, typefl T){
//     Returns the first-order derivative of the refractive index respect to the wavelength dn/dλ.
    typefl B =  0.53718 + 5.8035e-5 * T + 1.9819e-7 * T*T;
    typefl C =  0.0911014;
    typefl D =  1504 + 0.25935 * T - 0.00023326 * T*T;
    typefl E =  758.048;
    return -L/n_GaP(L,T) * ( B/pow(L*L-C,2) + D/pow(L*L-E,2) );
}

__host__ __device__ typefl d2ndl2_GaP(typefl L, typefl T){
//     Returns the second-order derivative of the refractive index respect to the wavelength d²n/dλ².
    typefl B =  0.53718 + 5.8035e-5 * T + 1.9819e-7 * T*T;
    typefl C =  0.0911014;
    typefl D =  1504 + 0.25935 * T - 0.00023326 * T*T;
    typefl E =  758.048;
    return dndl_GaP(L, T)/L + 4*L*L/n_GaP(L,T)*( B/pow(L*L-C,3) + D/pow(L*L-E,3) );
}

__host__ __device__ typefl group_vel_GaP(typefl L,typefl T){
//     Returns the group-velocity vg(λ) = c/(n(λ)-λdn/dλ)
    return C/(n_GaP(L,T)-L*dndl_GaP(L,T));
}

__host__ __device__ typefl gvd_GaP(typefl L,typefl T){
//     Returns the group-velocity β(λ)=λ^3/(2πc²)(d²n/dλ²).
    return pow(L,3)*d2ndl2_GaP(L, T)/(2*PI*C*C);
}


/*
int main (){
    typefl T=30., L = 1.5;
    printf("\nValor n = %.15f\n", n_GaP(L,T));
    printf("\nValor group_vel = %.15f\n", group_vel_GaP(L,T));
    printf("\nValor GVD = %.15f\n", gvd_GaP(L,T));
    
    return 0;
}*/
