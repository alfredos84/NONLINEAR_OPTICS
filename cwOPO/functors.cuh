#ifndef _FUNCTORSCUH
#define _FUNCTORSCUH



/**
 * This file contains the functors needed for the overloaded operators 
 * and the transformations done along the script. 
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
using complex_t = cuFloatComplex;
using real_t = float;

using rVech_t = thrust::host_vector<real_t>;
using rVecd_t = thrust::device_vector<real_t>;
using cVech_t = thrust::host_vector<complex_t>;
using cVecd_t = thrust::device_vector<complex_t>;	


struct RealScale // This functor scales by N, with N as a real constant
{
	real_t Norm;
	RealScale(real_t N) {Norm = N;};
	__host__ __device__
	cuFloatComplex operator()(cuFloatComplex V1)
	{
		return make_float2(V1.x*Norm,
						   V1.y*Norm);
	}
};


struct ComplexScale // This functor scales by N, with N as a complex constant
{
	complex_t Norm;
	ComplexScale(complex_t N) {Norm = N;};
	__host__ __device__
	cuFloatComplex operator()(cuFloatComplex V1)
	{
		return make_float2(V1.x*Norm.x - V1.y*Norm.y,
						   V1.x*Norm.y + V1.y*Norm.x);
		
	}
};


struct ComplexSum // This functor performs the sum
{
	__host__ __device__
	cuFloatComplex operator()(cuFloatComplex V1, cuFloatComplex V2)
	{
		return make_float2(V1.x+V2.x,
						   V1.y+V2.y);
	}
};


struct ComplexSubstract // This functor performs the substraction
{
	__host__ __device__
	cuFloatComplex operator()(cuFloatComplex V1, cuFloatComplex V2)
	{
		return make_float2( V1.x - V2.x,
							V1.y - V2.y );
	}
};


struct ComplexMult // This functor performs the multiplication
{
	__host__ __device__
	cuFloatComplex operator()(cuFloatComplex V1, cuFloatComplex V2)
	{
		return make_float2( V1.x*V2.x - V1.y*V2.y,
							V1.x*V2.y + V1.y*V2.x );
	}
};


struct ComplexMultbyRealCoef // This functor performs the a*A*B, a real
{
	real_t a;
	ComplexMultbyRealCoef(real_t A)  {a=A;};
	__host__ __device__
	cuFloatComplex operator()(cuFloatComplex V1, cuFloatComplex V2)
	{
		return make_float2( a*(V1.x*V2.x - V1.y*V2.y),
							a*(V1.x*V2.y + V1.y*V2.x) );
	}
};


struct ComplexMultbyComplexCoef // This functor performs the a*A*B, a complex
{
	complex_t a;
	ComplexMultbyComplexCoef(complex_t A)  {a=A;};
	__host__ __device__
	cuFloatComplex operator()(cuFloatComplex V1, cuFloatComplex V2)
	{
		return make_float2( a.x*(V1.x*V2.x - V1.y*V2.y) - a.y*(V1.x*V2.y + V1.y*V2.x),
					  a.y*(V1.x*V2.x - V1.y*V2.y) + a.x*(V1.x*V2.y + V1.y*V2.x) );
	}
};


struct ComplexLinearCombRealCoef // This functor performs the a*A+b*B, a,b reals
{
	real_t a, b;
	ComplexLinearCombRealCoef(real_t A, real_t B)  {a=A, b=B;};

	__host__ __device__
	cuFloatComplex operator()(cuFloatComplex V1, cuFloatComplex V2)
	{
		return make_float2(a*V1.x + b*V2.x,
						   a*V1.y + b*V2.y);
	}
};


struct Conjugate // This functor performs complex conjugation A -> A*
{
	__host__ __device__
	cuFloatComplex operator()(cuFloatComplex a)
	{
		return make_float2(a.x,
					 -a.y);
	}
};


#endif // -> #ifdef _FUNCTORSCUH
