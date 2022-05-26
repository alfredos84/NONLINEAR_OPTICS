#ifndef _FFTCUH
#define _FFTCUH



/**
 * This file contains the direct and inverse Fourier trasmorm performed in GPU.
 * 
 * The usual definition for the Fourier transform in this contexts is the 
 * opposite of that used in CUDA:
 * 
 *  fft: computes de FFT from the frequency to the time domain
 * ifft: computes de FFT from the time to the frequency domain
 */



#include <iostream>

#include <cuda_runtime.h>
#include <cufft.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/generate.h>
#include <thrust/transform.h>


using cVec_d = thrust::device_vector<cuFloatComplex>;  // define a data type for complex vectors on device


struct CufftNormalize // This functor normalizes by N the CUFFT_INVERSE 
{
	float Norm;
	CufftNormalize(float N) {Norm = N;};
	__device__ cuFloatComplex operator()(cuFloatComplex a){return make_float2(a.x/Norm, a.y/Norm);}
};


void fft ( cVec_d &fw, cVec_d &ft, cufftHandle plan )
{
	
	cuFloatComplex * ptr_ft  = thrust::raw_pointer_cast(&ft[0]);
	cuFloatComplex * ptr_fw  = thrust::raw_pointer_cast(&fw[0]);
	cufftExecC2C( plan, ptr_fw, ptr_ft, CUFFT_FORWARD );
	
	return ;
	
}



void ifft ( cVec_d &ft, cVec_d &fw, cufftHandle plan )
{
	unsigned int N = ft.size();

	cuFloatComplex * ptr_ft  = thrust::raw_pointer_cast(&ft[0]);
	cuFloatComplex * ptr_fw  = thrust::raw_pointer_cast(&fw[0]);
	
	cufftExecC2C( plan, ptr_ft, ptr_fw, CUFFT_INVERSE );
	thrust::transform( fw.begin(), fw.end(), fw.begin(), CufftNormalize( (float)N ) ); 

	return ;
	
}


#endif // -> #ifdef _FFTCUH
