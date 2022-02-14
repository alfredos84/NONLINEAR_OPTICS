#pragma once

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
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
// #include <curand.h>

#include "common.h"
#include "refindex.h"
#include "SaveFiles.h"


// Complex data type
#ifdef DOUBLEPRECISION
	typedef cufftDoubleComplex CC;
	typedef double typefl_t;
#else
	typedef cufftComplex CC;
	typedef float typefl_t;
#endif


/* FUNCTIONS */
void normalize_N(CC *in_t, int SIZE)
{
    for (int i = 0; i < SIZE; i++){
        in_t[i].x = in_t[i].x/SIZE;
        in_t[i].y = in_t[i].y/SIZE;
    }
}

__global__ void modulus1(CC *z, typefl_t *w, int SIZE){
    
    long int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < SIZE){
        w[idx] = z[idx].x * z[idx].x + z[idx].y * z[idx].y;    
    }
}

__global__ void reduce(typefl_t* vector){
    
    int index = threadIdx.x + blockIdx.x * blockDim.x ;
    int tid = threadIdx.x;
    int i = blockDim.x/2;
    
    while (i != 0) {
        if (tid < i)
            vector[index] += vector[index + i];
        __syncthreads();
        i /= 2;
    }
}

__global__ void Scale(CC *a, int SIZE, double s){
    // compute idx and idy, the location of the element in the original LX*LY array 
    long int idx = blockIdx.x*blockDim.x+threadIdx.x;
    
    if ( idx < SIZE){
        a[idx].x = a[idx].x * s;
        a[idx].y = a[idx].y * s;
    }       
}


void NoiseGeneratorCPU ( CC *As,  unsigned int SIZE )
{

	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0.0,1.0e-15);
	typefl_t nsx, nsy;
    
	for (int i=0; i<SIZE; ++i) {
		nsx = distribution(generator); As[i].x = nsx;
		nsy = distribution(generator); As[i].y = nsy;
	}
	
	return ;
	
}
/*
__global__ void circulargaussian(typefl_t *dx, typefl_t *dy, CC *d_noise_norm, int SIZE)
{
    unsigned long int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < SIZE){
        d_noise_norm[idx].x = dx[idx];
        d_noise_norm[idx].y = dy[idx];
    }
}


void NoiseGeneratorCPU (CC *h_noise, int N)
{
    
    int nBytes =  sizeof(CC)*N;    
    // parameters for kernels
    int numthreads = 1 << 5;
    int numblocks = (N + numthreads - 1) / numthreads;
    dim3 block(numthreads);
    dim3 grid(numblocks);
    
    // Allocate vectors on device
    double *dx, *dy;
    double *h_mod = (double *)malloc(N*sizeof(double));
    CC *d_noise_norm;
    
    // Allocate n doubles on device
    CHECK(cudaMalloc((void **)&dx, N*sizeof(double)));
    CHECK(cudaMalloc((void **)&dy, N*sizeof(double)));
    CHECK(cudaMalloc((void **)&d_noise_norm, nBytes));
    CHECK(cudaMemset(dx, 0, sizeof(double)*N));
    CHECK(cudaMemset(dx, 0, sizeof(double)*N));
    CHECK(cudaMemset(d_noise_norm, 0, nBytes));
    srand(time(0));
	printf("\nCheking different seeds for noise generator:\n");
    long int seed1 = rand() % 100 + 1; printf("Seed 1 = %zu\n", seed1);
    long int seed2 = rand() % 100 + 1; printf("Seed 2 = %zu\n", seed2);
    if (seed1 != seed2)
		printf("Noise OK!\n");
	else
		printf("Noise not OK :\n");
	curandGenerator_t gen1, gen2;
    double mean = 0.0, std = 1.0e-15;
    // Create pseudo-random number generator
    CHECK_CURAND(curandCreateGenerator(&gen1, CURAND_RNG_PSEUDO_DEFAULT));
    // Set seed
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen1, seed1)); //1234ULL
    // Generate n doubles on device
    CHECK_CURAND(curandGenerateNormalDouble(gen1, dx, (size_t) N, mean, std));
    
    // Create pseudo-random number generator
    CHECK_CURAND(curandCreateGenerator(&gen2, CURAND_RNG_PSEUDO_DEFAULT));
    // Set seed
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen2, seed2)); //123L
    // Generate n doubles on device 
    CHECK_CURAND(curandGenerateNormalDouble(gen2, dy, (size_t) N, mean, std));
    
    circulargaussian<<< grid, block>>>(dx, dy, d_noise_norm, N);
    CHECK(cudaDeviceSynchronize());
    //CHECK(cudaGetLastError());
    
    // Cleanup 
    CHECK_CURAND(curandDestroyGenerator(gen1));
    CHECK_CURAND(curandDestroyGenerator(gen2));
    
    CHECK(cudaMemcpy(h_noise, d_noise_norm, nBytes, cudaMemcpyDeviceToHost));
	
    CHECK(cudaFree(dx));    CHECK(cudaFree(dy));
    CHECK(cudaFree(d_noise_norm));
}*/

void input_field_T(CC *Ap, typefl_t *T, int SIZE, typefl_t T0, typefl_t POWER, char m)
{
	switch(m){
		case 'c' :
			std::cout << "Wave:                   Continuous Wave\n" << std::endl;
			for (int i = 0; i < SIZE; i++){
				Ap[i].x = sqrt(POWER);
				Ap[i].y = 0;
			}
			break;
		case 'g' :
			std::cout << "Wave:                   Gaussian pulse\n" << std::endl;
			for (int i = 0; i < SIZE; i++){
				Ap[i].x = sqrt(POWER) * exp(-(T[i]*T[i])/(2*T0*T0));
				Ap[i].y = 0;
			}
			break;
		case 's' :
			std::cout << "Wave:                   Soliton\n" << std::endl;	
			for (int i = 0; i < SIZE; i++){
				Ap[i].x = sqrt(POWER) * 1/cosh(T[i]/T0);
				Ap[i].y = 0;
			}
		break;
	}
}

void inic_vector_T(typefl_t *T, int SIZE, typefl_t T_WIDTH, typefl_t dT)
{
    for (int i = 0; i < SIZE; i++){
        T[i] = i * dT -T_WIDTH/2.0;
    }
    return ;
}

void inic_vector_F(typefl_t *F, int SIZE, typefl_t DF)
{
    for (int i = 0; i < SIZE; i++){
        F[i] = i * DF - SIZE* DF/2.0;
    }
    
    return ;
}

void inic_vector_Z(typefl_t *Z, typefl_t SIZE, typefl_t STEP)
{
    for (int i = 0; i < SIZE; i++)
        Z[i] = SIZE*i/(SIZE-1);
	
    return ;
}

typefl_t v_max(typefl_t vp, typefl_t vi)
{
	if(vp<=vi){
		return vi;
	}
	else{
		return vp;
	}
}

int factorial( int number ){
    if( number <= 1 ){
        return 1;
    } /* end if */
    else{
        return ( number * factorial( number - 1 ) );
    }
}

void cpx_sum (CC *a, CC *b, CC *c, int SIZE){
    for(int i = 0; i<SIZE; i++){
        c[i].x = a[i].x + b[i].x;
        c[i].y = a[i].y + b[i].y;
    }
}

void cpx_prod (CC *a, CC *b, CC *c, int SIZE){
    for(int i = 0; i<SIZE; i++){
        c[i].x = a[i].x * b[i].x - a[i].y * b[i].y;
        c[i].y = a[i].x * b[i].y + a[i].y * b[i].x;
    }
}

void cpx_mod (CC *a, CC *b, int SIZE){
    for(int i = 0; i<SIZE; i++){
        b[i].x = a[i].x * a[i].x + a[i].y * a[i].y;
        b[i].y = 0;
    }
} // return b = |a|^2

void fftshift( typefl_t *V_ss, typefl_t *v, int SIZE ){
    int i;
    int c = SIZE/2;//(int) floor((float)SIZE/2);
    for ( i = 0; i < SIZE/2; i++ ){
        V_ss[i+c]= v[i];
        V_ss[i] = v[i+c];

    }
}


// SCALE (A=aA)
__global__ void CUFFTscale(CC *a, int SIZE, typefl_t s)
{
    // compute idx and idy, the location of the element in the original LX*LY array 
    unsigned long int idx = blockIdx.x*blockDim.x+threadIdx.x;
    
    if ( idx < SIZE){
        a[idx].x = a[idx].x / s;
        a[idx].y = a[idx].y / s;
    }       
}


__global__ void ComplexSumGPU (CC *a, CC *b, CC *c, int nx, int ny)
{
	unsigned long int column = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned long int row = blockIdx.y;
    
    if( column < nx and row < ny){
	  c[row*nx+column].x = a[row*nx+column].x + b[row*nx+column].x ;
	  c[row*nx+column].y = a[row*nx+column].y + b[row*nx+column].y ;
    }
}

__global__ void ComplexProductGPU (CC *c, CC *a, CC *b, int nx, int ny )
{

	unsigned long int column = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned long int row = blockIdx.y;
    
    if( column < nx and row < ny){
	  c[row*nx+column].x = a[row*nx+column].x * b[row*nx+column].x - a[row*nx+column].y * b[row*nx+column].y ;
	  c[row*nx+column].y = a[row*nx+column].x * b[row*nx+column].y + a[row*nx+column].y * b[row*nx+column].x ;
    }
    
    return ;
}


__global__ void SquareVectorsGPU( typefl_t *Squared, CC *Field, int SIZE )
{
	
    unsigned long int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if( idx < SIZE )
        Squared[idx] = Field[idx].x*Field[idx].x + Field[idx].y*Field[idx].y;
 
	return;
}


__global__ void dAdz( CC *dAp, CC *dAs, CC *Ap, CC *As, typefl_t lp, typefl_t ls, typefl_t Temperature, typefl_t deff, typefl_t Lambda, typefl_t z, int SIZE )
 {
    
	unsigned long int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
	#ifdef DOUBLEPRECISION
		typefl_t np = n_PPLN(lp,Temperature);	typefl_t kp = 2*PI*deff/lp/np; 
		typefl_t ns = n_PPLN(ls,Temperature);	typefl_t ks = 2*PI*deff/ls/ns;
		typefl_t dk = 2*PI*(np/lp - 2*ns/ls - 1/Lambda);
		
		if (idx < SIZE){
			dAp[idx].x = -kp*( As[idx].x*As[idx].y + As[idx].y*As[idx].x )*cos(dk*z) + kp*( As[idx].x*As[idx].x - As[idx].y*As[idx].y )*sin(dk*z) ;
			dAp[idx].y = +kp*( As[idx].x*As[idx].x - As[idx].y*As[idx].y )*cos(dk*z) + kp*( As[idx].x*As[idx].y + As[idx].y*As[idx].x )*sin(dk*z) ;
			dAs[idx].x = -ks*( Ap[idx].x*As[idx].x + Ap[idx].y*As[idx].y )*sin(dk*z) - ks*( Ap[idx].y*As[idx].x - Ap[idx].x*As[idx].y )*cos(dk*z) ;
			dAs[idx].y = +ks*( Ap[idx].x*As[idx].x + Ap[idx].y*As[idx].y )*cos(dk*z) - ks*( Ap[idx].y*As[idx].x - Ap[idx].x*As[idx].y )*sin(dk*z) ;
		}
	#else
		typefl_t np = n_PPLN(lp,Temperature);	typefl_t kp = 2*PI*deff/lp/np; 
		typefl_t ns = n_PPLN(ls,Temperature);	typefl_t ks = 2*PI*deff/ls/ns;
		typefl_t dk = 2*PI*(np/lp - 2*ns/ls - 1/Lambda);
		
		if (idx < SIZE){
			dAp[idx].x = -kp*( As[idx].x*As[idx].y + As[idx].y*As[idx].x )*cosf(dk*z) + kp*( As[idx].x*As[idx].x - As[idx].y*As[idx].y )*sinf(dk*z) ;
			dAp[idx].y = +kp*( As[idx].x*As[idx].x - As[idx].y*As[idx].y )*cosf(dk*z) + kp*( As[idx].x*As[idx].y + As[idx].y*As[idx].x )*sinf(dk*z) ;
			dAs[idx].x = -ks*( Ap[idx].x*As[idx].x + Ap[idx].y*As[idx].y )*sinf(dk*z) - ks*( Ap[idx].y*As[idx].x - Ap[idx].x*As[idx].y )*cosf(dk*z) ;
			dAs[idx].y = +ks*( Ap[idx].x*As[idx].x + Ap[idx].y*As[idx].y )*cosf(dk*z) - ks*( Ap[idx].y*As[idx].x - Ap[idx].x*As[idx].y )*sinf(dk*z) ;
		}
	#endif
	
	return ;
}


__global__ void LinealCombination( CC *auxp, CC *auxs, CC *Ap, CC *As, CC *kp, CC *ks, double s, int SIZE)
{

	unsigned long int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < SIZE){
		auxp[idx].x = Ap[idx].x + kp[idx].x * s;
		auxp[idx].y = Ap[idx].y + kp[idx].y * s;
		auxs[idx].x = As[idx].x + ks[idx].x * s;
		auxs[idx].y = As[idx].y + ks[idx].y * s;
	}

	return ;
}

__global__ void rk4(CC *Ap, CC *As,CC *k1p, CC *k1s, CC *k2p, CC *k2s,CC *k3p, CC *k3s,CC *k4p, CC *k4s, typefl_t dz, int SIZE){
    
	unsigned long int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
	if (idx < SIZE){
		Ap[idx].x = Ap[idx].x + (k1p[idx].x + 2*k2p[idx].x + 2*k3p[idx].x + k4p[idx].x) * dz / 6;
		Ap[idx].y = Ap[idx].y + (k1p[idx].y + 2*k2p[idx].y + 2*k3p[idx].y + k4p[idx].y) * dz / 6;
		As[idx].x = As[idx].x + (k1s[idx].x + 2*k2s[idx].x + 2*k3s[idx].x + k4s[idx].x) * dz / 6;
		As[idx].y = As[idx].y + (k1s[idx].y + 2*k2s[idx].y + 2*k3s[idx].y + k4s[idx].y) * dz / 6;
	}
	
	return ;
}

__global__ void LinearOperator(CC *auxp, CC *auxs, CC *Apw, CC* Asw, typefl_t *w, typefl_t lp, typefl_t ls, typefl_t Temperature, typefl_t alphap, typefl_t alphas, int SIZE, typefl_t z)
{
	
	unsigned long int idx = threadIdx.x + blockIdx.x * blockDim.x;

	#ifdef DOUBLEPRECISION
		typefl_t vp = group_vel_PPLN(lp,Temperature); typefl_t b2p = gvd_PPLN(lp,Temperature); typefl_t attenp = exp(-0.5*alphap*z);
		typefl_t vs = group_vel_PPLN(ls,Temperature); typefl_t b2s = gvd_PPLN(ls,Temperature); typefl_t attens = exp(-0.5*alphas*z);
		
		if (idx < SIZE){		
			auxp[idx].x = Apw[idx].x * cos(z*(w[idx]*(1/vs-1/vp)+0.5*w[idx]*w[idx]*b2p)) - Apw[idx].y * sin(z*(w[idx]*(1/vs-1/vp)+0.5*w[idx]*w[idx]*b2p));
			auxp[idx].y = Apw[idx].y * cos(z*(w[idx]*(1/vs-1/vp)+0.5*w[idx]*w[idx]*b2p)) + Apw[idx].x * sin(z*(w[idx]*(1/vs-1/vp)+0.5*w[idx]*w[idx]*b2p));
			auxs[idx].x = Asw[idx].x * cos(z*(w[idx]*(1/vs-1/vs)+0.5*w[idx]*w[idx]*b2s)) - Asw[idx].y * sin(z*(w[idx]*(1/vs-1/vs)+0.5*w[idx]*w[idx]*b2s));
			auxs[idx].y = Asw[idx].y * cos(z*(w[idx]*(1/vs-1/vs)+0.5*w[idx]*w[idx]*b2s)) + Asw[idx].x * sin(z*(w[idx]*(1/vs-1/vs)+0.5*w[idx]*w[idx]*b2s));
		}
		if (idx < SIZE){
			Apw[idx].x = auxp[idx].x * attenp;
			Apw[idx].y = auxp[idx].y * attenp;
			Asw[idx].x = auxs[idx].x * attens;
			Asw[idx].y = auxs[idx].y * attens;
		}
	#else
		typefl_t vp = group_vel_PPLN(lp,Temperature); typefl_t b2p = gvd_PPLN(lp,Temperature); typefl_t attenp = expf(-0.5*alphap*z);
		typefl_t vs = group_vel_PPLN(ls,Temperature); typefl_t b2s = gvd_PPLN(ls,Temperature); typefl_t attens = expf(-0.5*alphas*z);
		
		if (idx < SIZE){		
			auxp[idx].x = Apw[idx].x * cosf(z*(w[idx]*(1/vs-1/vp)+0.5*w[idx]*w[idx]*b2p)) - Apw[idx].y * sinf(z*(w[idx]*(1/vs-1/vp)+0.5*w[idx]*w[idx]*b2p));
			auxp[idx].y = Apw[idx].y * cosf(z*(w[idx]*(1/vs-1/vp)+0.5*w[idx]*w[idx]*b2p)) + Apw[idx].x * sinf(z*(w[idx]*(1/vs-1/vp)+0.5*w[idx]*w[idx]*b2p));
			auxs[idx].x = Asw[idx].x * cosf(z*(w[idx]*(1/vs-1/vs)+0.5*w[idx]*w[idx]*b2s)) - Asw[idx].y * sinf(z*(w[idx]*(1/vs-1/vs)+0.5*w[idx]*w[idx]*b2s));
			auxs[idx].y = Asw[idx].y * cosf(z*(w[idx]*(1/vs-1/vs)+0.5*w[idx]*w[idx]*b2s)) + Asw[idx].x * sinf(z*(w[idx]*(1/vs-1/vs)+0.5*w[idx]*w[idx]*b2s));
		}
		if (idx < SIZE){
			Apw[idx].x = auxp[idx].x * attenp;
			Apw[idx].y = auxp[idx].y * attenp;
			Asw[idx].x = auxs[idx].x * attens;
			Asw[idx].y = auxs[idx].y * attens;
		}
	#endif
	
	return ;
}

__global__ void equal(CC *Apw, CC *auxp, int SIZE)
{
	
	unsigned long int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < SIZE){
		Apw[idx].x = auxp[idx].x;
		Apw[idx].y = auxp[idx].y;
	}
	
	return ;
}

__global__ void AddPhase(CC *As, CC *aux, typefl_t R, typefl_t delta, int nn, int SIZE)
{

	unsigned long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	#ifdef DOUBLEPRECISION
		if (idx < SIZE){
			aux[idx].x = sqrt(R) * ( As[idx].x * cos(PI*(nn+delta)) - As[idx].y * sin(PI*(nn+delta)) );
			aux[idx].y = sqrt(R) * ( As[idx].y * cos(PI*(nn+delta)) + As[idx].x * sin(PI*(nn+delta)) );
		}
		if (idx < SIZE){
			As[idx].x = aux[idx].x;
			As[idx].y = aux[idx].y;
		}
	#else
		if (idx < SIZE){
			aux[idx].x = sqrtf(R) * ( As[idx].x * cosf(PI*(nn+delta)) - As[idx].y * sinf(PI*(nn+delta)) );
			aux[idx].y = sqrtf(R) * ( As[idx].y * cosf(PI*(nn+delta)) + As[idx].x * sinf(PI*(nn+delta)) );
		}
		if (idx < SIZE){
			As[idx].x = aux[idx].x;
			As[idx].y = aux[idx].y;
		}
	#endif
	
	return ;
}

__global__ void AddGDD(CC *As, CC *aux, typefl_t *w, typefl_t GDD, int SIZE)
{
	
	unsigned long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	#ifdef DOUBLEPRECISION
		if (idx < SIZE){
			aux[idx].x = As[idx].x * cos(0.5*w[idx]*w[idx]*GDD) - As[idx].y * sin(0.5*w[idx]*w[idx]*GDD);
			aux[idx].y = As[idx].x * sin(0.5*w[idx]*w[idx]*GDD) + As[idx].y * cos(0.5*w[idx]*w[idx]*GDD);
		}
		if (idx < SIZE){
			As[idx].x = aux[idx].x;
			As[idx].y = aux[idx].y;
		}
	#else
		if (idx < SIZE){
			aux[idx].x = As[idx].x * cosf(0.5*w[idx]*w[idx]*GDD) - As[idx].y * sinf(0.5*w[idx]*w[idx]*GDD);
			aux[idx].y = As[idx].x * sinf(0.5*w[idx]*w[idx]*GDD) + As[idx].y * cosf(0.5*w[idx]*w[idx]*GDD);
		}
		if (idx < SIZE){
			As[idx].x = aux[idx].x;
			As[idx].y = aux[idx].y;
		}
	#endif
	
	return ;
}

__global__ void read_pump(CC *Ap_total, CC *Ap, int N_rt, int nn, int N_ps, int extra_win)
{
		
	unsigned long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	if(nn == 0){
		if (idx < (N_ps+extra_win)){
			Ap[idx].x = Ap_total[idx].x;
			Ap[idx].y = Ap_total[idx].y;
		}
	}
	else if(nn > 0 && nn < (N_rt-1)){
		int aux1 = extra_win/2;
		if (idx < (N_ps+extra_win)){
			Ap[idx].x = Ap_total[idx + (nn*N_ps + aux1)].x;
			Ap[idx].y = Ap_total[idx + (nn*N_ps + aux1)].y;
		}
	}
	else{
		if (idx < (N_ps+extra_win)){
			Ap[idx].x = Ap_total[idx + (N_ps*N_rt-1)-(N_ps+extra_win)].x;
			Ap[idx].y = Ap_total[idx + (N_ps*N_rt-1)-(N_ps+extra_win)].y;
		}
	}
}


void EvolutionInCrystal( typefl_t *w_ext_gpu, dim3 grid, dim3 block, CC *Ap_gpu, CC *As_gpu, CC *Apw_gpu, CC *Asw_gpu, CC *k1p_gpu, CC *k1s_gpu, CC *k2p_gpu, CC *k2s_gpu, CC *k3p_gpu, CC *k3s_gpu, CC *k4p_gpu, CC *k4s_gpu, CC *auxp_gpu, CC *auxs_gpu, typefl_t lp, typefl_t ls, typefl_t Temperature, typefl_t alphap, typefl_t alphas, typefl_t deff, typefl_t Lambda, typefl_t rho, typefl_t dz, int steps_z, int SIZE, int nBytes )
{
	// Set plan for cuFFT 1D and 2D//
	cufftHandle plan1D;
	#ifdef DOUBLEPRECISION
		cufftPlan1d(&plan1D, SIZE, CUFFT_Z2Z, 1);
	#else
		cufftPlan1d(&plan1D, SIZE, CUFFT_C2C, 1);
	#endif
		
	const typefl_t PI   = 3.14159265358979323846;          //pi

	typefl_t np = n_PPLN(lp, Temperature); typefl_t vp = group_vel_PPLN(lp, Temperature); typefl_t b2p = gvd_PPLN(lp, Temperature); 
	typefl_t kp = 2*PI*deff/(n_PPLN(lp, Temperature)*lp); // kappa pump [1/V]
	
	typefl_t ns = n_PPLN(ls, Temperature); typefl_t vs = group_vel_PPLN(ls, Temperature); typefl_t b2s = gvd_PPLN(ls, Temperature); 
	typefl_t ks = 2*PI*deff/(n_PPLN(ls, Temperature)*ls); // kappa signal [1/V]
	
	typefl_t z = 0;
	for (int s = 0; s < steps_z; s++){
		/* First RK4 for dz/2 */
		//k1 = dAdz(kappas,dk,z,A)
// 		dAdz( dAp, dAs, Ap, As, lp, ls, Temperature, deff, Lambda, z, SIZE )
		dAdz<<<grid,block>>>( k1p_gpu, k1s_gpu, Ap_gpu, As_gpu, lp, ls, Temperature, deff, Lambda, z, SIZE );
		CHECK(cudaDeviceSynchronize()); 
		//k2 = dAdz(kappas,dk,z+dz/2,A+k1/2) -> aux = A+k1/2
		LinealCombination<<<grid,block>>>( auxp_gpu, auxs_gpu, Ap_gpu, As_gpu, k1p_gpu, k1s_gpu, 0.5, SIZE );
		CHECK(cudaDeviceSynchronize());   
		dAdz<<<grid,block>>>( k2p_gpu, k2s_gpu, auxp_gpu, auxs_gpu, lp, ls, Temperature, deff, Lambda, z+dz/4, SIZE );
		CHECK(cudaDeviceSynchronize());
		// k3 = dAdz(kappas,dk,z+dz/2,A+k2/2)
		LinealCombination<<<grid,block>>>( auxp_gpu, auxs_gpu, Ap_gpu, As_gpu, k2p_gpu, k2s_gpu, 0.5, SIZE );
		CHECK(cudaDeviceSynchronize());   
		dAdz<<<grid,block>>>( k3p_gpu, k3s_gpu, auxp_gpu, auxs_gpu, lp, ls, Temperature, deff, Lambda, z+dz/4, SIZE );
		CHECK(cudaDeviceSynchronize());
		// k4 = dAdz(kappas,dk,z+dz,A+k3)
		LinealCombination<<<grid,block>>>( auxp_gpu, auxs_gpu, Ap_gpu, As_gpu, k3p_gpu, k3s_gpu, 1.0, SIZE );
		CHECK(cudaDeviceSynchronize());   
		dAdz<<<grid,block>>>( k4p_gpu, k4s_gpu, auxp_gpu, auxs_gpu, lp, ls, Temperature, deff, Lambda, z+dz/2, SIZE );
		CHECK(cudaDeviceSynchronize());
		// A = A+(k1+2*k2+2*k3+k4)*dz/6
		rk4<<<grid,block>>>( Ap_gpu, As_gpu,k1p_gpu, k1s_gpu, k2p_gpu, k2s_gpu,k3p_gpu, k3s_gpu,k4p_gpu, k4s_gpu,  dz/2, SIZE );
		CHECK(cudaDeviceSynchronize());

		// Linear operator for dz
		#ifdef DOUBLEPRECISION
			cufftExecZ2Z(plan1D, (CC *)As_gpu, (CC *)Asw_gpu, CUFFT_INVERSE);
			CHECK(cudaDeviceSynchronize());
			CUFFTscale<<<grid,block>>>(Asw_gpu, SIZE, SIZE);
			CHECK(cudaDeviceSynchronize());
			cufftExecZ2Z(plan1D, (CC *)Ap_gpu, (CC *)Apw_gpu, CUFFT_INVERSE);
			CHECK(cudaDeviceSynchronize());
			CUFFTscale<<<grid,block>>>(Apw_gpu, SIZE, SIZE);
			CHECK(cudaDeviceSynchronize());
			LinearOperator<<<grid,block>>>( auxp_gpu, auxs_gpu, Apw_gpu, Asw_gpu, w_ext_gpu, lp, ls, Temperature, alphap, alphas, SIZE, dz);
			CHECK(cudaDeviceSynchronize());
			cufftExecZ2Z(plan1D, (CC *)Asw_gpu, (CC *)As_gpu, CUFFT_FORWARD);
			CHECK(cudaDeviceSynchronize());
			cufftExecZ2Z(plan1D, (CC *)Apw_gpu, (CC *)Ap_gpu, CUFFT_FORWARD);
			CHECK(cudaDeviceSynchronize());
		#else
			cufftExecC2C(plan1D, (CC *)As_gpu, (CC *)Asw_gpu, CUFFT_INVERSE);
			CHECK(cudaDeviceSynchronize());
			CUFFTscale<<<grid,block>>>(Asw_gpu, SIZE, SIZE);
			CHECK(cudaDeviceSynchronize());
			cufftExecC2C(plan1D, (CC *)Ap_gpu, (CC *)Apw_gpu, CUFFT_INVERSE);
			CHECK(cudaDeviceSynchronize());
			CUFFTscale<<<grid,block>>>(Apw_gpu, SIZE, SIZE);
			CHECK(cudaDeviceSynchronize());
			LinearOperator<<<grid,block>>>( auxp_gpu, auxs_gpu, Apw_gpu, Asw_gpu, w_ext_gpu, lp, ls, Temperature, alphap, alphas, SIZE, dz);
			CHECK(cudaDeviceSynchronize());
			cufftExecC2C(plan1D, (CC *)Asw_gpu, (CC *)As_gpu, CUFFT_FORWARD);
			CHECK(cudaDeviceSynchronize());
			cufftExecC2C(plan1D, (CC *)Apw_gpu, (CC *)Ap_gpu, CUFFT_FORWARD);
			CHECK(cudaDeviceSynchronize());
		#endif	

		/* First RK4 for dz/2 */
		//k1 = dAdz(kappas,dk,z,A)
// 		dAdz( dAp, dAs, Ap, As, lp, ls, Temperature, deff, Lambda, z, SIZE )
		dAdz<<<grid,block>>>( k1p_gpu, k1s_gpu, Ap_gpu, As_gpu, lp, ls, Temperature, deff, Lambda, z, SIZE );
		CHECK(cudaDeviceSynchronize()); 
		//k2 = dAdz(kappas,dk,z+dz/2,A+k1/2) -> aux = A+k1/2
		LinealCombination<<<grid,block>>>( auxp_gpu, auxs_gpu, Ap_gpu, As_gpu, k1p_gpu, k1s_gpu, 0.5, SIZE );
		CHECK(cudaDeviceSynchronize());   
		dAdz<<<grid,block>>>( k2p_gpu, k2s_gpu, auxp_gpu, auxs_gpu, lp, ls, Temperature, deff, Lambda, z+dz/4, SIZE );
		CHECK(cudaDeviceSynchronize());
		// k3 = dAdz(kappas,dk,z+dz/2,A+k2/2)
		LinealCombination<<<grid,block>>>( auxp_gpu, auxs_gpu, Ap_gpu, As_gpu, k2p_gpu, k2s_gpu, 0.5, SIZE );
		CHECK(cudaDeviceSynchronize());   
		dAdz<<<grid,block>>>( k3p_gpu, k3s_gpu, auxp_gpu, auxs_gpu, lp, ls, Temperature, deff, Lambda, z+dz/4, SIZE );
		CHECK(cudaDeviceSynchronize());
		// k4 = dAdz(kappas,dk,z+dz,A+k3)
		LinealCombination<<<grid,block>>>( auxp_gpu, auxs_gpu, Ap_gpu, As_gpu, k3p_gpu, k3s_gpu, 1.0, SIZE );
		CHECK(cudaDeviceSynchronize());   
		dAdz<<<grid,block>>>( k4p_gpu, k4s_gpu, auxp_gpu, auxs_gpu, lp, ls, Temperature, deff, Lambda, z+dz/2, SIZE );
		CHECK(cudaDeviceSynchronize());
		// A = A+(k1+2*k2+2*k3+k4)*dz/6
		rk4<<<grid,block>>>( Ap_gpu, As_gpu,k1p_gpu, k1s_gpu, k2p_gpu, k2s_gpu,k3p_gpu, k3s_gpu,k4p_gpu, k4s_gpu,  dz/2, SIZE );
		CHECK(cudaDeviceSynchronize());
		z+=dz;
	}

	cufftDestroy(plan1D);
}

__global__ void WriteField(CC *As_total, CC *As, int nn, int N_ps, int extra_win, int N_rt){
		
	unsigned long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	if(nn == 0){
		if (idx < N_ps){
			As_total[idx].x = As[idx].x;
			As_total[idx].y = As[idx].y;
		}
	}
	else if(nn > 0 && nn<(N_rt-1)){
		if (idx < N_ps){
			As_total[idx + nn*N_ps].x = As[idx + extra_win/2].x;
			As_total[idx + nn*N_ps].y = As[idx + extra_win/2].y;
		}
	}
	else{
		if (idx < N_ps){
			As_total[idx + nn*N_ps].x = As[idx + extra_win].x;
			As_total[idx + nn*N_ps].y = As[idx + extra_win].y;
		}
	}	
}

__global__ void PhaseModulatorIntraCavity(CC *As_gpu, CC *aux, typefl_t m, typefl_t OMEGA, typefl_t *T, int SIZE){
	
	unsigned long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	#ifdef DOUBLEPRECISION
		if (idx < SIZE){
			aux[idx].x = As_gpu[idx].x * cos(m*sin(2*PI*OMEGA*T[idx])) - As_gpu[idx].y * sin(m*sin(2*PI*OMEGA*T[idx]));
			aux[idx].y = As_gpu[idx].x * sin(m*sin(2*PI*OMEGA*T[idx])) + As_gpu[idx].y * cos(m*sin(2*PI*OMEGA*T[idx]));
		}
		if (idx < SIZE){
			As_gpu[idx].x = aux[idx].x;
			As_gpu[idx].y = aux[idx].y;
		}
	#else
	    if (idx < SIZE){
			aux[idx].x = As_gpu[idx].x * cosf(m*sinf(2*PI*OMEGA*T[idx])) - As_gpu[idx].y * sinf(m*sinf(2*PI*OMEGA*T[idx]));
			aux[idx].y = As_gpu[idx].x * sinf(m*sinf(2*PI*OMEGA*T[idx])) + As_gpu[idx].y * cosf(m*sinf(2*PI*OMEGA*T[idx]));
		}
		if (idx < SIZE){
			As_gpu[idx].x = aux[idx].x;
			As_gpu[idx].y = aux[idx].y;
		}
	#endif
}
