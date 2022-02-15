#pragma once

#include <iostream>
#include <string>
#include <random>
#include <fstream>
#include <iomanip>
#include <typeinfo>
#include <cstdio>
#include <cstdlib>
#include <stdio.h>

#include <sys/time.h>

#include <cuda_runtime.h>
#include <cufft.h>

#include "common.h"
#include "refindex.h"

// Complex data type
#ifdef DOUBLEPRECISION
    typedef cufftDoubleComplex CC;
    typedef double typefl_t;
#else
    typedef cufftComplex CC;
    typedef float typefl_t;
#endif


// SCALE (A=aA)
__global__ void CUFFTscale(CC *a, int SIZE, typefl_t s)
{
    // compute idx and idy, the location of the element in the original LX*LY array 
    unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
    
    if ( idx < SIZE){
        a[idx].x = a[idx].x / s;
        a[idx].y = a[idx].y / s;
    }       
}

__global__ void ComplexSumGPU (CC *c, CC *a, CC *b, int SIZE)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < SIZE){
        c[idx].x = a[idx].x + b[idx].x ;
        c[idx].y = a[idx].y + b[idx].y ;
    }
    return ;
}

__global__ void ComplexProductGPU (CC *c, CC *a, CC *b, int SIZE)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < SIZE){
        c[idx].x = a[idx].x * b[idx].x - a[idx].y * b[idx].y ;
        c[idx].y = a[idx].x * b[idx].y + a[idx].y * b[idx].x ;
    }
    return ;
}

__global__ void SquareVectorsGPU( typefl_t *Squared, CC *Field, const int nx, const int ny )
{
    unsigned int column = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int row =  blockIdx.y;
    
    if(column<nx && row<ny)
        Squared[row*nx+column] = Field[row*nx+column].x*Field[row*nx+column].x + Field[row*nx+column].y*Field[row*nx+column].y;
    return;
}

__global__ void EqualVectorsGPU(CC *Apw, CC *auxp, int SIZE){

    unsigned  int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < SIZE){
	  Apw[idx].x = auxp[idx].x;
	  Apw[idx].y = auxp[idx].y;
    }
    return ;
}

/////////////////////// Nonlinear part /////////////////////
//  op_func_t pFunRefIndex,
//  __global__ void dAdz( CC *dAp, CC *dAs, CC *dAi, CC *Ap, CC *As, CC *Ai, typefl_t lp, typefl_t ls, typefl_t li, typefl_t Temperature, typefl_t deff, typefl_t Lambda, typefl_t z, int SIZE ){
// 	
// 	typefl_t np = n_PPLN(lp,Temperature); typefl_t ns = n_PPLN(ls,Temperature); typefl_t ni = n_PPLN(li,Temperature);
// 	typefl_t kp = 2*PI*deff/lp/np;
// 	typefl_t ks = 2*PI*deff/ls/ns;
// 	typefl_t ki = 2*PI*deff/li/ni;
//     
// 	typefl_t dk = 0*2*PI*(np/lp - ns/ls - ni/li - 1/Lambda);
// 	
// 	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
//     
// 	if (idx < SIZE){
// 		dAp[idx].x = -kp*( As[idx].x*Ai[idx].y + As[idx].y*Ai[idx].x )*cos(dk*z) + kp*( As[idx].x*Ai[idx].x - As[idx].y*Ai[idx].y )*sin(dk*z) ;
// 		dAp[idx].y = +kp*( As[idx].x*Ai[idx].x - As[idx].y*Ai[idx].y )*cos(dk*z) + kp*( As[idx].x*Ai[idx].y + As[idx].y*Ai[idx].x )*sin(dk*z) ;
// 
// 		dAs[idx].x = -ks*( Ap[idx].y*Ai[idx].x - Ap[idx].x*Ai[idx].y )*cos(dk*z) - ks*( Ap[idx].x*Ai[idx].x + Ap[idx].y*Ai[idx].y )*sin(dk*z) ;
// 		dAs[idx].y = +ks*( Ap[idx].x*Ai[idx].x + Ap[idx].y*Ai[idx].y )*cos(dk*z) - ks*( Ap[idx].y*Ai[idx].x - Ap[idx].x*Ai[idx].y )*sin(dk*z) ;
// 		
// 		dAi[idx].x = -ki*( Ap[idx].y*As[idx].x - Ap[idx].x*As[idx].y )*cos(dk*z) - ki*( Ap[idx].x*As[idx].x + Ap[idx].y*As[idx].y )*sin(dk*z) ;
// 		dAi[idx].y = +ki*( Ap[idx].x*As[idx].x + Ap[idx].y*As[idx].y )*cos(dk*z) - ki*( Ap[idx].y*As[idx].x - Ap[idx].x*As[idx].y )*sin(dk*z) ;
// 	}
// 	return ;
// }

 __global__ void dAdz( CC *dAp, CC *dAs, CC *Ap, CC *As, typefl_t lp, typefl_t ls, typefl_t Temperature, typefl_t deff, typefl_t Lambda, typefl_t z, int SIZE )
{
	
	typefl_t np = n_PPLN(lp,Temperature);	typefl_t kp = 2*PI*deff/lp/np; 
	typefl_t ns = n_PPLN(ls,Temperature);	typefl_t ks = 2*PI*deff/ls/ns;
    
	typefl_t dk = 0*2*PI*(np/lp - 2*ns/ls - 1/Lambda);
	
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
	if (idx < SIZE){
		dAp[idx].x = -kp*( As[idx].x*As[idx].y + As[idx].y*As[idx].x )*cos(dk*z) + kp*( As[idx].x*As[idx].x - As[idx].y*As[idx].y )*sin(dk*z) ;
		dAp[idx].y = +kp*( As[idx].x*As[idx].x - As[idx].y*As[idx].y )*cos(dk*z) + kp*( As[idx].x*As[idx].y + As[idx].y*As[idx].x )*sin(dk*z) ;
		dAs[idx].x = -ks*( Ap[idx].x*As[idx].x + Ap[idx].y*As[idx].y )*sin(dk*z) - ks*( Ap[idx].y*As[idx].x - Ap[idx].x*As[idx].y )*cos(dk*z) ;
		dAs[idx].y = +ks*( Ap[idx].x*As[idx].x + Ap[idx].y*As[idx].y )*cos(dk*z) - ks*( Ap[idx].y*As[idx].x - Ap[idx].x*As[idx].y )*sin(dk*z) ;
	}
	return ;
}



// __global__ void rk4( CC *Ap, CC *As, CC *Ai, CC *k1p, CC *k1s, CC *k1i, CC *k2p, CC *k2s, CC *k2i, CC *k3p, CC *k3s, CC *k3i, CC *k4p, CC *k4s, CC *k4i, typefl_t dz, int SIZE ){
//     unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
//     
// 	if (idx < SIZE){
// 		Ap[idx].x = Ap[idx].x + (k1p[idx].x + 2*k2p[idx].x + 2*k3p[idx].x + k4p[idx].x) * dz / 6;
// 		Ap[idx].y = Ap[idx].y + (k1p[idx].y + 2*k2p[idx].y + 2*k3p[idx].y + k4p[idx].y) * dz / 6;
// 		As[idx].x = As[idx].x + (k1s[idx].x + 2*k2s[idx].x + 2*k3s[idx].x + k4s[idx].x) * dz / 6;
// 		As[idx].y = As[idx].y + (k1s[idx].y + 2*k2s[idx].y + 2*k3s[idx].y + k4s[idx].y) * dz / 6;
// 		Ai[idx].x = Ai[idx].x + (k1i[idx].x + 2*k2i[idx].x + 2*k3i[idx].x + k4i[idx].x) * dz / 6;
// 		Ai[idx].y = Ai[idx].y + (k1i[idx].y + 2*k2i[idx].y + 2*k3i[idx].y + k4i[idx].y) * dz / 6;		
// 	}
//     return ;
// }

__global__ void rk4( CC *Ap, CC *As, CC *k1p, CC *k1s, CC *k2p, CC *k2s, CC *k3p, CC *k3s, CC *k4p, CC *k4s, typefl_t dz, int SIZE )
{

	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
	if (idx < SIZE){
        Ap[idx].x = Ap[idx].x + (k1p[idx].x + 2*k2p[idx].x + 2*k3p[idx].x + k4p[idx].x) * dz / 6;
		Ap[idx].y = Ap[idx].y + (k1p[idx].y + 2*k2p[idx].y + 2*k3p[idx].y + k4p[idx].y) * dz / 6;
        As[idx].x = As[idx].x + (k1s[idx].x + 2*k2s[idx].x + 2*k3s[idx].x + k4s[idx].x) * dz / 6;
		As[idx].y = As[idx].y + (k1s[idx].y + 2*k2s[idx].y + 2*k3s[idx].y + k4s[idx].y) * dz / 6;	
	}
    return ;
}

/////////////////////// Linear part /////////////////////
// __global__ void LinealCombination(CC *auxp, CC *auxs, CC *auxi, CC *Ap, CC *As, CC *Ai,  CC *kAp, CC *kAs, CC *kAi, typefl_t s, int SIZE){
// 
//     unsigned long int idx = threadIdx.x + blockIdx.x * blockDim.x;
//     
//     if (idx < SIZE){
//         auxp[idx].x = Ap[idx].x + kAp[idx].x * s;
//         auxp[idx].y = Ap[idx].y + kAp[idx].y * s;
// 		auxs[idx].x = As[idx].x + kAs[idx].x * s;
//         auxs[idx].y = As[idx].y + kAs[idx].y * s;
// 		auxi[idx].x = Ai[idx].x + kAi[idx].x * s;
//         auxi[idx].y = Ai[idx].y + kAi[idx].y * s;
//     }
//     return ;
// }

__global__ void LinealCombination(CC *auxp, CC *auxs, CC *Ap, CC *As, CC *kAp, CC *kAs, typefl_t s, int SIZE)
{

    unsigned long int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < SIZE){
        auxp[idx].x = Ap[idx].x + kAp[idx].x * s;
        auxp[idx].y = Ap[idx].y + kAp[idx].y * s;
		auxs[idx].x = As[idx].x + kAs[idx].x * s;
        auxs[idx].y = As[idx].y + kAs[idx].y * s;
    }
    return ;
}

// __global__ void LinearOperator(CC *auxp, CC *auxs, CC *auxi, CC *Apw, CC* Asw, CC* Aiw, typefl_t *w, typefl_t lp, typefl_t ls, typefl_t li, typefl_t Temperature, typefl_t alphap, typefl_t alphas, typefl_t alphai, int SIZE, typefl_t z){
// 	
// 	typefl_t vp = group_vel_PPLN(lp,Temperature); typefl_t b2p = gvd_PPLN(lp,Temperature); typefl_t attenp = exp(-0.5*alphap*z);	
// 	typefl_t vs = group_vel_PPLN(ls,Temperature); typefl_t b2s = gvd_PPLN(ls,Temperature); typefl_t attens = exp(-0.5*alphas*z);
// 	typefl_t vi = group_vel_PPLN(li,Temperature); typefl_t b2i = gvd_PPLN(li,Temperature); typefl_t atteni = exp(-0.5*alphai*z);
// 	
// 	long int idx = threadIdx.x + blockIdx.x * blockDim.x;
// 
// 	if (idx < SIZE){
// 		auxp[idx].x = Apw[idx].x * cos(z*(w[idx]*(1/vs-1/vp)+0.5*w[idx]*w[idx]*b2p)) - Apw[idx].y * sin(z*(w[idx]*(1/vs-1/vp)+0.5*w[idx]*w[idx]*b2p));
// 		auxp[idx].y = Apw[idx].y * cos(z*(w[idx]*(1/vs-1/vp)+0.5*w[idx]*w[idx]*b2p)) + Apw[idx].x * sin(z*(w[idx]*(1/vs-1/vp)+0.5*w[idx]*w[idx]*b2p));
// 		auxs[idx].x = Asw[idx].x * cos(z*(w[idx]*(1/vs-1/vs)+0.5*w[idx]*w[idx]*b2s)) - Asw[idx].y * sin(z*(w[idx]*(1/vs-1/vs)+0.5*w[idx]*w[idx]*b2s));
// 		auxs[idx].y = Asw[idx].y * cos(z*(w[idx]*(1/vs-1/vs)+0.5*w[idx]*w[idx]*b2s)) + Asw[idx].x * sin(z*(w[idx]*(1/vs-1/vs)+0.5*w[idx]*w[idx]*b2s));
// 		auxi[idx].x = Aiw[idx].x * cos(z*(w[idx]*(1/vs-1/vi)+0.5*w[idx]*w[idx]*b2i)) - Aiw[idx].y * sin(z*(w[idx]*(1/vs-1/vi)+0.5*w[idx]*w[idx]*b2i));
// 		auxi[idx].y = Aiw[idx].y * cos(z*(w[idx]*(1/vs-1/vi)+0.5*w[idx]*w[idx]*b2i)) + Aiw[idx].x * sin(z*(w[idx]*(1/vs-1/vi)+0.5*w[idx]*w[idx]*b2i));  
// 	}
// 	if (idx < SIZE){
// 		Apw[idx].x = auxp[idx].x * attenp;
// 		Apw[idx].y = auxp[idx].y * attenp;
// 		Asw[idx].x = auxs[idx].x * attens;
// 		Asw[idx].y = auxs[idx].y * attens;
// 		Aiw[idx].x = auxi[idx].x * atteni;
// 		Aiw[idx].y = auxi[idx].y * atteni;
// 	}
// 	return ;
// }

__global__ void LinearOperator(CC *auxp, CC *auxs, CC *Apw, CC* Asw, typefl_t *w, typefl_t lp, typefl_t ls, typefl_t Temperature, typefl_t alphap, typefl_t alphas, int SIZE, typefl_t z)
{
	
	typefl_t vp = group_vel_PPLN(lp,Temperature); typefl_t b2p = gvd_PPLN(lp,Temperature); typefl_t attenp = exp(-0.5*alphap*z);
	typefl_t vs = group_vel_PPLN(ls,Temperature); typefl_t b2s = gvd_PPLN(ls,Temperature); typefl_t attens = exp(-0.5*alphas*z);
	
	unsigned long int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < SIZE){
		auxp[idx].x = Apw[idx].x * cos(z*(w[idx]*(1/vs-1/vp)+0.5*w[idx]*w[idx]*b2p)) - Apw[idx].y * sin(z*(w[idx]*(1/vs-1/vp)+0.5*w[idx]*w[idx]*b2p));
		auxp[idx].y = Apw[idx].y * cos(z*(w[idx]*(1/vs-1/vp)+0.5*w[idx]*w[idx]*b2p)) + Apw[idx].x * sin(z*(w[idx]*(1/vs-1/vp)+0.5*w[idx]*w[idx]*b2p));
		auxs[idx].x = Asw[idx].x * cos(z*(w[idx]*(1/vs-1/vs)+0.5*w[idx]*w[idx]*b2s)) - Asw[idx].y * sin(z*(w[idx]*(1/vs-1/vs)+0.5*w[idx]*w[idx]*b2s));
		auxs[idx].y = Asw[idx].y * cos(z*(w[idx]*(1/vs-1/vs)+0.5*w[idx]*w[idx]*b2s)) + Asw[idx].x * sin(z*(w[idx]*(1/vs-1/vs)+0.5*w[idx]*w[idx]*b2s));
	}
	if (idx < SIZE){
		Apw[idx].x = auxp[idx].x;
		Apw[idx].y = auxp[idx].y;
		Asw[idx].x = auxs[idx].x;
		Asw[idx].y = auxs[idx].y;
	}
// 	if (idx < SIZE){
// 		Apw[idx].x = auxp[idx].x * attenp;
// 		Apw[idx].y = auxp[idx].y * attenp;
// 		Asw[idx].x = auxs[idx].x * attens;
// 		Asw[idx].y = auxs[idx].y * attens;
// 	}
	return ;
}


__global__ void AddPhase(CC *Field, CC *aux, typefl_t R, typefl_t delta, int nn, int SIZE)
{
    long int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < SIZE){
	  aux[idx].x = sqrt(R) * ( Field[idx].x * cos(PI*(nn+delta)) - Field[idx].y * sin(PI*(nn+delta)) );
	  aux[idx].y = sqrt(R) * ( Field[idx].y * cos(PI*(nn+delta)) + Field[idx].x * sin(PI*(nn+delta)) );
    }
    if (idx < SIZE){
	  Field[idx].x = aux[idx].x;
	  Field[idx].y = aux[idx].y;
    }
    return ;
}


__global__ void AddGDD(CC *Field, CC *aux, typefl_t *w, typefl_t GDD, int SIZE)
{
	
	unsigned long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
    if (idx < SIZE){
        aux[idx].x = Field[idx].x * cos(0.5*w[idx]*w[idx]*GDD) - Field[idx].y * sin(0.5*w[idx]*w[idx]*GDD);
        aux[idx].y = Field[idx].x * sin(0.5*w[idx]*w[idx]*GDD) + Field[idx].y * cos(0.5*w[idx]*w[idx]*GDD);
	}
    if (idx < SIZE){
		Field[idx].x = aux[idx].x;
		Field[idx].y = aux[idx].y;
	}
	return ;
}


__global__ void SetField2D2DGPU(CC *Field2D, const unsigned int nx, const unsigned int ny, typefl_t E0, typefl_t waist)
{

    unsigned long int column = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned long int row = blockIdx.y;

    if( column < nx and row < ny){
		Field2D[row*nx+column].x = E0*exp(-2*(pow(column-nx/2,2)+pow(row-ny/2,2))/pow(waist,2));
		Field2D[row*nx+column].y = 0.0;
    }
    
    return;
}


__global__ void BeamPropagator ( CC *eiQz, typefl_t lambda, typefl_t Temperature, typefl_t rho, typefl_t alpha, const unsigned int nx, const unsigned int ny,  typefl_t dx,  typefl_t dy, typefl_t z )
{
	
    typefl_t refindex = n_PPLN(lambda, Temperature);
    typefl_t k = 2*PI*refindex/lambda;
    typefl_t df = 1/dx/nx; 
    typefl_t atten = exp(-alpha*z/2);
    
    unsigned long int column = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned long int row = blockIdx.y;
    
    if( column < nx and row < ny){
	  eiQz[row*nx+column].x = + atten * cos( z * (2*pow(df*PI,2)/k * ( pow(column - typefl_t(nx)/2,2) + pow(row - typefl_t(ny)/2,2) ) + 2*PI*df*tan(rho)*(column-typefl_t(nx)/2))); 
	  eiQz[row*nx+column].y = - atten * sin( z * (2*pow(df*PI,2)/k * ( pow(column - typefl_t(nx)/2,2) + pow(row - typefl_t(ny)/2,2) ) + 2*PI*df*tan(rho)*(column-typefl_t(nx)/2))); 
    }
    return ;
}


__global__ void FFTShift2DH( CC *Field, CC *aux, const int nx, const int ny )
{
	unsigned long int column = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned long int row = blockIdx.y;
	unsigned long int c = nx/2; //(int) floor((typefl_t)nx/2);

	if (column < c and row < ny){
		Field[row*nx+column+c].x  =  aux[row*nx+column].x;
		Field[row*nx+column+c].y  =  aux[row*nx+column].y;
		Field[row*nx+column].x  =  aux[row*nx+column+c].x;
		Field[row*nx+column].y  =  aux[row*nx+column+c].y;
	}

	return ;
}


__global__ void FFTShift2DV( CC *Field, CC *aux, const int nx, const int ny )
{
    unsigned long int column = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned long int row = blockIdx.y;
    unsigned long int r = ny/2;
	
	if (row < r and column < nx){
		Field[(row+r)*nx+column].x  =  aux[row*nx+column].x;
		Field[(row+r)*nx+column].y  =  aux[row*nx+column].y;
		Field[row*nx+column].x  =  aux[(row+r)*nx+column].x;
		Field[row*nx+column].y  =  aux[(row+r)*nx+column].y;
	}    
	return ;
}


void FFTShift2D ( CC* d_trans, const int nx, const int ny)
{
	// invoke kernel at host side
	int dimx = 1 << 5;
	dim3 block(dimx);
	dim3 grid( (nx + block.x - 1) / block.x, ny );
	size_t nBytes =  nx*ny*sizeof(CC);
	CC *aux;
	CHECK(cudaMalloc((void **)&aux,nBytes));
	CHECK(cudaMemcpy(aux, d_trans, nBytes, cudaMemcpyDeviceToDevice));
	FFTShift2DV<<<grid, block>>>(d_trans, aux, nx, ny);
	cudaDeviceSynchronize();
	CHECK(cudaMemcpy(aux, d_trans, nBytes, cudaMemcpyDeviceToDevice));
	FFTShift2DH<<<grid, block>>>(d_trans, aux, nx, ny);
	cudaDeviceSynchronize();
	CHECK(cudaFree(aux));
	return ;
}


__global__ void ReadPump(CC *Ap_total, CC *Ap, int N_rt, int nn, int N_ps, int extra_win)
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
	return ;
}


__global__ void write_field(CC *As_total, CC *As, int nn, int N_ps, int extra_win, int N_rt)
{
		
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


__global__ void PM_signal_intra_cavity(CC *Field, CC *aux, typefl_t m, typefl_t OMEGA, typefl_t *T, int SIZE)
{
	
	unsigned long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
    if (idx < SIZE){
        aux[idx].x = Field[idx].x * cos(m*sin(2*PI*OMEGA*T[idx])) - Field[idx].y * sin(m*sin(2*PI*OMEGA*T[idx]));
        aux[idx].y = Field[idx].x * sin(m*sin(2*PI*OMEGA*T[idx])) + Field[idx].y * cos(m*sin(2*PI*OMEGA*T[idx]));
	}
    if (idx < SIZE){
		Field[idx].x = aux[idx].x;
		Field[idx].y = aux[idx].y;
	}
}
