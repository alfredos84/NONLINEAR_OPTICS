#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <string.h>
#include <curand.h>
#include "common.h"
#include "functions.h"


// Complex data type
typedef cufftDoubleComplex CC;

double pot(double a, int n){
	double aux=1;
	for (int i=0; i<n; i++)
		aux *= a;
	return aux;
}

/* FUNCTIONS */
void normalize_N(CC *in_t, int SIZE){
    for (int i = 0; i < SIZE; i++){
        in_t[i].x = in_t[i].x/SIZE;
        in_t[i].y = in_t[i].y/SIZE;
    }
}
__global__ void circulargaussian(double *dx, double *dy, CC *d_noise_norm, int SIZE){
    long int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < SIZE){
        d_noise_norm[idx].x = dx[idx];
        d_noise_norm[idx].y = dy[idx];
    }    
}

__global__ void modulus1(CC *z, double *w, int SIZE){
    
    long int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < SIZE){
        w[idx] = z[idx].x * z[idx].x + z[idx].y * z[idx].y;    
    }
}

__global__ void reduce(double* vector){
    
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

__global__ void scale(CC *a, int SIZE, double s){
    // compute idx and idy, the location of the element in the original LX*LY array 
    long int idx = blockIdx.x*blockDim.x+threadIdx.x;
    
    if ( idx < SIZE){
        a[idx].x = a[idx].x * s;
        a[idx].y = a[idx].y * s;
    }       
}

void noise_generator (CC *h_noise, int N){    
    
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
    // Create pseudo-random number generator
    CHECK_CURAND(curandCreateGenerator(&gen1, CURAND_RNG_PSEUDO_DEFAULT));
    // Set seed
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen1, seed1)); //1234ULL
    // Generate n doubles on device
    CHECK_CURAND(curandGenerateNormalDouble(gen1, dx, (size_t) N, 0.0, 1.0));
    
    // Create pseudo-random number generator
    CHECK_CURAND(curandCreateGenerator(&gen2, CURAND_RNG_PSEUDO_DEFAULT));
    // Set seed
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen2, seed2)); //123L
    // Generate n doubles on device 
    CHECK_CURAND(curandGenerateNormalDouble(gen2, dy, (size_t) N, 0.0, 1.0));
    
    circulargaussian<<< grid, block>>>(dx, dy, d_noise_norm, N);
    CHECK(cudaDeviceSynchronize());
    //CHECK(cudaGetLastError());
    
    // Cleanup 
    CHECK_CURAND(curandDestroyGenerator(gen1));
    CHECK_CURAND(curandDestroyGenerator(gen2));
    
    CHECK(cudaMemcpy(h_noise, d_noise_norm, nBytes, cudaMemcpyDeviceToHost));
    
    CHECK(cudaMemcpy(h_noise, d_noise_norm, nBytes, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dx));
    CHECK(cudaFree(dy));
    CHECK(cudaFree(d_noise_norm));
}

void input_field_T(CC *Ap, double *T, int SIZE, double T0, double POWER, char m){
    switch(m){
        case 'c' :
            printf("Wave: Continuous Wave\n");
            for (int i = 0; i < SIZE; i++){
                Ap[i].x = sqrt(POWER);
                Ap[i].y = 0;
            }
            break;
        case 'g' :
            printf("Wave: Gaussian pulse\n");
            for (int i = 0; i < SIZE; i++){
                Ap[i].x = sqrt(POWER) * exp(-(T[i]*T[i])/(2*T0*T0));
                Ap[i].y = 0;
            }
            break;
        case 's' :
            printf("Wave: Soliton\n");
            for (int i = 0; i < SIZE; i++){
                Ap[i].x = sqrt(POWER) * 1/cosh(T[i]/T0);
                Ap[i].y = 0;
            }
            break;
    }
}

void inic_vector_T(double *T, int SIZE, double T_WIDTH, double dT){
    for (int i = 0; i < SIZE; i++){
        T[i] = i * dT -T_WIDTH/2.0;
    }
}

void inic_vector_F(double *F, int SIZE, double DF){
    for (int i = 0; i < SIZE; i++){
        F[i] = i * DF - SIZE* DF/2.0;
    }
}

double inic_vector_Z(double *Z, double SIZE, double STEP){
    for (int i = 0; i < SIZE; i++)
        Z[i] = SIZE*i/(SIZE-1);
    return 0;
}

double v_max(double vp, double vi){
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

void fftshift( double *V_ss, double *v, int SIZE ){
    int i;
    int c = SIZE/2;//(int) floor((float)SIZE/2);
    for ( i = 0; i < SIZE/2; i++ ){
        V_ss[i+c]= v[i];
        V_ss[i] = v[i+c];

    }
}

// SCALE (A=aA)
__global__ void CUFFTscale(CC *a, int SIZE, int s){
    // compute idx and idy, the location of the element in the original LX*LY array 
    long int idx = blockIdx.x*blockDim.x+threadIdx.x;
    
    if ( idx < SIZE){
        a[idx].x = a[idx].x / s;
        a[idx].y = a[idx].y / s;
    }       
}

__global__ void cpx_sum_GPU (CC *a, CC *b, CC *c, int SIZE){
    long int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < SIZE){
        c[idx].x = a[idx].x + b[idx].x ;
        c[idx].y = a[idx].y + b[idx].y ;
    }
}

__global__ void cpx_prod_GPU (CC *a, CC *b, CC *c, int SIZE){
    long int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < SIZE){
        c[idx].x = a[idx].x * b[idx].x - a[idx].y * b[idx].y ;
        c[idx].y = a[idx].x * b[idx].y + a[idx].y * b[idx].x ;
    }
}

 __global__ void dAdz( CC *Ap, CC *As, CC *dAp, CC *dAs, double kp, double ks, double dk, double z, int SIZE ){
    long int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < SIZE){
		dAp[idx].x = -kp*( As[idx].x*As[idx].y + As[idx].y*As[idx].x )*cos(dk*z) + kp*( As[idx].x*As[idx].x - As[idx].y*As[idx].y )*sin(dk*z) ;
		dAp[idx].y = kp*( As[idx].x*As[idx].x - As[idx].y*As[idx].y )*cos(dk*z) + kp*( As[idx].x*As[idx].y + As[idx].y*As[idx].x )*sin(dk*z) ;

		dAs[idx].x = -ks*( Ap[idx].x*As[idx].x + Ap[idx].y*As[idx].y )*sin(dk*z) - ks*( Ap[idx].y*As[idx].x - Ap[idx].x*As[idx].y )*cos(dk*z) ;
		dAs[idx].y = ks*( Ap[idx].x*As[idx].x + Ap[idx].y*As[idx].y )*cos(dk*z) - ks*( Ap[idx].y*As[idx].x - Ap[idx].x*As[idx].y )*sin(dk*z) ;
	}
}

__global__ void lineal(CC *Ap, CC *As, CC *kp, CC *ks, CC *auxp, CC *auxs, int SIZE, double s){
    
    long int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < SIZE){
        auxp[idx].x = Ap[idx].x + kp[idx].x * s;
        auxp[idx].y = Ap[idx].y + kp[idx].y * s;
		auxs[idx].x = As[idx].x + ks[idx].x * s;
        auxs[idx].y = As[idx].y + ks[idx].y * s;
    }
}

__global__ void rk4(CC *Ap, CC *As,CC *k1p, CC *k1s, CC *k2p, CC *k2s,CC *k3p, CC *k3s,CC *k4p, CC *k4s, double dz, int SIZE){
    
    long int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < SIZE){
        Ap[idx].x = Ap[idx].x + (k1p[idx].x + 2*k2p[idx].x + 2*k3p[idx].x + k4p[idx].x) * dz / 6;
		Ap[idx].y = Ap[idx].y + (k1p[idx].y + 2*k2p[idx].y + 2*k3p[idx].y + k4p[idx].y) * dz / 6;
        As[idx].x = As[idx].x + (k1s[idx].x + 2*k2s[idx].x + 2*k3s[idx].x + k4s[idx].x) * dz / 6;
		As[idx].y = As[idx].y + (k1s[idx].y + 2*k2s[idx].y + 2*k3s[idx].y + k4s[idx].y) * dz / 6;
	}
}

__global__ void linear_operator(CC *auxp, CC *auxs, CC *Apw, CC* Asw, double *w, double vm, double vp, double vs, double b2p, double b2s, int SIZE, double z){
	
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < SIZE){		
        auxp[idx].x = Apw[idx].x * cos(z*(w[idx]*(1/vp-1/vm)+0.5*w[idx]*w[idx]*b2p)) - Apw[idx].y * sin(z*(w[idx]*(1/vp-1/vm)+0.5*w[idx]*w[idx]*b2p));
		auxp[idx].y = Apw[idx].y * cos(z*(w[idx]*(1/vp-1/vm)+0.5*w[idx]*w[idx]*b2p)) + Apw[idx].x * sin(z*(w[idx]*(1/vp-1/vm)+0.5*w[idx]*w[idx]*b2p));
        auxs[idx].x = Asw[idx].x * cos(z*(w[idx]*(1/vs-1/vm)+0.5*w[idx]*w[idx]*b2s)) - Asw[idx].y * sin(z*(w[idx]*(1/vs-1/vm)+0.5*w[idx]*w[idx]*b2s));
		auxs[idx].y = Asw[idx].y * cos(z*(w[idx]*(1/vs-1/vm)+0.5*w[idx]*w[idx]*b2s)) + Asw[idx].x * sin(z*(w[idx]*(1/vs-1/vm)+0.5*w[idx]*w[idx]*b2s));
	}
	if (idx < SIZE){
		Apw[idx].x = auxp[idx].x;
		Apw[idx].y = auxp[idx].y;
		Asw[idx].x = auxs[idx].x;
		Asw[idx].y = auxs[idx].y;		
	}	
}

__global__ void equal(CC *Apw, CC *auxp, int SIZE){
	
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < SIZE){
		Apw[idx].x = auxp[idx].x;
		Apw[idx].y = auxp[idx].y;
	}	
}

__global__ void add_phase(CC *As, CC *aux, float R, double delta, int nn, int SIZE){
	const double PI   = 3.14159265358979323846;          //pi
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
    if (idx < SIZE){
        aux[idx].x = sqrt(R) * ( As[idx].x * cos(PI*(nn+delta)) - As[idx].y * sin(PI*(nn+delta)) );
		aux[idx].y = sqrt(R) * ( As[idx].y * cos(PI*(nn+delta)) + As[idx].x * sin(PI*(nn+delta)) );
	}
    if (idx < SIZE){
		As[idx].x = aux[idx].x;
		As[idx].y = aux[idx].y;
	}	
}

__global__ void add_GDD(CC *As, CC *aux, double *w, double GDD, int SIZE){
	
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
    if (idx < SIZE){
        aux[idx].x = As[idx].x * cos(0.5*w[idx]*w[idx]*GDD) - As[idx].y * sin(0.5*w[idx]*w[idx]*GDD);
        aux[idx].y = As[idx].x * sin(0.5*w[idx]*w[idx]*GDD) + As[idx].y * cos(0.5*w[idx]*w[idx]*GDD);
	}
    if (idx < SIZE){
		As[idx].x = aux[idx].x;
		As[idx].y = aux[idx].y;
	}	
}

__global__ void read_pump(CC *Ap_total, CC *Ap, int N_rt, int nn, int N_ps, int extra_win){
		
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	if(nn == 0){
		if (idx < (N_ps+extra_win)){
			Ap[idx].x = Ap_total[idx].x;
			Ap[idx].y = Ap_total[idx].y;
		}
	}
	else if(nn > 0 && nn < (N_rt-2)){
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

void evol_in_crystal(double *w_ext_gpu, dim3 grid, dim3 block, CC *Ap_gpu, CC *As_gpu, CC *Apw_gpu, CC *Asw_gpu, CC *k1p_gpu, CC *k1s_gpu, CC *k2p_gpu, CC *k2s_gpu, CC *k3p_gpu, CC *k3s_gpu, CC *k4p_gpu, CC *k4s_gpu, CC *auxp_gpu, CC *auxs_gpu, double vm, double vp, double vs, double b2p, double b2s, double kp, double ks, double dk, double dz, int steps_z, int SIZE, int nBytes, cufftHandle plan){
	double z = 0;
	for (int s = 0; s < steps_z; s++){
		/* First RK4 for dz/2 */
		//k1 = dAdz(kappas,dk,z,A)
		dAdz<<<grid,block>>>( Ap_gpu, As_gpu, k1p_gpu, k1s_gpu, kp, ks, dk, z, SIZE );
		CHECK(cudaDeviceSynchronize()); 
		//k2 = dAdz(kappas,dk,z+dz/2,A+k1/2) -> aux = A+k1/2
		lineal<<<grid,block>>>( Ap_gpu, As_gpu, k1p_gpu, k1s_gpu, auxp_gpu, auxs_gpu, SIZE, 0.5 );
		CHECK(cudaDeviceSynchronize());   
		dAdz<<<grid,block>>>( auxp_gpu, auxs_gpu, k2p_gpu, k2s_gpu, kp, ks, dk, z+dz/4, SIZE );
		CHECK(cudaDeviceSynchronize());
		// k3 = dAdz(kappas,dk,z+dz/2,A+k2/2)
		lineal<<<grid,block>>>( Ap_gpu, As_gpu, k2p_gpu, k2s_gpu, auxp_gpu, auxs_gpu, SIZE, 0.5 );
		CHECK(cudaDeviceSynchronize());   
		dAdz<<<grid,block>>>( auxp_gpu, auxs_gpu, k3p_gpu, k3s_gpu, kp, ks, dk, z+dz/4, SIZE );
		CHECK(cudaDeviceSynchronize());
		// k4 = dAdz(kappas,dk,z+dz,A+k3)
		lineal<<<grid,block>>>( Ap_gpu, As_gpu, k3p_gpu, k3s_gpu, auxp_gpu, auxs_gpu, SIZE, 1.0 );
		CHECK(cudaDeviceSynchronize());   
		dAdz<<<grid,block>>>( auxp_gpu, auxs_gpu, k4p_gpu, k4s_gpu, kp, ks, dk, z+dz/2, SIZE );
		CHECK(cudaDeviceSynchronize());
		// A = A+(k1+2*k2+2*k3+k4)*dz/6
		rk4<<<grid,block>>>( Ap_gpu, As_gpu,k1p_gpu, k1s_gpu, k2p_gpu, k2s_gpu,k3p_gpu, k3s_gpu,k4p_gpu, k4s_gpu,  dz/2, SIZE );
		CHECK(cudaDeviceSynchronize());

		// Linear operator for dz
		cufftExecZ2Z(plan, (CC *)As_gpu, (CC *)Asw_gpu, CUFFT_INVERSE);
		CHECK(cudaDeviceSynchronize());
		CUFFTscale<<<grid,block>>>(Asw_gpu, SIZE, SIZE);
		CHECK(cudaDeviceSynchronize());
		cufftExecZ2Z(plan, (CC *)Ap_gpu, (CC *)Apw_gpu, CUFFT_INVERSE);
		CHECK(cudaDeviceSynchronize());
		CUFFTscale<<<grid,block>>>(Apw_gpu, SIZE, SIZE);
		CHECK(cudaDeviceSynchronize());
		linear_operator<<<grid,block>>>( auxp_gpu, auxs_gpu, Apw_gpu, Asw_gpu, w_ext_gpu, vm, vp, vs, b2p, b2s, SIZE, dz);
		CHECK(cudaDeviceSynchronize());
		cufftExecZ2Z(plan, (CC *)Asw_gpu, (CC *)As_gpu, CUFFT_FORWARD);
		CHECK(cudaDeviceSynchronize());
		cufftExecZ2Z(plan, (CC *)Apw_gpu, (CC *)Ap_gpu, CUFFT_FORWARD);
		CHECK(cudaDeviceSynchronize());

		//k1 = dAdz(kappas,dk,z,A)
		dAdz<<<grid,block>>>( Ap_gpu, As_gpu, k1p_gpu, k1s_gpu, kp, ks, dk, z, SIZE );
		CHECK(cudaDeviceSynchronize()); 
		//k2 = dAdz(kappas,dk,z+dz/2,A+k1/2) -> aux = A+k1/2
		lineal<<<grid,block>>>( Ap_gpu, As_gpu, k1p_gpu, k1s_gpu, auxp_gpu, auxs_gpu, SIZE, 0.5 );
		CHECK(cudaDeviceSynchronize());   
		dAdz<<<grid,block>>>( auxp_gpu, auxs_gpu, k2p_gpu, k2s_gpu, kp, ks, dk, z+dz/4, SIZE );
		CHECK(cudaDeviceSynchronize());
		// k3 = dAdz(kappas,dk,z+dz/2,A+k2/2)
		lineal<<<grid,block>>>( Ap_gpu, As_gpu, k2p_gpu, k2s_gpu, auxp_gpu, auxs_gpu, SIZE, 0.5 );
		CHECK(cudaDeviceSynchronize());   
		dAdz<<<grid,block>>>( auxp_gpu, auxs_gpu, k3p_gpu, k3s_gpu, kp, ks, dk, z+dz/4, SIZE );
		CHECK(cudaDeviceSynchronize());
		// k4 = dAdz(kappas,dk,z+dz,A+k3)
		lineal<<<grid,block>>>( Ap_gpu, As_gpu, k3p_gpu, k3s_gpu, auxp_gpu, auxs_gpu, SIZE, 1.0 );
		CHECK(cudaDeviceSynchronize());   
		dAdz<<<grid,block>>>( auxp_gpu, auxs_gpu, k4p_gpu, k4s_gpu, kp, ks, dk, z+dz/2, SIZE );
		CHECK(cudaDeviceSynchronize());
		// A = A+(k1+2*k2+2*k3+k4)*dz/6
		rk4<<<grid,block>>>( Ap_gpu, As_gpu,k1p_gpu, k1s_gpu, k2p_gpu, k2s_gpu,k3p_gpu, k3s_gpu,k4p_gpu, k4s_gpu,  dz/2, SIZE );
		CHECK(cudaDeviceSynchronize());
		z+=dz;
	}
}

__global__ void write_field(CC *As_total, CC *As, int nn, int N_ps, int extra_win, int N_rt){
		
	long long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
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
			As_total[idx + N_rt*N_ps - N_ps+ 1].x = As[idx + extra_win].x;
			As_total[idx + N_rt*N_ps - N_ps+ 1].y = As[idx + extra_win].y;
		}
	}	
}
