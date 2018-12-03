#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <string.h>
#include <curand.h>
#include "common.h"

#define PI2 2.0 * 3.14159265358979323846  //2*pi
#define C 299792458*1E9/1E12  // speed of ligth in vacuum [nm/ps]
#define NELEMS(x)  (sizeof(x) / sizeof((x).x)) // number of elements of an array

// Complex data type
typedef cufftDoubleComplex CC;

	/* FUNCTIONS */
void normalize_N(CC *in_t, int SIZE){
    for (int i = 0; i < SIZE; i++){
        in_t[i].x = in_t[i].x/SIZE;
        in_t[i].y = in_t[i].y/SIZE;
    }
}
__global__ void circulargaussian(double *dx, double *dy, CC *d_noise_norm, int SIZE){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < SIZE){
        d_noise_norm[idx].x = dx[idx];
        d_noise_norm[idx].y = dy[idx];
    }    
}

// COMPUTE DE MODULUS OF A COMPLEX NUMBER (IF Z = X +iy -> |Z|^2 = X^2+Y^2)
__global__ void modulus(CC *z, double *w, int SIZE){
	
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < SIZE)
        w[idx] = z[idx].x * z[idx].x + z[idx].y * z[idx].y;    
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
        int idx = blockIdx.x*blockDim.x+threadIdx.x;
        
        if ( idx < SIZE){
                a[idx].x = a[idx].x * s;
                a[idx].y = a[idx].y * s;
        }       
}

void noise_generator (CC *h_noise, double SNR, int N, double POWER){    

	int nBytes =  sizeof(CC)*N;    
	// parameters for kernels
	int numthreads = 1 << 5;
	int numblocks = (N + numthreads - 1) / numthreads;
	dim3 block(numthreads);
	dim3 grid(numblocks);

	
	// set up device	
	/*int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("Using Device %d: %s\n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));*/

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

	curandGenerator_t gen1, gen2;
	// Create pseudo-random number generator
	CHECK_CURAND(curandCreateGenerator(&gen1, CURAND_RNG_PSEUDO_DEFAULT));
	// Set seed
	CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen1, 1234ULL));
	// Generate n doubles on device
	CHECK_CURAND(curandGenerateNormalDouble(gen1, dx, (size_t) N, 0.0, 1.0));
	
	// Create pseudo-random number generator
	CHECK_CURAND(curandCreateGenerator(&gen2, CURAND_RNG_PSEUDO_DEFAULT));
	// Set seed
	CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen2, 123L));
	// Generate n doubles on device 
	CHECK_CURAND(curandGenerateNormalDouble(gen2, dy, (size_t) N, 0.0, 1.0));
	
	circulargaussian<<< grid, block>>>(dx, dy, d_noise_norm, N);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaGetLastError());
	// Cleanup 
	CHECK_CURAND(curandDestroyGenerator(gen1));
	CHECK_CURAND(curandDestroyGenerator(gen2));
	
	CHECK(cudaMemcpy(h_noise, d_noise_norm, nBytes, cudaMemcpyDeviceToHost));

	double *d_mod;
	CHECK(cudaMalloc((void **)&d_mod, N*sizeof(double)));
	modulus<<<grid, block>>>(d_noise_norm, d_mod, N);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaGetLastError()); 
	// compute the sum using reduction
	reduce<<<grid, block>>>(d_mod);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaGetLastError()); 
	CHECK(cudaMemcpy(h_mod, d_mod, N * sizeof(double), cudaMemcpyDeviceToHost));
	double sum=0;
	for (int i=0; i<numblocks; i+=numthreads)
		sum = sum + h_mod[i];
	
	free(h_mod);
	double Pnoise = sum/N;
	double a = POWER / (Pnoise * pow(10.0,SNR/10));
	Pnoise = Pnoise * a;
	double aux = sqrt(Pnoise);
	scale<<<grid, block>>>(d_noise_norm, N, aux);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaGetLastError()); 
	// Copy device memory to host
	CHECK(cudaMemcpy(h_noise, d_noise_norm, nBytes, cudaMemcpyDeviceToHost));
	CHECK(cudaFree(d_mod));
	CHECK(cudaFree(dx));
	CHECK(cudaFree(dy));
	CHECK(cudaFree(d_noise_norm));
}

void input_field_T(CC *in_t, double *T, int SIZE, double T0, double POWER){
	
	for (int i = 0; i < SIZE; i++){
        //in_t[i].x = sqrt(POWER) * 1/cosh(T[i]/T0);
		in_t[i].x = sqrt(POWER);
		in_t[i].y = 0;
        }
}

double inic_vector_T(double *T, int SIZE, double T_WIDTH){
    for (int i = 0; i < SIZE; i++){
        T[i] = -T_WIDTH/2.0 + T_WIDTH*i/(SIZE-1);
    }
    return fabs(T[1]-T[0]); //return dT
}

double inic_vector_Traman(double *TT, int SIZE, double T_WIDTH){
    for (int i = 0; i < SIZE; i++)
        TT[i] = T_WIDTH*i/(SIZE-1);
    return 0;
}

double inic_vector_F(double *F, int SIZE, double DT){
    for (int i = 0; i < SIZE; i++){
        F[i] = (-SIZE/2.0 + SIZE*i/(SIZE-1))/(SIZE*DT);
    }
    return fabs(F[1]-F[0]); //return dF
}

double inic_vector_Z(double *Z, double SIZE, double STEP){
    for (int i = 0; i < SIZE; i++)
        Z[i] = SIZE*i/(SIZE-1);
    return 0;
}

int factorial( int number ){
    if( number <= 1 ){
        return 1;
    } /* end if */
    else{
        return ( number * factorial( number - 1 ) );
    }
}

void linear_operator(CC *dop, CC *FREC, double betas[], int length_betas, int SIZE, double STEP_Z){
    double sum;    
    for(int i = 0; i<SIZE; i++){
        sum = 0.0;
        for(int j = 0; j < length_betas; j++)
            sum = sum + betas[j] * pow( PI2*FREC[i].x, j+2 ) / factorial(j+2);
        dop[i].x = cos(STEP_Z * sum / 2.0);
        dop[i].y = sin(STEP_Z * sum / 2.0); 
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

void RAMAN_RESP(CC *HR, int SIZE, float TAU1, float TAU2, double *TIME){
    for (int i = 0; i<SIZE; i++){
        HR[i].x = (pow(TAU1,2) + pow(TAU2,2))/(TAU1 * pow(TAU2,2)) * exp(-TIME[i]/TAU2) * sin(TIME[i]/ TAU1);
        HR[i].y = 0;
    }    
}

void inic_selfst(CC *SST, CC *FREC, double OMEGA_0, double GAMMA, int SIZE){
    if(OMEGA_0 == 0){
        for (int i = 0; i<SIZE; i++){
            SST[i].x = 0;
            SST[i].y = GAMMA;    
        }
    }
    else{
        for (int i = 0; i<SIZE; i++){
            SST[i].x = 0;
            SST[i].y = GAMMA*(1 + PI2*FREC[i].x/OMEGA_0);    
        }
    }
}

/*void fftshift( CC *v, int SIZE ){
    CC *tmp =  (CC*)malloc(nBytes);;
    int i;
    for ( i = 0; i < SIZE; i++){
        tmp[i].x = v[i].x;
        tmp[i].y = v[i].y;
    }
    int c = (int) floor((float)SIZE/2);
    for ( i = 0; i < c; i++ ){
        v[i+c].x = tmp[i].x;
        v[i+c].y = tmp[i].y;
        v[i].x = tmp[i+c].x;
        v[i].y = tmp[i+c].y;
    }
    free(tmp);
}*/

void freq_shift( CC *V_ss, double *v, int SIZE ){
    int i;
    int c = (int) floor((float)SIZE/2);
    for ( i = 0; i < c; i++ ){
        V_ss[i+c].x = v[i];
        V_ss[i+c].y = 0;
        V_ss[i].x = v[i+c];
        V_ss[i].y = 0;
    }
}

// SCALE (A=aA)
__global__ void CUFFTscale(CC *a, int SIZE, int s){
        // compute idx and idy, the location of the element in the original LX*LY array 
        int idx = blockIdx.x*blockDim.x+threadIdx.x;
        
        if ( idx < SIZE){
                a[idx].x = a[idx].x / s;
                a[idx].y = a[idx].y / s;
        }       
}

__global__ void SimpleScale(double *a, int SIZE, double s){
        // compute idx and idy, the location of the element in the original LX*LY array 
        int idx = blockIdx.x*blockDim.x+threadIdx.x;
        
        if ( idx < SIZE)
                a[idx] = a[idx] * s;
}

__global__ void final(CC *a, CC *b, CC *c,  CC *d, CC *e, double s, int SIZE){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

     if (idx < SIZE){
        e[idx].x = a[idx].x + b[idx].x * s/6 + c[idx].x * s/3 + d[idx].x * s/3;
        e[idx].y = a[idx].y + b[idx].y * s/6 + c[idx].y * s/3 + d[idx].y * s/3;
    }
}

// SCALE (C=A + B*s)
__global__ void lineal(CC *a, CC *b, CC *c, int SIZE, double s){
	
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < SIZE){
		c[idx].x = a[idx].x + b[idx].x * s;
		c[idx].y = a[idx].y + b[idx].y * s;
	}
}

__global__ void KFR( CC *a, CC *b, CC *c, float FR, int SIZE ){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < SIZE){
        c[idx].x = a[idx].x * (1-FR) + b[idx].x * FR;
        c[idx].y = a[idx].y * (1-FR) + b[idx].y * FR;
    }
}

void COMPUTE_TFN( CC *g, CC *u, CC *u_W, CC *hR_W, double *FREC, CC *SST, int SIZE, float FR, int nBytes){

      // parameters for kernels
    int dimx = 1 << 5;
    dim3 block(dimx);
    dim3 grid((SIZE + block.x - 1) / block.x);
    
    cufftHandle plan;
    cufftPlan1d(&plan, SIZE, CUFFT_Z2Z, 1);
    
    CC *op1 =  (CC*)malloc(nBytes);
    CC *op1_W =  (CC*)malloc(nBytes);
    CC *op2 =  (CC*)malloc(nBytes);
    CC *op2_W =  (CC*)malloc(nBytes);
    CC *op3 =  (CC*)malloc(nBytes);
    CC *op3_W =  (CC*)malloc(nBytes);
    CC *op4_W =  (CC*)malloc(nBytes);

    cpx_mod( u, op1, SIZE ); // op1 = |u|^2

    CC *d_op1, *d_op1_W, *d_op2, *d_op2_W, *d_op3, *d_op3_W;
    CHECK(cudaMalloc((void **)&d_op1, nBytes));
    CHECK(cudaMalloc((void **)&d_op1_W, nBytes));
    CHECK(cudaMemset(d_op1, 0, nBytes));
    CHECK(cudaMemset(d_op1_W, 0, nBytes));
    CHECK(cudaMalloc((void **)&d_op2, nBytes));
    CHECK(cudaMalloc((void **)&d_op2_W, nBytes));
    CHECK(cudaMemset(d_op2, 0, nBytes));
    CHECK(cudaMemset(d_op2_W, 0, nBytes));
    CHECK(cudaMalloc((void **)&d_op3, nBytes));
    CHECK(cudaMalloc((void **)&d_op3_W, nBytes));
    CHECK(cudaMemset(d_op3, 0, nBytes));
    CHECK(cudaMemset(d_op3_W, 0, nBytes));
    
    // Copy host memory to device
    CHECK(cudaMemcpy(d_op1, op1, nBytes, cudaMemcpyHostToDevice));
    cufftExecZ2Z(plan, (CC *)d_op1, (CC *)d_op1_W, CUFFT_INVERSE);
    cudaDeviceSynchronize();
    CUFFTscale<<<grid,block>>>(d_op1_W, SIZE, SIZE);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());    
    CHECK(cudaMemcpy(op1_W, d_op1_W, nBytes, cudaMemcpyDeviceToHost));
    
    cpx_prod (op1_W, hR_W, op2_W, SIZE); // op2 = op1 * hR_W = |u|^2 * hR_W

    CHECK(cudaMemcpy(d_op2_W, op2_W, nBytes, cudaMemcpyHostToDevice));
    cufftExecZ2Z(plan, (CC *)d_op2_W, (CC *)d_op2, CUFFT_FORWARD);
    cudaDeviceSynchronize();
    CHECK(cudaMemcpy(op2, d_op2, nBytes, cudaMemcpyDeviceToHost));
    
    CC *aux =  (CC*)malloc(nBytes);
    CC *d_aux;  CHECK(cudaMalloc((void **)&d_aux, nBytes));  CHECK(cudaMemset(d_aux, 0, nBytes));
    KFR<<<grid,block>>>(d_op1, d_op2, d_aux, FR, SIZE);

    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());    
    CHECK(cudaMemcpy(aux, d_aux, nBytes, cudaMemcpyDeviceToHost));
    cpx_prod (u, aux, op3, SIZE); // op3
    free(aux); CHECK(cudaFree(d_aux));
    
    CHECK(cudaMemcpy(d_op3, op3, nBytes, cudaMemcpyHostToDevice));
    cufftExecZ2Z(plan, (CC *)d_op3, (CC *)d_op3_W, CUFFT_INVERSE);
    cudaDeviceSynchronize();
    CUFFTscale<<<grid,block>>>(d_op3_W, SIZE, SIZE);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());    
    CHECK(cudaMemcpy(op3_W, d_op3_W, nBytes, cudaMemcpyDeviceToHost));
    cpx_prod (SST, op3_W, op4_W, SIZE); // op4_W
    
    for( int i = 0; i < SIZE; i++ ){
        g[i].x = op4_W[i].x;
        g[i].y = op4_W[i].y;
    }

    free(op1); free(op1_W);
    free(op2); free(op2_W);
    free(op3); free(op3_W);
    free(op4_W);
    CHECK(cudaFree(d_op1));
    CHECK(cudaFree(d_op2));
    CHECK(cudaFree(d_op3));
    CHECK(cudaFree(d_op1_W));
    CHECK(cudaFree(d_op2_W));
    CHECK(cudaFree(d_op3_W));
    
   //Destroy CUFFT context
    cufftDestroy(plan);
}
