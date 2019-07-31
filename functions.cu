#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <string.h>
#include <curand.h>
#include "common.h"
#include "functions.h"

const double PI2 = 2.0 * 3.14159265358979323846;    //2*pi
//const double C = 299792458*1E9/1E12 ;               // speed of ligth in vacuum [nm/ps]
#define NELEMS(x)  (sizeof(x) / sizeof((x).x))      // number of elements of an array

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

__global__ void equal(CC *a, CC *b, int SIZE){
    
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < SIZE){
        a[idx].x = b[idx].x;    
        a[idx].y = b[idx].y;    
    }
}

// COMPUTE DE MODULUS OF A COMPLEX NUMBER (IF Z = X +iY -> |Z|^2 = X^2+Y^2)
__global__ void modulus(CC *z, CC *w, int SIZE){
    
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < SIZE){
        w[idx].x = z[idx].x * z[idx].x + z[idx].y * z[idx].y;    
        w[idx].y = 0;    
    }
}

__global__ void modulus1(CC *z, double *w, int SIZE){
    
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
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
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    
    if ( idx < SIZE){
        a[idx].x = a[idx].x * s;
        a[idx].y = a[idx].y * s;
    }       
}

void noise_generator (CC *h_noise, double SNR, int N, double POWER){    
    
    int nBytes =  sizeof(CC)*N;    
    // parameters for kernels
    int numthreads = 1 << 7;
    int numblocks = (N + numthreads - 1) / numthreads;
    dim3 block(numthreads);
    dim3 grid(numblocks);
    
    
    // set up device	
    /*int dev = 0;
     *	cudaDeviceProp deviceProp;
     *	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
     *	printf("Using Device %d: %s\n", dev, deviceProp.name);
     *	CHECK(cudaSetDevice(dev));*/
    
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
    int seed1 = rand() % 100 + 1; printf("Seed 1 = %d\n", seed1);
    int seed2 = rand() % 100 + 1; printf("Seed 2 = %d\n", seed2);
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
    
    double *d_mod;
    CHECK(cudaMalloc((void **)&d_mod, N*sizeof(double)));
    modulus1<<<grid, block>>>(d_noise_norm, d_mod, N);
    CHECK(cudaDeviceSynchronize());
    //CHECK(cudaGetLastError()); 
    // compute the sum using reduction
    reduce<<<grid, block>>>(d_mod);
    CHECK(cudaDeviceSynchronize());
    //CHECK(cudaGetLastError()); 
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
    //CHECK(cudaGetLastError()); 
    // Copy device memory to host
    CHECK(cudaMemcpy(h_noise, d_noise_norm, nBytes, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_mod));
    CHECK(cudaFree(dx));
    CHECK(cudaFree(dy));
    CHECK(cudaFree(d_noise_norm));
}

void input_field_T(CC *in_t, double *T, int SIZE, double T0, double POWER, char m){
    switch(m){
        case 'c' :
            printf("Wave: Continuous Wave\n");
            for (int i = 0; i < SIZE; i++){
                in_t[i].x = sqrt(POWER);
                in_t[i].y = 0;
            }
            break;
        case 'g' :
            printf("Wave: Gaussian pulse\n");
            for (int i = 0; i < SIZE; i++){
                in_t[i].x = sqrt(POWER) * exp(-0.5*T[i]*T[i]/(T0*T0));
                in_t[i].y = 0;
            }
            break;
        case 's' :
            printf("Wave: Soliton\n");
            for (int i = 0; i < SIZE; i++){
                in_t[i].x = sqrt(POWER) * 1/cosh(T[i]/T0);
                in_t[i].y = 0;
            }
            break;            
    }
}

void inic_vector_T(double *T, int SIZE, double T_WIDTH, double dT){
    for (int i = 0; i < SIZE; i++){
        T[i] = i * dT -T_WIDTH/2.0;
    }
}

void inic_vector_Traman(double *TT, int SIZE, double T_WIDTH){
    for (int i = 0; i < SIZE; i++)
        TT[i] = T_WIDTH*i/(SIZE-1);
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
        for(int j = 0; j < length_betas; j++){
            sum = sum + betas[j] * pow( PI2*FREC[i].x, j+2 ) / factorial(j+2);
        }
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

void RAMAN_RESP(CC *HR, int SIZE, double TAU1, double TAU2, double *TIME){
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

void freq_shift( CC *V_ss, double *v, int SIZE ){
    int i;
    int c = SIZE/2;//(int) floor((float)SIZE/2);
    for ( i = 0; i < SIZE/2; i++ ){
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
        a[idx].x = a[idx].x / (double)s;
        a[idx].y = a[idx].y / (double)s;
    }       
}

__global__ void SimpleScale(double *a, int SIZE, double s){
    // compute idx and idy, the location of the element in the original LX*LY array 
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    
    if ( idx < SIZE)
        a[idx] = a[idx] * s;
}

__global__ void cpx_prod_GPU (CC *a, CC *b, CC *c, int SIZE){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < SIZE){
        c[idx].x = a[idx].x * b[idx].x - a[idx].y * b[idx].y ;
        c[idx].y = a[idx].x * b[idx].y + a[idx].y * b[idx].x ;
    }
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

__global__ void KFR( CC *a, CC *b, CC *c, double FR, int SIZE ){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < SIZE){
        c[idx].x = a[idx].x * (1-FR) + b[idx].x * FR;
        c[idx].y = a[idx].y * (1-FR) + b[idx].y * FR;
    }
}

void COMPUTE_TFN( CC *g, CC *d_u, CC *d_u_W, CC *d_hR_W, CC *d_SST, int SIZE, double FR, int nBytes, CC *d_op1, CC *d_op1_W, CC *d_op2, CC *d_op2_W, CC *d_op3, CC *d_op3_W, CC *d_op4_W){
    // parameters for kernels
    int dimx = 1 << 7;
    dim3 block(dimx);
    dim3 grid((SIZE + block.x - 1) / block.x);
    
    cufftHandle plan;
    cufftPlan1d(&plan, SIZE, CUFFT_Z2Z, 1);
    
    modulus<<<grid,block>>>( d_u, d_op1, SIZE ); // d_op1 = |d_u|^2.
    CHECK(cudaDeviceSynchronize());  

    //CHECK(cudaGetLastError());   
    cufftExecZ2Z(plan, (CC *)d_op1, (CC *)d_op1_W, CUFFT_INVERSE); // d_op1_W.
    CHECK(cudaDeviceSynchronize());  
    CUFFTscale<<<grid,block>>>(d_op1_W, SIZE, SIZE);
    CHECK(cudaDeviceSynchronize());  
/*    if (kk == 2){
        CC *pipa = (CC *)malloc(nBytes);   
        CHECK(cudaMemcpy(pipa, d_op1, nBytes, cudaMemcpyDeviceToHost));
        FILE *au;	
        au = fopen("prueba.txt", "w+");
        for ( int i = 0; i < SIZE; i++ )
            fprintf(au, "%15.20f\t%15.20f\n",pipa[i].x, pipa[i].y);// writing data into file
        fclose(au);//closing file
        free(pipa);
    }   */       
    cpx_prod_GPU<<<grid,block>>>(d_op1_W, d_hR_W, d_op2_W ,SIZE); // d_op2_W = d_op1_W * d_hR_W.
    CHECK(cudaDeviceSynchronize());  
    //CHECK(cudaGetLastError());   

    cufftExecZ2Z(plan, (CC *)d_op2_W, (CC *)d_op2, CUFFT_FORWARD); // d_op2.
    CHECK(cudaDeviceSynchronize());  
    //CHECK(cudaGetLastError());   
 
    CC *d_aux2;  
    CHECK(cudaMalloc((void **)&d_aux2, nBytes));
    //CHECK(cudaMemset(d_aux2, 0, nBytes));
    KFR<<<grid,block>>>(d_op1, d_op2, d_aux2, FR, SIZE); // d_aux2 = (1-fR) * d_op1 + fR * d_op2
    CHECK(cudaDeviceSynchronize());  
    //CHECK(cudaGetLastError()); 
    
    cpx_prod_GPU<<<grid,block>>>(d_u , d_aux2, d_op3, SIZE); // d_op3 = d_aux2 * d_u.
    CHECK(cudaDeviceSynchronize());  
    //CHECK(cudaGetLastError());   
    CHECK(cudaFree(d_aux2));

    cufftExecZ2Z(plan, (CC *)d_op3, (CC *)d_op3_W, CUFFT_INVERSE); // d_op3_W.
    CHECK(cudaDeviceSynchronize());  
    CUFFTscale<<<grid,block>>>(d_op3_W, SIZE, SIZE);
    CHECK(cudaDeviceSynchronize());  
    
    cpx_prod_GPU<<<grid,block>>>(d_SST, d_op3_W, d_op4_W, SIZE); // d_op4_W = d_SST * d_op3.
    CHECK(cudaDeviceSynchronize());  
    equal<<<grid,block>>>(g, d_op4_W, SIZE);
    CHECK(cudaDeviceSynchronize());  

    //Destroy CUFFT context
    cufftDestroy(plan);
}
