#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <string.h>
#include <curand.h>
#include "common.h"

typedef cufftDoubleComplex CC;

void normalize_N(CC *in_t, int SIZE);
__global__ void circulargaussian(double *dx, double *dy, CC *d_noise_norm, int SIZE);
__global__ void equal(CC *a, CC *b, int SIZE);
__global__ void modulus(CC *z, CC *w, int SIZE);
__global__ void modulus1(CC *z, double *w, int SIZE);
__global__ void reduce(double* vector);
__global__ void scale(CC *a, int SIZE, double s);
void noise_generator (CC *h_noise, double SNR, int N, double POWER);
void input_field_T(CC *in_t, double *T, int SIZE, double T0, double POWER, char m, int step);
void inic_vector_T(double *T, int SIZE, double T_WIDTH, double dT);
void inic_vector_Traman(double *TT, int SIZE, double T_WIDTH);
void inic_vector_F(double *F, int SIZE, double DT);
double inic_vector_Z(double *Z, double SIZE, double STEP);
int factorial( int number );
void linear_operator(CC *dop, CC *FREC, double betas[], int length_betas, int SIZE, double STEP_Z);
void cpx_prod (CC *a, CC *b, CC *c, int SIZE);
void cpx_mod (CC *a, CC *b, int SIZE);
void RAMAN_RESP(CC *HR, int SIZE, double TAU1, double TAU2, double *TIME);
void inic_selfst(CC *SST, CC *FREC, double OMEGA_0, double GAMMA, int SIZE);
void freq_shift( CC *V_ss, double *v, int SIZE );
__global__ void CUFFTscale(CC *a, int SIZE, int s);
__global__ void SimpleScale(double *a, int SIZE, double s);
__global__ void cpx_prod_GPU (CC *a, CC *b, CC *c, int SIZE);
__global__ void final(CC *a, CC *b, CC *c,  CC *d, CC *e, double s, int SIZE);
__global__ void lineal(CC *a, CC *b, CC *c, int SIZE, double s);
__global__ void KFR( CC *a, CC *b, CC *c, double FR, int SIZE );
void COMPUTE_TFN( CC *g, CC *d_u, CC *d_u_W, CC *d_hR_W, CC *d_SST, int SIZE, double FR, int nBytes, CC *d_op1, CC *d_op1_W, CC *d_op2, CC *d_op2_W, CC *d_op3, CC *d_op3_W, CC *d_op4_W);
