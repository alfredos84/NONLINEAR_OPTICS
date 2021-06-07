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

double pot(double a, int n);
void normalize_N(CC *in_t, int SIZE);
__global__ void reduce(double* vector);
__global__ void scale(CC *a, int SIZE, double s);
__global__ void modulus1(CC *z, double *w, int SIZE);
void noise_generator (CC *h_noise, int N);
void input_field_T(CC *Ap, double *T, int SIZE, double T0, double POWER, char m);
void inic_vector_T(double *T, int SIZE, double T_WIDTH, double dT);
void inic_vector_F(double *F, int SIZE, double DF);
double inic_vector_Z(double *Z, double SIZE, double STEP);
double v_max(double vp, double vi);
int factorial( int number );
void cpx_sum (CC *a, CC *b, CC *c, int SIZE);
void cpx_prod (CC *a, CC *b, CC *c, int SIZE);
void cpx_mod (CC *a, CC *b, int SIZE);
void fftshift( double *V_ss, double *v, int SIZE );
__global__ void CUFFTscale(CC *a, int SIZE, int s);
__global__ void cpx_sum_GPU (CC *a, CC *b, CC *c, int SIZE);
__global__ void equal(CC *Apw, CC *auxp, int SIZE);
__global__ void cpx_prod_GPU (CC *a, CC *b, CC *c, int SIZE);
__global__ void dAdz( CC *Ap, CC *As, CC *dAp, CC *dAs, double kp, double ks, double dk, double z, int SIZE );
__global__ void lineal(CC *Ap, CC *As, CC *kp, CC *ks, CC *auxp, CC *auxs, int SIZE, double s);
__global__ void rk4(CC *Ap, CC *As,CC *k1p, CC *k1s, CC *k2p, CC *k2s,CC *k3p, CC *k3s,CC *k4p, CC *k4s, double dz, int SIZE);
__global__ void linear_operator(CC *auxp, CC *auxs, CC *Apw, CC* Asw, double *w, double vm, double vp, double vs, double b2p, double b2s, int SIZE, double z);
__global__ void add_phase(CC *As, CC *aux,float R, double delta, int nn, int SIZE);
__global__ void add_GDD(CC *As, CC *aux, double *w, double GDD, int SIZE);
__global__ void read_pump(CC *Ap_total, CC *Ap, int N_rt, int nn, int N_ps, int extra_win);
void evol_in_crystal(double *w_ext_gpu, dim3 grid, dim3 block, CC *Ap_gpu, CC *As_gpu, CC *Apw_gpu, CC *Asw_gpu, CC *k1p_gpu, CC *k1s_gpu, CC *k2p_gpu, CC *k2s_gpu, CC *k3p_gpu, CC *k3s_gpu, CC *k4p_gpu, CC *k4s_gpu, CC *auxp_gpu, CC *auxs_gpu, double vm, double vp, double vs, double b2p, double b2s, double kp, double ks, double dk, double dz, int steps_z, int SIZE, int nBytes, cufftHandle plan);
__global__ void write_field(CC *As_total, CC *As, int nn, int N_ps, int extra_win, int N_rt);
