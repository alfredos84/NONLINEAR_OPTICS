#pragma once


void normalize_N(CC *in_t, int SIZE);
__global__ void circulargaussian(double *dx, double *dy, CC *d_noise_norm, int SIZE);
__global__ void modulus(CC *z, double *w, int SIZE);
__global__ void reduce(double* vector);
__global__ void scale(CC *a, int SIZE, double s);
void noise_generator (CC *h_noise, double SNR, int N, double POWER);
void input_field_T(CC *in_t, double *T, int SIZE, double T0, double POWER);
double inic_vector_T(double *T, int SIZE, double T_WIDTH);
double inic_vector_Traman(double *TT, int SIZE, double T_WIDTH);
double inic_vector_F(double *F, int SIZE, double DT);
double inic_vector_Z(double *Z, double SIZE, double STEP);
int factorial( int number );
void linear_operator(CC *dop, CC *FREC, double betas[], int length_betas, int SIZE, double STEP_Z);
void cpx_prod (CC *a, CC *b, CC *c, int SIZE);
void cpx_mod (CC *a, CC *b, int SIZE);
void RAMAN_RESP(CC *HR, int SIZE, float TAU1, float TAU2, double *TIME);
void inic_selfst(CC *SST, CC *FREC, double OMEGA_0, double GAMMA, int SIZE);
void freq_shift( CC *V_ss, double *v, int SIZE );
__global__ void CUFFTscale(CC *a, int SIZE, int s);
__global__ void SimpleScale(double *a, int SIZE, double s);
__global__ void final(CC *a, CC *b, CC *c,  CC *d, CC *e, double s, int SIZE);
__global__ void lineal(CC *a, CC *b, CC *c, int SIZE, double s);
__global__ void KFR( CC *a, CC *b, CC *c, float FR, int SIZE );
void COMPUTE_TFN( CC *g, CC *u, CC *u_W, CC *hR_W, double *FREC, CC *SST, int SIZE, float FR, int nBytes);