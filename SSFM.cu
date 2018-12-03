
// compile with "nvcc SSFM.cu functions.cu -lcufftw -lcufft -lcurand -o SSFMcu"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <string.h>
#include <curand.h>
#include "common.h"

// Complex data type
typedef cufftDoubleComplex CC;

#define PI2 2.0 * 3.14159265358979323846  //2*pi
#define C 299792458*1E9/1E12  // speed of ligth in vacuum [nm/ps]
#define NELEMS(x)  (sizeof(x) / sizeof((x).x)) // number of elements of an array

#include "functions.h"

int main(int argc, char *argv[]){
	int i; 
	int N = 1<<16; // number of points
	
	// parameters for kernels
	int dimx = 1 << 5;
	dim3 block(dimx);
	dim3 grid((N + block.x - 1) / block.x);
	
	// set up device	
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("Using Device %d: %s\n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));
	
	int nBytes =  sizeof(CC)*N;
	double t_width = (double )(N * 0.003); // time window size
	double T0 = 0.1; //temporal width of pulses [ps]
	double lambda_0 = 5000; // central wavelength [nm]
        double w0 = PI2 * C / lambda_0; // angular frequency in 2*pi*[THz]
	double betas[3] = {-0.05, 0.004,0}; // betas [ps^i / km]
	int lb = 3; // number of betas that are included
	double dT, dF; // step time and frequency
	double sol_ord, P0, factor; // soliton order, power and factor for anomalous dispersion
	
	double gamma = 0.1; // nonlinear parameter gamma [1/W/m]
	float tau1 = 0.0155, tau2 = 0.2305; // Raman times [ps] 
	float fr = 0.1; // fractional Raman contribution
	
        char kindpower = 'p'; // select among different kind of power
	switch(kindpower) {
        case 'n': // select soliton order and then associated power will be computed
            sol_ord = 1; // Soliton order
            P0 = pow(sol_ord,2) * fabs(betas[0])/(gamma*pow(T0,2));
            break;
        case 'p': // select power and then soliton order will be computed
            factor = 0.7; // normalized power from cutoff
            P0 = (fabs(betas[0])* w0 * w0/gamma)*factor; // peak power of input [W]
            sol_ord = sqrt(P0*gamma*pow(T0,2)) / fabs(betas[0]);
            break;
        case 'a': // arbitrary power
            double P0 = 50.00; // peak power of input [W]
            break;
    }
	
		
    /* Distances */
	double LD = pow(T0,2) / fabs(betas[0]);  // dispersion lenght
	// double LD3 = pow(T0,3) / fabs(betas[2]); // third order dispersion length
    double LNL = 1/gamma/P0; // nonlinear length
    double Zfiss = LD/sol_ord; // soliton fission length
    double Zsol = 0.5 * 3.14159265358979323846 * LD; // soliton period
	double flength = 0.0012;
	double h = flength/4000; // z step
	int steps_z = (int )floor(flength/h); // number of steps in Z

	/* Set plan for cuFFT */
	cufftHandle plan_1;
	cufftPlan1d(&plan_1, N, CUFFT_Z2Z, 1);
	
	CC *u1 = (CC*)malloc(nBytes);	CC *u1_W = (CC*)malloc(nBytes);
	CC *u2 = (CC*)malloc(nBytes);	CC *u2_W = (CC*)malloc(nBytes);
	CC *u3 = (CC*)malloc(nBytes);	CC *u3_W = (CC*)malloc(nBytes);
	CC *u4 = (CC*)malloc(nBytes);	CC *u4_W = (CC*)malloc(nBytes);
	CC *u_ip = (CC*)malloc(nBytes);	CC *D_OP = (CC*)malloc(nBytes);     // Linear operator exp(Dh/2)
	CC *hR = (CC*)malloc(nBytes);  // Raman response in time domain
	CC *hR_W = (CC*)malloc(nBytes);     // Raman response in frequency domain
	CC *self_st = (CC*)malloc(nBytes);  // Self-steepening
	CC *alpha1 = (CC*)malloc(nBytes);  CC *alpha2 = (CC*)malloc(nBytes);
	CC *alpha3 = (CC*)malloc(nBytes);  CC *alpha4 = (CC*)malloc(nBytes);
	CC *V_ss = (CC*)malloc(nBytes);
	/*************************/    

	/* Time, frequency and Z vectors*/
	double *T;    
	T = (double*) malloc(sizeof(double) * N);
	dT = inic_vector_T(T, N, t_width);

	double *TT;    
	TT = (double*) malloc(sizeof(double) * N);
	inic_vector_Traman(TT, N, t_width);
	
	double *V;    
	V = (double*) malloc(sizeof(double) * N);
	dF = inic_vector_F(V, N, dT);
	
	freq_shift( V_ss, V, N ); //frequecy used in DOP and self-steepening

	double *Z;
	Z = (double*) malloc(sizeof(double) * steps_z);
	inic_vector_Z(Z, steps_z, h);
	
	/*************************/
	/* Set Raman */
	
	/* Raman */
	RAMAN_RESP(hR, N, tau1, tau2, TT);
	CC *d_hR, *d_hR_W;
	CHECK(cudaMalloc((void **)&d_hR, nBytes));
	CHECK(cudaMalloc((void **)&d_hR_W, nBytes));
	CHECK(cudaMemset(d_hR, 0, nBytes));
	CHECK(cudaMemset(d_hR_W, 0, nBytes));
	// Copy host memory to device
	CHECK(cudaMemcpy(d_hR, hR, nBytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_hR_W, hR_W, nBytes, cudaMemcpyHostToDevice));
	cufftExecZ2Z(plan_1, (CC *)d_hR, (CC *)d_hR_W, CUFFT_INVERSE);
	cudaDeviceSynchronize();
	//Scale Raman W
	scale<<<grid,block>>>(d_hR_W, N, dT);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaGetLastError());    
	CHECK(cudaMemcpy(hR_W, d_hR_W, nBytes, cudaMemcpyDeviceToHost));
	
	/*************************/
	
	/* Self steepening */
	inic_selfst(self_st, V_ss, w0, gamma, N);
	/*******************/

	printf("N = %i points\nbeta2 = %f ps^2/km\ndT = %f ps\ndF = %f THz\nfr = %f\ntau1 = %f ps\ntau2 = %f ps\nw0 = %.2f THz\nlambda0 = %.1f nm\nstep size = %f m\nDistance = %f m\nPower = %.2f W\np = %0.2f\n",N, betas[0], dT, dF, fr, tau1, tau2, w0, lambda_0, h,flength, P0, factor);
	
	/* Input field and envelope expressed in the interaction picture */    
	linear_operator(D_OP, V_ss, betas, lb, N, h); //set exp(D*h/2) as a function of omega = 2*pi*f
	unsigned int NOISE = 1;
	if (NOISE == 0){
		input_field_T(u1, T, N, T0, P0); // signal without noise
	}
	else{
		input_field_T(u1, T, N, T0, P0);
		CC *h_noise = (CC *)malloc(nBytes);
		double SNR = 30; // Signal-to-Noise ratio
		noise_generator(h_noise, SNR, N, P0 );
		for (int j = 0; j < N; j++){
			u1[j].x = u1[j].x + h_noise[j].x;
			u1[j].y = u1[j].y + h_noise[j].y;
		}
		free(h_noise);
	}
	
	/* Device vectors */
	CC *d_u_ip, *d_alpha1, *d_alpha2, *d_alpha3, *d_alpha4, *d_u1_W, *d_u1, *d_u2_W, *d_u2, *d_u3_W, *d_u3, *d_u4_W, *d_u4;
	CHECK(cudaMalloc((void **)&d_u1_W, nBytes)); CHECK(cudaMemset(d_u1_W, 0, nBytes));
	CHECK(cudaMalloc((void **)&d_u1, nBytes)); CHECK(cudaMemset(d_u1, 0, nBytes));
	CHECK(cudaMemcpy(d_u1, u1, nBytes, cudaMemcpyHostToDevice));    
    
	/* computes FFT for input field */
	cufftExecZ2Z(plan_1, (CC *)d_u1, (CC *)d_u1_W, CUFFT_INVERSE);
	CHECK(cudaDeviceSynchronize());
	CUFFTscale<<<grid,block>>>(d_u1_W, N, N);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaGetLastError());
	CHECK(cudaMemcpy(u1_W, d_u1_W, nBytes, cudaMemcpyDeviceToHost));
	
	
	/* Saving some vectors */
	FILE *uno;	
	uno = fopen("T.txt", "w+");
	for ( int i = 0; i < N; i++ )
		fprintf(uno, "%15.10f\t", T[i]);// writing data into file
	fclose(uno);//closing file

	FILE *dos;
	dos = fopen("V.txt", "w+");
	for ( int i = 0; i < N; i++ )
		fprintf(dos, "%15.10f\t", V[i]);// writing data into file
	fclose(dos);//closing file
	
	
	/* Allocating memory on GPU */
	CHECK(cudaMalloc((void **)&d_u_ip, nBytes));
	CHECK(cudaMalloc((void **)&d_alpha1, nBytes));
	CHECK(cudaMalloc((void **)&d_alpha2, nBytes));
	CHECK(cudaMalloc((void **)&d_alpha3, nBytes));
	CHECK(cudaMalloc((void **)&d_alpha4, nBytes));
	CHECK(cudaMalloc((void **)&d_u2_W, nBytes));
	CHECK(cudaMalloc((void **)&d_u2, nBytes)); 
	CHECK(cudaMalloc((void **)&d_u3_W, nBytes));
	CHECK(cudaMalloc((void **)&d_u3, nBytes)); 
	CHECK(cudaMalloc((void **)&d_u4_W, nBytes));
	CHECK(cudaMalloc((void **)&d_u4, nBytes)); 
	/*************************/	
	
	printf("Starting main loop on CPU & GPU...\n");
	double iStart = seconds();
	
    /* START MAIN LOOP */   
	for (int s = 1; s < steps_z; s++){
		cpx_prod (D_OP, u1_W, u_ip, N); // A_I(w,z) = exp(D*h/2)*A(w,z)
		
		COMPUTE_TFN( alpha1, u1, u1_W, hR_W, V, self_st, N, fr, nBytes );
		
		CC *aux1 =  (CC*)malloc(nBytes);
		cpx_prod (D_OP, alpha1, aux1, N);
		for (i = 0; i<N; i++){
			alpha1[i].x = aux1[i].x;
			alpha1[i].y = aux1[i].y;        
		}
		free(aux1);
			
		CHECK(cudaMemcpy(d_u_ip, u_ip, nBytes, cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_alpha1, alpha1, nBytes, cudaMemcpyHostToDevice));
		lineal<<<grid,block>>>(d_u_ip, d_alpha1, d_u2_W, N, h/2);

		CHECK(cudaMemcpy(u2_W, d_u2_W, nBytes, cudaMemcpyDeviceToHost));
		CHECK(cudaDeviceSynchronize());
		CHECK(cudaGetLastError());
		cufftExecZ2Z(plan_1, (CC *)d_u2_W, (CC *)d_u2, CUFFT_FORWARD);
		cudaDeviceSynchronize();
		CHECK(cudaMemcpy(u2, d_u2, nBytes, cudaMemcpyDeviceToHost));
		COMPUTE_TFN( alpha2, u2, u2_W, hR_W, V, self_st, N, fr, nBytes );
		CHECK(cudaMemcpy(d_alpha2, alpha2, nBytes, cudaMemcpyHostToDevice));
		lineal<<<grid,block>>>(d_u_ip, d_alpha2, d_u3_W, N, h/2);  

		CHECK(cudaMemcpy(u3_W, d_u3_W, nBytes, cudaMemcpyDeviceToHost));
		CHECK(cudaDeviceSynchronize());
		CHECK(cudaGetLastError());     

		cufftExecZ2Z(plan_1, (CC *)d_u3_W, (CC *)d_u3, CUFFT_FORWARD);
		CHECK(cudaDeviceSynchronize());
		CHECK(cudaGetLastError()); 
		CHECK(cudaMemcpy(u3, d_u3, nBytes, cudaMemcpyDeviceToHost));

		COMPUTE_TFN( alpha3, u3, u3_W, hR_W, V, self_st, N, fr, nBytes );
		CHECK(cudaMemcpy(d_alpha3, alpha3, nBytes, cudaMemcpyHostToDevice));        
		CC *aux2 =  (CC*)malloc(nBytes);
		CC *d_aux2; CHECK(cudaMalloc((void **)&d_aux2, nBytes));  CHECK(cudaMemset(d_aux2, 0, nBytes));
		lineal<<<grid,block>>>(d_u_ip, d_alpha3, d_aux2, N, h); 

		CHECK(cudaDeviceSynchronize());
		CHECK(cudaGetLastError());    
		CHECK(cudaMemcpy(aux2, d_aux2, nBytes, cudaMemcpyDeviceToHost));       
		cpx_prod (D_OP, aux2, u4_W, N);
		free(aux2); CHECK(cudaFree(d_aux2));
		
		CHECK(cudaMemcpy(d_u4_W, u4_W, nBytes, cudaMemcpyHostToDevice));
		cufftExecZ2Z(plan_1, (CC *)d_u4_W, (CC *)d_u4, CUFFT_FORWARD);
		cudaDeviceSynchronize();
		CHECK(cudaMemcpy(u4, d_u4, nBytes, cudaMemcpyDeviceToHost));

		COMPUTE_TFN( alpha4, u4, u4_W, hR_W, V, self_st, N, fr, nBytes );

		CC *aux3 =  (CC*)malloc(nBytes);
		CC *aux4 =  (CC*)malloc(nBytes);
		
		CC *d_aux3; CHECK(cudaMalloc((void **)&d_aux3, nBytes)); CHECK(cudaMemset(d_aux3, 0, nBytes));
		CC *d_aux4; CHECK(cudaMalloc((void **)&d_aux4, nBytes)); CHECK(cudaMemset(d_aux4, 0, nBytes));

		final<<<grid,block>>>(d_u_ip, d_alpha1, d_alpha2, d_alpha3, d_aux3, h, N); 

		CHECK(cudaMemcpy(aux3, d_aux3, nBytes, cudaMemcpyDeviceToHost));
		cpx_prod (D_OP, aux3, aux4, N);

		free(aux3); CHECK(cudaFree(d_aux3));
		
		CHECK(cudaMemcpy(d_aux4, aux4, nBytes, cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_alpha4, alpha4, nBytes, cudaMemcpyHostToDevice));
		CHECK(cudaMemset(d_u1_W, 0, nBytes));
		lineal<<<grid,block>>>(d_aux4, d_alpha4, d_u1_W, N, h/6);

		free(aux4); CHECK(cudaFree(d_aux4));
		
		CHECK(cudaMemcpy(u1_W, d_u1_W, nBytes, cudaMemcpyDeviceToHost));

		cufftExecZ2Z(plan_1, (CC *)d_u1_W, (CC *)d_u1, CUFFT_FORWARD);
		CHECK(cudaDeviceSynchronize());
		CHECK(cudaMemcpy(u1, d_u1, nBytes, cudaMemcpyDeviceToHost));
		
		//printf("%.2f %% completed...\n", (double) s*100/(steps_z-1));   
		
		CHECK(cudaMemset(d_alpha1, 0, nBytes));
		CHECK(cudaMemset(d_alpha2, 0, nBytes));
		CHECK(cudaMemset(d_alpha3, 0, nBytes));
		CHECK(cudaMemset(d_alpha4, 0, nBytes));
		CHECK(cudaMemset(d_u1_W, 0, nBytes));	
		CHECK(cudaMemset(d_u1, 0, nBytes));/////////
		CHECK(cudaMemset(d_u2_W, 0, nBytes));
		CHECK(cudaMemset(d_u2, 0, nBytes));/////////
		CHECK(cudaMemset(d_u3_W, 0, nBytes));
		CHECK(cudaMemset(d_u3, 0, nBytes));/////////
		CHECK(cudaMemset(d_u4_W, 0, nBytes));
		CHECK(cudaMemset(d_u4, 0, nBytes));/////////
		}
		FILE *tres, *cuatro;
		tres = fopen("output_T.txt", "w+");
		cuatro = fopen("output_W.txt", "w+");
		for ( i = 0; i < N; i++ ){
			fprintf(tres, "%15.10f\t%15.10f\n", u1[i].x, u1[i].y);// writing data into file
		fprintf(cuatro, "%15.10f\t%15.10f\n", u1_W[i].x, u1_W[i].y);// writing data into file
	}
	fclose(tres);//closing file	
	fclose(cuatro);//closing file
	

	double iElaps = seconds() - iStart;
	printf("...time elapsed %f sec\n", iElaps);
	
	/* Deallocating memory and destroying plans */
	free(u1); free(u1_W); free(u2); free(u2_W);
	free(u3); free(u3_W); free(u4); free(u4_W);
	free(alpha1); free(alpha2); free(alpha3); free(alpha4);
	free(u_ip); free(D_OP); free(self_st); free(hR);
	free(hR_W); free(V_ss); free(T); free(TT);
	free(V); free(Z);
	CHECK(cudaFree(d_u_ip)); CHECK(cudaFree(d_alpha1));
	CHECK(cudaFree(d_alpha2)); CHECK(cudaFree(d_alpha3));
	CHECK(cudaFree(d_u1)); CHECK(cudaFree(d_u2));
	CHECK(cudaFree(d_u3)); CHECK(cudaFree(d_u4));
	CHECK(cudaFree(d_u1_W)); CHECK(cudaFree(d_u2_W));
	CHECK(cudaFree(d_u3_W)); CHECK(cudaFree(d_u4_W));
	//Destroy CUFFT context
	cufftDestroy(plan_1);
	
	cudaDeviceReset();

	return 0;
}