// Compile with "nvcc OPO_base.cu functions.cu --gpu-architecture=sm_60 -lcufftw -lcufft -lcurand -o OPO" for GeForce MX250 (Pascal)
//-gencode=arch=compute_60,code=sm_60

/*#endif     
#if defined(NOISE)         
#endif */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <string.h>
#include <curand.h>
#include "common.h"
#include "functions.h"

// Complex data type //
typedef cufftDoubleComplex CC;

int main(int argc, char *argv[]){
	printf("\n\n\n#######---Welcome to OPO calculator---#######\n\n\n");
	
	// timing the code
	double iStart = seconds();

	const double PI   = 3.14159265358979323846;          //pi
	const double C    = 299792458*1E6/1E12;              // speed of ligth in vacuum [um/ps]
	const double EPS0 = 8.8541878128E-12*1E12/1E6;        // vacuum pertivity [W.ps/V²μm] 

	//int n_realiz  = atoi(argv[1]);
	char filename1[60],filename2[60],filename3[60];
	//size_t n_fact = sizeof(factores)/sizeof(factores[0]);

	// Set up device //
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("\n\nUsing Device %d: GPU %s\n\n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));

	printf("Setting constants and vectors in host... \n");
	
	// Set parameters and constants
	int N_rt        = atoi(argv[3]); // number of round trips to cover the input pulse
	
	// Define wavelengths
	double lp          = 1.064;         // pump wavelenght [μm]
	double ls          = 2*lp;          // signal wavelenght [μm]
	double li          = 1/(1/lp-1/ls); 
	double fp          = C/lp;          // pump frequency [THz]
	double wp          = 2*PI*C/lp;     
	double fs          = C/ls;          // signal frequency [THz]
	double ws          = 2*PI*C/ls;     

	// refractive index, group_vel and GVD
	// Nonlinear crystals: 1 "MgO-PPLN", 2 - "sPPLT", 3 - "GaP"
	
	double np, ns, ni, vp, vs, vi, vm, b2p, b2s, deff, Lcr;
	
	printf("\nNonlinear crystal PPLN");
	//ORIGINAL np = 1.636908;	ns = 1.636851; ni = 1.636851;
	np = 2.147638;	ns = 2.114934; ni = 2.114934;
	vp = C/2.20179;	vs = C/2.17109;	vi = C/2.17109;
	//ORIGINAL vp = C/1.656;	vs = C/1.676;	vi = C/1.676;
	vm = v_max(vp,vi);
	// b2p = 4.05e-8;	b2s = -1.715e-7;
	b2p = 2.32243e-7;	b2s = -6.9427e-8;
	deff = 14.77e-6; // effective d [pm/V]
	printf("\ndeff = %.2f pV/m", deff*1e+6);
	Lcr = 5e3;  // crystal lenght [um]
	printf("\nCrystal length = %.2f mm", Lcr*1e-3);
	
	double divs = atoi(argv[2]);
	double dz = Lcr/divs; // number of z step inside the crystal
	int steps_z = (int )floor(Lcr/dz); // number of steps in Z
	double *Z;
	Z = (double*) malloc(sizeof(double) * steps_z);
	inic_vector_Z(Z, steps_z, dz);
	
	double kp = 2*PI*deff/(np*lp); // kappa pump [1/V]
	double ks = 2*PI*deff/(ns*ls); // kappa signal [1/V]
	double ki = 2*PI*deff/(ni*li); // kappa idler [1/V]
	double dk = 0; // mismatch

	// Cavity
	double Lcav = atof(argv[4]) * Lcr;  // cavity lenght [um]
	printf("\nCavity length = %1.3f mm\n", Lcav*1e-3);
	float R = 0.99;  // mirror reflectivity 
	double t_rt = (Lcav-Lcr)/C + Lcr/vm;
	double GDD = 0;// -b2s*Lcr; // GDD [ps²]

	printf("\n\n\ndelta critical = %.4f\n\n\n",ns*Lcr/(t_rt*Lcav));
	
	// Time and frequency discretization
	double T_width     = (double ) (N_rt*t_rt); // total time for input ns-pulse
	int ex             = atoi(argv[1]);
	int N_ps           = 1<<ex;  // points per time slice
	int extra_win      = 32; // extra pts for short-time slices
	printf("\nUsing N=2^%d=  %d points\n\n", ex, N_ps);
	double dT          = t_rt/N_ps; // time step in [ps]
	double dF          = 1/t_rt; // frequency step in [THz]
	const int SIZE           = N_ps+extra_win;
	const long int SIZEL     = N_ps*N_rt;
	
	/* vector T for one round trip */
	double *T;
	T = (double*) malloc(sizeof(double) * SIZE);
	inic_vector_T(T, SIZE, t_rt, dT); // temp. grid for slice
	FILE *trt;
	trt = fopen("T.dat", "w+");
	for ( int m = 0; m < SIZE; m++ )
		fprintf(trt, "%15.10f\n", T[m]);// writing data into file
	fclose(trt);//closing file 
	
	/* vectors Tp for the complete time */
	double *Tp;
	Tp = (double*) malloc(sizeof(double) * SIZEL);
	inic_vector_T(Tp, SIZEL, T_width, dT); // temp. grid for pump field
	printf("\nSaving total time...");
	
	FILE *tfp;
	tfp = fopen("Tp.dat", "w+");
	for ( long int m = 0; m < SIZEL; m++ ){
		fprintf(tfp, "%15.25f\n", Tp[m]);// writing data into file
	}
	fclose(tfp); //closing files
	printf("   OK!\n\n");
	
	/* vector F_p for the complete pump frequency */
	printf("Defining F_p...");
	double dF_p        = 1/T_width;
	double *F_p;    
	F_p = (double*) malloc(sizeof(double) * SIZEL);
	inic_vector_F(F_p, SIZEL, dF_p);
	printf("max(Fp) = %f THz\n\n", F_p[SIZEL-1]);
	
	FILE *ffp;
	ffp = fopen("freq.dat", "w+");
	for ( long int m = 0; m < SIZEL; m++ )
		fprintf(ffp, "%15.25f\n", F_p[m]);// writing data into file
	fclose(ffp);  //closing files
	
	/* vector w_p for the complete pump angular frequency */
	double *w_p;    
	w_p = (double*) malloc(sizeof(double) * SIZEL);
	for (int i=0; i<SIZEL; i++){
		w_p[i] = 2*PI*F_p[i];  // ang freq [2*pi*THz]
	}

	double *F_ext = (double*) malloc(sizeof(double) * SIZE);
	inic_vector_F(F_ext, SIZE, dF); // extended freq. grid [THz]
	
	double *w_ext = (double*) malloc(sizeof(double) * SIZE);
	fftshift(w_ext,F_ext, SIZE);
	for (int i=0; i<SIZE; i++){
		w_ext[i] = 2*PI*w_ext[i];  // ang freq [2*pi*THz]
	}
	// Host vectors //
	int nBytes =  sizeof(CC)*SIZE;
	CC *As_total   = (CC*)malloc(sizeof(CC) * SIZEL);
	
	printf("Defining input field...");
	/* Define input pump */
	CC *Ap_total       = (CC*)malloc(sizeof(CC) * SIZEL);
	CC *Ap_total_out   = (CC*)malloc(sizeof(CC) * SIZEL);
	double FWHM        = 2000; // input pump with fwhm [ps]
	double sigmap      = FWHM/2/sqrt(2*log(2)); // standar deviation for gaussian pulse [ps]
	double waist       = 55; // beam waist radius [um]
	double spot        = PI*waist*waist; // spot area [μm²]
	printf("\n\nspot = %.0f μm²\n", spot);
	double P_th        = spot*np*EPS0*C*pot( ls*ns*(1-R)/PI/deff/Lcr, 2 )/R/8;
	printf("Power threshold = %.2f W\n", P_th);
	double Power       = 2; //8*P_th;
	printf("Power = %.2f W\n", Power);
	double Intensity   = Power/spot; // intensity [W/μm²]
	printf("I = %.2f W/um²\n", Intensity);
	double Ap0         = sqrt(Intensity*2/C/np/EPS0); // input field amplitud [V/μm] 
	printf("Ap0 = %.2f V/um\n", Ap0);
	char m = 'c'; // Wave form: 'c' = CW, 'g' = gaussian pulse, 's' = soliton//
	input_field_T(Ap_total, Tp, SIZEL, sigmap, Ap0*Ap0, m); // input pump
	

	CC *Ap         = (CC*)malloc(nBytes); // time slice pump
	CC *As         = (CC*)malloc(nBytes);
	noise_generator(As, SIZE, Ap0*Ap0 ); // inicial noisy signal
	for ( int i = 0; i < SIZE; i++ ){
		As[i].x = As[i].x*1E-4;
		As[i].y = As[i].y*1E-4;
	}
	FILE *red;
	red = fopen("noise.dat", "w+");
	for ( int l = 0; l < SIZE; l++ )
		fprintf(red, "%15.20f\t%15.20f\n", As[l].x, As[l].y);// writing data into file
	fclose(red);//closing file  
	
	// Device vectors	
	// Parameters for kernels
	int dimx = 1 << 7; //ceil(SIZE/1024);
	dim3 block(dimx);
	int N = SIZE;
	dim3 grid((N + block.x - 1) / block.x);
	printf("\nKernels dimensions:\n<<<grid_dim, block_dim>>> = <<<(%d,%d), (%d,%d)>>>\n", grid.x, grid.y, block.x, block.y);
	
	printf("Setting constants and vectors in device...\n\n");
	double *w_ext_gpu;
	CHECK(cudaMalloc((void **)&w_ext_gpu, sizeof(double) * SIZE ));
	
	CC *As_gpu, *Ap_gpu, *As_total_gpu, *Ap_total_out_gpu, *Ap_total_gpu, *Asw_gpu, *Apw_gpu;
	
	CHECK(cudaMalloc((void **)&As_gpu, nBytes ));
	CHECK(cudaMalloc((void **)&Ap_gpu, nBytes ));
	CHECK(cudaMemset(Ap_gpu, 0, nBytes));
	CHECK(cudaMalloc((void **)&As_total_gpu, sizeof(CC) * SIZEL ));
	CHECK(cudaMemset(As_total_gpu, 0, sizeof(CC) * SIZEL));
	CHECK(cudaMalloc((void **)&Ap_total_out_gpu, sizeof(CC) * SIZEL ));
	CHECK(cudaMemset(Ap_total_out_gpu, 0, sizeof(CC) * SIZEL));
	CHECK(cudaMalloc((void **)&Ap_total_gpu, sizeof(CC) * SIZEL ));
	CHECK(cudaMalloc((void **)&Asw_gpu, nBytes ));
	CHECK(cudaMalloc((void **)&Apw_gpu, nBytes ));
	
	CHECK(cudaMemcpy(As_gpu, As, nBytes, cudaMemcpyHostToDevice));	
	CHECK(cudaMemcpy(Ap_gpu, Ap, nBytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(w_ext_gpu, w_ext, sizeof(double) * SIZE , cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(Ap_total_gpu, Ap_total, sizeof(CC) * SIZEL, cudaMemcpyHostToDevice));
	//CHECK(cudaGetLastError());
	
	// Aux device vectors //
	CC *k1p_gpu, *k2p_gpu, *k3p_gpu, *k4p_gpu, *k1s_gpu, *k2s_gpu, *k3s_gpu, *k4s_gpu;
	CHECK(cudaMalloc((void **)&k1p_gpu, nBytes ));
	CHECK(cudaMalloc((void **)&k2p_gpu, nBytes ));
	CHECK(cudaMalloc((void **)&k3p_gpu, nBytes ));
	CHECK(cudaMalloc((void **)&k4p_gpu, nBytes ));
	CHECK(cudaMalloc((void **)&k1s_gpu, nBytes ));
	CHECK(cudaMalloc((void **)&k2s_gpu, nBytes ));
	CHECK(cudaMalloc((void **)&k3s_gpu, nBytes ));
	CHECK(cudaMalloc((void **)&k4s_gpu, nBytes ));
	CHECK(cudaMemset(k1p_gpu, 0, nBytes));
	CHECK(cudaMemset(k2p_gpu, 0, nBytes));
	CHECK(cudaMemset(k3p_gpu, 0, nBytes));
	CHECK(cudaMemset(k4p_gpu, 0, nBytes));
	CHECK(cudaMemset(k1s_gpu, 0, nBytes));
	CHECK(cudaMemset(k2s_gpu, 0, nBytes));
	CHECK(cudaMemset(k3s_gpu, 0, nBytes));
	CHECK(cudaMemset(k4s_gpu, 0, nBytes));
	
	CC *auxp_gpu, *auxs_gpu;
	CHECK(cudaMalloc((void **)&auxp_gpu, nBytes ));
	CHECK(cudaMalloc((void **)&auxs_gpu, nBytes ));
	CHECK(cudaMemset(auxp_gpu, 0, nBytes));
	CHECK(cudaMemset(auxs_gpu, 0, nBytes));
	
	// Set plan for cuFFT //
	cufftHandle plan;
	cufftPlan1d(&plan, SIZE, CUFFT_Z2Z, 1);
	
	double delta = atof(argv[5]);
	delta /= 10000;
	printf("delta = %1.5f\n\n", delta);
	printf("Starting main loop on CPU & GPU...\n");
	// START MAIN LOOP //
	for (int nn = 0; nn < N_rt; nn++){
		if( nn%100 == 0 || nn == N_rt-1 )
			printf("#round-trip = %d\n",nn);
		read_pump<<<grid,block>>>(Ap_total_gpu, Ap_gpu, N_rt, nn, N_ps, extra_win);
		CHECK(cudaDeviceSynchronize()); 

		evol_in_crystal(w_ext_gpu, grid, block, Ap_gpu, As_gpu, Apw_gpu, Asw_gpu, k1p_gpu, k1s_gpu, k2p_gpu, k2s_gpu,k3p_gpu, k3s_gpu,k4p_gpu, k4s_gpu, auxp_gpu, auxs_gpu,  vm, vp, vs, b2p, b2s, kp, ks, dk, dz, steps_z, SIZE, nBytes, plan);
		
		add_phase<<<grid,block>>>(As_gpu, auxs_gpu, R, delta, nn, SIZE);
		CHECK(cudaDeviceSynchronize());
		
		/*cufftExecZ2Z(plan, (CC *)As_gpu, (CC *)Asw_gpu, CUFFT_INVERSE);
		CHECK(cudaDeviceSynchronize());
		CUFFTscale<<<grid,block>>>(Asw_gpu, SIZE, SIZE);
		CHECK(cudaDeviceSynchronize());
		add_GDD<<<grid,block>>>(Asw_gpu, auxs_gpu, w_ext_gpu, GDD, SIZE);
		CHECK(cudaDeviceSynchronize());
		cufftExecZ2Z(plan, (CC *)Asw_gpu, (CC *)As_gpu, CUFFT_FORWARD);
		CHECK(cudaDeviceSynchronize());
		*/
		
		write_signal<<<grid,block>>>(As_total_gpu, As_gpu, nn, N_ps, extra_win, N_rt);
		CHECK(cudaDeviceSynchronize());
		write_signal<<<grid,block>>>(Ap_total_out_gpu, Ap_gpu, nn, N_ps, extra_win, N_rt);
		CHECK(cudaDeviceSynchronize());

		/*if(nn==N_rt-1){
			CHECK(cudaMemcpy(As, As_gpu, nBytes, cudaMemcpyDeviceToHost));
			CHECK(cudaMemcpy(Ap, Ap_gpu, nBytes, cudaMemcpyDeviceToHost));
			FILE *cinco;
			sprintf(filename5, "signal_rt_%d_delta_%1.4f.dat", nn+1,delta);
			cinco = fopen(filename5, "w+");
			for ( int l = 0; l < SIZE; l++ )
				fprintf(cinco, "%15.20f\t%15.20f\n", As[l].x, As[l].y);// writing data into file
			fclose(cinco);//closing file  

			FILE *cuatro;
			sprintf(filename4, "pump_rt_%d_delta_%1.4f.dat", nn+1,delta);
			cuatro = fopen(filename4, "w+");
			for ( int i = 0; i < SIZE; i++ )
				fprintf(cuatro, "%15.20f\t%15.20f\n", Ap[i].x, Ap[i].y);// writing data into file
			fclose(cuatro);//closing file  
		}*/
	}
	
	CHECK(cudaMemcpy(As_total, As_total_gpu, sizeof(CC) * SIZEL, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(Ap_total_out, Ap_total_out_gpu, sizeof(CC) * SIZEL, cudaMemcpyDeviceToHost));
	
	printf("\nSaving outputs pump and signal...");
	sprintf(filename1, "signal_total_delta_%1.4f.dat", delta);
	sprintf(filename2, "pump_total_delta_%1.4f.dat", delta);
	FILE *uno, *dos;
	uno = fopen(filename1, "w+");
	dos = fopen(filename2, "w+");
	for ( long int m = 0; m < SIZEL; m++ ){
		fprintf(uno, "%15.20f\t%15.20f\n", As_total[m].x, As_total[m].y);// writing data into file
		fprintf(dos, "%15.20f\t%15.20f\n", Ap_total_out[m].x, Ap_total_out[m].y);// writing data into file
	}
	fclose(uno); fclose(dos); //closing file
	printf("   OK!");
	
	cufftHandle plan1;
	cufftPlan1d(&plan1, SIZEL, CUFFT_Z2Z, 1);
	CC *Asw_total_gpu;
	CHECK(cudaMalloc((void **)&Asw_total_gpu, sizeof(CC) * SIZEL ));
	
	cufftExecZ2Z(plan1, (CC *)As_total_gpu, (CC *)Asw_total_gpu, CUFFT_FORWARD);
	CHECK(cudaDeviceSynchronize());
	//CUFFTscale<<<grid,block>>>(Asw_total_gpu, SIZEL, SIZEL);
	//CHECK(cudaDeviceSynchronize());
	
	CC *Asw_total;
	Asw_total = (CC*)malloc(sizeof(CC) * SIZEL);
	CHECK(cudaMemcpy(Asw_total, Asw_total_gpu, sizeof(CC) * SIZEL, cudaMemcpyDeviceToHost));
	
	printf("\nSaving signal spectrum...");
	sprintf(filename3, "signal_spectrum_delta_%.4f.dat", delta);
	FILE *tres;
	tres= fopen(filename3, "w+");
	for ( long int m = 0; m < SIZEL; m++ )
		fprintf(tres, "%15.35f\t%15.35f\n", Asw_total[m].x, Asw_total[m].y);// writing data into file
	fclose(tres);	//closing files
	printf("   OK!");
	
	// finish timing
	double iElaps = seconds() - iStart;
	if(iElaps>60){
		printf("\n\n...time elapsed %.3f min \n\n\n", iElaps/60);
	}
	else{
		printf("\n\n...time elapsed %.3f sec \n\n\n", iElaps);
	}

	// Deallocating memory and destroying plans //
	free(Tp);	free(T);	free(F_p);	free(w_p);
	free(w_ext);	free(F_ext);	free(Z);
	free(As);	free(Ap);	free(Ap_total); free(As_total); 
	
	CHECK(cudaFree(Asw_total_gpu)); free(Asw_total);
	CHECK(cudaFree(Ap_total_out_gpu));	free(Ap_total_out);
	CHECK(cudaFree(As_gpu));         CHECK(cudaFree(Ap_gpu));
	CHECK(cudaFree(As_total_gpu));   
	CHECK(cudaFree(Ap_total_gpu));	 CHECK(cudaFree(w_ext_gpu));
	CHECK(cudaFree(k1p_gpu));        CHECK(cudaFree(k2p_gpu));
	CHECK(cudaFree(k3p_gpu));        CHECK(cudaFree(k4p_gpu));
	CHECK(cudaFree(k1s_gpu));        CHECK(cudaFree(k2s_gpu));
	CHECK(cudaFree(k3s_gpu));        CHECK(cudaFree(k4s_gpu));	
	CHECK(cudaFree(auxs_gpu));       CHECK(cudaFree(auxp_gpu));
	
	// Destroy CUFFT context //
	cufftDestroy(plan);	cufftDestroy(plan1);          
	cudaDeviceReset();

    return 0;
}
