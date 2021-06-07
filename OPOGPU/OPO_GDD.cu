// Compile with "nvcc OPO_IPCconference.cu functions.cu --gpu-architecture=sm_60 -lcufftw -lcufft -lcurand -o OPO" for GeForce MX250 (Pascal)
//-gencode=arch=compute_60,code=sm_60

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
	double lp          = atof(argv[7])*1e-3;	// pump wavelength [μm]
	double ls          = 2*lp;	// signal wavelength [μm]
	double li          = 1/(1/lp-1/ls); 
	double fp          = C/lp;	// pump frequency [THz]
	double wp          = 2*PI*C/lp;     
	double fs          = C/ls;	// signal frequency [THz]
	double ws          = 2*PI*C/ls;     

	printf("Pump wavelength = %.0f nm \n", lp*1e3);
	printf("Signal wavelength = %.0f nm \n", ls*1e3);
	// refractive index, group_vel and GVD

	double np, ns, vp, vs, vm, b2p, b2s, deff, Lcr;
	
	/* BBO NONLINEAR CRYSTAL */
	
	if(atof(argv[7]) == 532){	
		printf("\nPumping nonlinear crystal BBO at 532 nm e>o+o");
		np = 1.656035;	ns = 1.654208;
		vp = C/1.70;	vs = C/1.674;
		vm = v_max(vp,vs);
		b2p = 1.283e-7;	b2s = 4.4e-8;
		deff = 2.01e-6; // effective d [pm/V]
		printf("\ndeff = %.2f pm/V", deff*1e6);
		Lcr = 5e3;  // crystal lenght [um]
		printf("\nCrystal length = %.2f mm", Lcr*1e-3);
	}
	else{
		printf("\nPumping nonlinear crystal BBO at 1064 nm e>o+o");
		np = 1.636908;	ns = 1.636851;
		vp = C/1.656; vs = C/1.676;
		vm = v_max(vp,vs);
		b2p = 4.05e-8; b2s = -1.715e-7;
		deff = 1.83e-6; // effective d [pm/V]
		printf("\ndeff = %.2f pm/V", deff*1e+6);
		Lcr = 5e3;  // crystal lenght [um]
		printf("\nCrystal length = %.2f mm", Lcr*1e-3);
	}
	
	double divs = atoi(argv[2]);
	double dz = Lcr/divs; // number of z step inside the crystal
	int steps_z = (int )floor(Lcr/dz); // number of steps in Z
	double *Z;
	Z = (double*) malloc(sizeof(double) * steps_z);
	inic_vector_Z(Z, steps_z, dz);
	
	double kp = 2*PI*deff/(np*lp); // kappa pump [1/V]
	double ks = 2*PI*deff/(ns*ls); // kappa signal [1/V]
// 	double ki = 2*PI*deff/(ni*li); // kappa idler [1/V]
	double dk = 0; // mismatch

	// Cavity
	double Lcav = atof(argv[4]) * Lcr;  // cavity lenght [um]
	printf("\nCavity length = %1.3f mm\n", Lcav*1e-3);
	float R = atof(argv[8])*1e-2;  // mirror reflectivity 
	printf("Mirror reflectivity = %d %%\n", atoi(argv[8]));
	double t_rt = (Lcav-Lcr)/C + Lcr/vm;
	printf("Round-trip time = %1.3f ps\n", t_rt);
    float gdd = atof(argv[10])/100;
	double GDD = -gdd*b2s*Lcr; // GDD [ps²]
	printf("GDD = %1.2f fs²\n", GDD*1e+3);
	printf("Cavity net dispersion = %1.2f fs²\n", (b2s*Lcr+GDD)*1e+3);
	double delta = atof(argv[5]);
	delta /= 100;
	printf("delta = %1.5f\n\n", delta);
	
	// Time and frequency discretization
	double T_width     = (double ) (N_rt*t_rt); // total time for input ns-pulse
	int ex             = atoi(argv[1]);
	int N_ps           = 1<<ex;  // points per time slice
	int extra_win      = 64; // extra pts for short-time slices
	printf("\nUsing N=2^%d=  %d points\n\n", ex, N_ps);
	double dT          = t_rt/N_ps; // time step in [ps]
	double dF          = 1/t_rt; // frequency step in [THz]
	int SIZE           = N_ps+extra_win;
	int SIZEL          = N_ps*N_rt;
    printf("SIZEL = %d\n", SIZEL);
	
	/* vector T for one round trip */
	double *T;
	T = (double*) malloc(sizeof(double) * SIZE);
	inic_vector_T(T, SIZE, t_rt, dT); // temp. grid for slice
	
	/* vectors Tp for the complete time */
	double *Tp;
	Tp = (double*) malloc(sizeof(double) * SIZEL);
	inic_vector_T(Tp, SIZEL, T_width, dT); // temp. grid for pump field

	/* vector F_p for the complete pump frequency */
	
	double dF_p        = 1/T_width;
	double *F_p;    
	F_p = (double*) malloc(sizeof(double) * SIZEL);
	inic_vector_F(F_p, SIZEL, dF_p);
	printf("Max frequency = %f THz\n\n", F_p[SIZEL-1]);
	
	short unsigned int save_vectors = atoi(argv[6]);
	if (save_vectors == 1){
		printf("\nSaving total time and frequency...");
		FILE *tfp;
		tfp = fopen("Tp.dat", "w+");
		for ( long int m = 0; m < SIZEL; m++ )
			fprintf(tfp, "%15.25f\n", Tp[m]);// writing data into file
		fclose(tfp); //closing files
		
		FILE *ffp;
		ffp = fopen("freq.dat", "w+");
		for ( long int m = 0; m < SIZEL; m++ )
			fprintf(ffp, "%15.25f\n", F_p[m]);// writing data into file
		fclose(ffp);  //closing files
		printf("   OK!\n\n");
		
		FILE *trt;
		trt = fopen("T.dat", "w+");
		for ( int m = 0; m < SIZE; m++ )
			fprintf(trt, "%15.10f\n", T[m]);// writing data into file
		fclose(trt);//closing file 
	}
	else{
		printf("\nTime and frequency were previuosly save...\n\n");
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
	CC *Asw_total = (CC*)malloc(sizeof(CC) * SIZEL);
	CC *Asw_av = (CC*)malloc(sizeof(CC) * SIZEL);
	memset(Asw_av , 0, SIZEL);
	
	printf("Defining input field...");
	/* Define input pump */
	CC *Ap_total       = (CC*)malloc(sizeof(CC) * SIZEL);
	CC *Ap_total_out   = (CC*)malloc(sizeof(CC) * SIZEL);
	double FWHM        = 8000; // input pump with fwhm [ps]
	printf("\nFWHM = %.2f ns", FWHM*1e-3);
	double sigmap      = FWHM/(2*sqrt(2*log(2))); // standar deviation for gaussian pulse [ps]
	printf("\nsigma = %.2f ns", sigmap*1e-3);
	double waist       = 55; // beam waist radius [um]
	double spot        = PI*waist*waist; // spot area [μm²]
	printf("\nspot = %.0f μm²\n", spot);
// 	int P = atoi(argv[9]); // interger times peak power
	double Peak_power       = atof(argv[9]);
	printf("Peak power = %.5f W\n", Peak_power);	
	double Ap0         = sqrt(2*Peak_power/(spot*np*EPS0*C)) ; // input field amplitud [V/μm] 	
	printf("Ap0 = %.2f V/um\n", Ap0);		

	char m = 'g'; // Wave form: 'c' = CW, 'g' = gaussian pulse, 's' = soliton//
	input_field_T(Ap_total, Tp, SIZEL, sigmap*sqrt(2), Ap0*Ap0, m); // input pump

// 	printf("\nSaving total pump input...");
// 	FILE *ipu;
// 	ipu = fopen("Ap_input.dat", "w+");
// 	for ( long int m = 0; m < SIZEL; m++ )
// 		fprintf(ipu, "%15.25f\t%15.25f\n", Ap_total[m].x, Ap_total[m].y);// writing data into file
// 	fclose(ipu); //closing files

	CC *Ap         = (CC*)malloc(nBytes); // time slice pump
	CC *As         = (CC*)malloc(nBytes);
	noise_generator(As, SIZE); // inicial noisy signal
	for ( int i = 0; i < SIZE; i++ ){
		As[i].x = As[i].x*1E-20;
		As[i].y = As[i].y*1E-20;
	}
	
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
	
	CC *As_gpu, *Ap_gpu, *As_total_gpu, *Ap_total_out_gpu, *Ap_total_gpu, *Asw_gpu, *Apw_gpu, *Asw_total_gpu;
	
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
	CHECK(cudaMalloc((void **)&Asw_total_gpu, sizeof(CC) * SIZEL ));
	
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
	cufftHandle plan1;
	cufftPlan1d(&plan1, SIZEL, CUFFT_Z2Z, 1);
	
	printf("Starting main loop on CPU & GPU...\n");
	// START MAIN LOOP //
	short num_realiz = 20; // number of realizations
	for (int rr = 0; rr < num_realiz; rr++){
		printf("\n######### Realization nº: %d\n\n", rr);
		for (int nn = 0; nn <= N_rt-1; nn++){
			if( nn%100 == 0 || nn == N_rt-1 )
				printf("#round-trip = %d\n",nn);
			read_pump<<<grid,block>>>(Ap_total_gpu, Ap_gpu, N_rt, nn, N_ps, extra_win);
			CHECK(cudaDeviceSynchronize()); 

			evol_in_crystal(w_ext_gpu, grid, block, Ap_gpu, As_gpu, Apw_gpu, Asw_gpu, k1p_gpu, k1s_gpu, k2p_gpu, k2s_gpu,k3p_gpu, k3s_gpu,k4p_gpu, k4s_gpu, auxp_gpu, auxs_gpu,  vm, vp, vs, b2p, b2s, kp, ks, dk, dz, steps_z, SIZE, nBytes, plan);

			add_phase<<<grid,block>>>(As_gpu, auxs_gpu, R, delta, nn, SIZE);
			CHECK(cudaDeviceSynchronize());

			if(GDD!=0){
	// 			printf("Adding GDD...\n");
				cufftExecZ2Z(plan, (CC *)As_gpu, (CC *)Asw_gpu, CUFFT_INVERSE);
				CHECK(cudaDeviceSynchronize());
				CUFFTscale<<<grid,block>>>(Asw_gpu, SIZE, SIZE);
				CHECK(cudaDeviceSynchronize());
				add_GDD<<<grid,block>>>(Asw_gpu, auxs_gpu, w_ext_gpu, GDD, SIZE);
				CHECK(cudaDeviceSynchronize());
				cufftExecZ2Z(plan, (CC *)Asw_gpu, (CC *)As_gpu, CUFFT_FORWARD);
				CHECK(cudaDeviceSynchronize());
			}
			write_field<<<grid,block>>>(As_total_gpu, As_gpu, nn, N_ps, extra_win, N_rt);
			CHECK(cudaDeviceSynchronize());
			write_field<<<grid,block>>>(Ap_total_out_gpu, Ap_gpu, nn, N_ps, extra_win, N_rt);
			CHECK(cudaDeviceSynchronize());
		}
		
		CHECK(cudaMemcpy(As_total, As_total_gpu, sizeof(CC) * SIZEL, cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy(Ap_total_out, Ap_total_out_gpu, sizeof(CC) * SIZEL, cudaMemcpyDeviceToHost));
		
// 		printf("\nSaving outputs pump and signal...");
// 		sprintf(filename1, "signal_total_delta_%1.4f_r_%d.dat", delta,rr);
// 		sprintf(filename2, "pump_total_delta_%1.4f_r_%d.dat", delta,rr);
// 		FILE *uno, *dos;
// 		uno = fopen(filename1, "w+");
// // 		dos = fopen(filename2, "w+");
// 		for ( long int m = 0; m < SIZEL; m++ ){
// 			fprintf(uno, "%15.20f\t%15.20f\n", As_total[m].x, As_total[m].y);// writing data into file
// // 			fprintf(dos, "%15.20f\t%15.20f\n", Ap_total_out[m].x, Ap_total_out[m].y);// writing data into file
// 		}
// 		fclose(uno);
// // 		fclose(dos); //closing file
// 		printf("   OK!");

		cufftExecZ2Z(plan1, (CC *)As_total_gpu, (CC *)Asw_total_gpu, CUFFT_INVERSE);
		CHECK(cudaDeviceSynchronize());
// 		CUFFTscale<<<grid,block>>>(Asw_total_gpu, SIZEL, SIZEL);
// 		CHECK(cudaDeviceSynchronize());

		CHECK(cudaMemcpy(Asw_total, Asw_total_gpu, sizeof(CC) * SIZEL, cudaMemcpyDeviceToHost));
		
		for (int i = 0; i < SIZEL; i++){
			Asw_av[i].x +=  Asw_total[i].x/SIZEL;
			Asw_av[i].y +=  Asw_total[i].y/SIZEL;
		}
	}
	for (int i = 0; i < SIZEL; i++){
		Asw_av[i].x /=  num_realiz;
		Asw_av[i].y /=  num_realiz;
	}
	
	printf("\nSaving signal spectrum...");
	sprintf(filename3, "signal_spectrum_delta_%.4f_gdd_%.2f.dat", delta, gdd);
	FILE *tres;
	tres= fopen(filename3, "w+");
	for ( int m = 0; m < SIZEL; m++ )
		fprintf(tres, "%15.35f\t%15.35f\n", Asw_av[m].x, Asw_av[m].y);// writing data into file
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
	free(Tp);	free(T);	free(F_p);
	free(w_ext);	free(F_ext);	free(Z);
	free(As);	free(Ap);	free(Ap_total); 
	free(As_total);	free(Asw_av);
	
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
