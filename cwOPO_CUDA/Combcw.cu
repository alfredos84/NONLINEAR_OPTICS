// Compile with "nvcc Comb_cw.cu functions.cu --gpu-architecture=sm_60 -lcufftw -lcufft -lcurand -o Comb_cw" for GeForce MX250 (Pascal)
//-gencode=arch=compute_60,code=sm_60

#include <iostream>
#include <string>
#include <random>
#include <fstream>
#include <iomanip>
#include <typeinfo>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <stdio.h>

#include <sys/time.h>

#include <cuda_runtime.h>
#include <cufft.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
// #include <curand.h>

#include "common.h"
#include "refindex.h"
#include "functions.h"
#include "SaveFiles.h"

// Complex data type
#ifdef DOUBLEPRECISION
	typedef cufftDoubleComplex CC;
	typedef double typefl_t;
#else
	typedef cufftComplex CC;
	typedef float typefl_t;
#endif
	

int main(int argc, char *argv[]){
	
	const typefl_t PI   = 3.14159265358979323846;          //pi
	const typefl_t C    = 299792458*1E6/1E12;              // speed of ligth in vacuum [um/ps]
	const typefl_t EPS0 = 8.8541878128E-12*1E12/1E6;       // vacuum pertivity [W.ps/V²μm] 

	// Display each command-line argument.
	std::cout << "\nCommand-line arguments:\n";
	for( int count = 0; count < argc; count++ )
		std::cout << "  argv[" << count << "]   " << argv[count] << "\n";
	
	std::cout << "\n\n\n#######---Welcome to OPO calculator---#######\n\n\n" << std::endl;
	#ifdef DOUBLEPRECISION
		std::cout << "Using DOUBLE PRECISION values" << std::endl;
	#else
		std::cout << "Using SINGLE PRECISION values" << std::endl;
	#endif
	time_t current_time;
	time(&current_time);
	printf("%s", ctime(&current_time));

    // timing the code
	double iStart = seconds();

	std::string Filename, SAux, Extension = ".dat";

	// Set up device //
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("\n\nUsing Device %d: GPU %s\n\n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));

	printf("Setting constants and vectors in host... \n");
	
	// Set parameters and constants
	int N_rt                 = atoi(argv[8]); // number of round trips to cover the input pulse

	// Define wavelengths
	typefl_t lp              = 0.532;//atof(argv[7])*1e-3; // pump wavelength [μm]
	typefl_t ls              = 2*lp; // signal wavelength [μm]
// 	typefl_t li              = 1/(1/lp-1/ls); 

	typefl_t Temperature     = 27;
	typefl_t deff            = 14.77e-6; // effective d [pm/V]
	typefl_t Lambda          = 6.97;     // grating period [um]
	typefl_t dk              = 0*2*PI*( n_PPLN(lp, Temperature)/lp - 2* n_PPLN(ls, Temperature)/ls - 1/Lambda);
	typefl_t alphap          = 0.002e-4; // pump linear absorption [1/um]
	typefl_t alphas          = 0.025e-4; // signal linear absorption [1/um]
	typefl_t rho             = 0;        // walk-off angle
	typefl_t Lcr             = 5e3;  // crystal length [um]
	typefl_t Lcav            = atoi(argv[4]) * Lcr;  // cavity length [um]
	typefl_t R               = atof(argv[5])*0.01;  // net reflectivity 
	typefl_t t_rt            = (Lcav-Lcr)/C + Lcr/group_vel_PPLN(lp,Temperature);
	typefl_t FSR             = 1/t_rt;
	typefl_t delta           = atof(argv[6]); if(R<0.5){delta *= 0.01;} else{delta *= 0.001;}
	typefl_t gdd             = atof(argv[7])*0.01;
	typefl_t GDD             = -gdd*gvd_PPLN(ls,Temperature)*Lcr; // GDD [ps²]

	int steps_z              = atoi(argv[3]);
	typefl_t dz              = Lcr/steps_z; // number of z step inside the crystal
	typefl_t *Z              = (typefl_t*) malloc(sizeof(typefl_t) * steps_z);
	inic_vector_Z(Z, steps_z, dz);
	
	// Time and frequency discretization
	
	unsigned int ex          = atoi(argv[2]);
	int N_ps                 = 1<<ex;  // points per time slice
	int extra_win            = 16; // extra pts for short-time slices
	typefl_t dT              = t_rt/N_ps; // time step in [ps]
	typefl_t dF              = 1/t_rt; // frequency step in [THz]
	int SIZE                 = N_ps+extra_win;
// 	int SIZEL          = N_ps*N_rt;
	unsigned int Nrts        = 256; // number of round trips to save
	int SIZEL                = N_ps*Nrts;
	typefl_t T_width         = (typefl_t ) (Nrts*t_rt); // total time for input ns-pulse
    
	/* vector T for one round trip */
	typefl_t *T = (typefl_t*) malloc(sizeof(typefl_t) * SIZE);
	inic_vector_T(T, SIZE, t_rt, dT); // temp. grid for slice

	/* vectors Tp for the complete time */
	typefl_t *Tp = (typefl_t*) malloc(sizeof(typefl_t) * SIZEL);
	inic_vector_T(Tp, SIZEL, T_width, dT); // temp. grid for pump field

	/* vector F_p for the complete pump frequency */
	typefl_t dF_p  = 1/T_width;
	typefl_t *F_p = (typefl_t*) malloc(sizeof(typefl_t) * SIZEL);
	inic_vector_F(F_p, SIZEL, dF_p);
	
	std::cout << "\n\nSimulation parameters:\n\n " << std::endl;
	std::cout << "Number of round trips   = " << N_rt  << std::endl;
	std::cout << "Pump wavelength         = " << lp*1e3 << " nm" << std::endl;
	std::cout << "Signal wavelength       = " << ls*1e3 << " nm" << std::endl;
	std::cout << "Temperature             = " << Temperature << " ºC" << std::endl;
	std::cout << "np                      = " << n_PPLN(lp, Temperature) << std::endl;
	std::cout << "ns                      = " << n_PPLN(ls, Temperature) << std::endl;
	std::cout << "\u03BD⁻¹ pump                = " << 1.0/group_vel_PPLN(lp,Temperature) << " ps/\u03BCm" << std::endl;
	std::cout << "\u03BD⁻¹ signal              = " << 1.0/group_vel_PPLN(ls,Temperature) << " ps/\u03BCm" << std::endl;
	std::cout << "GVD pump                = " << gvd_PPLN(lp,Temperature) << " ps²/\u03BCm" << std::endl;
	std::cout << "GVD signal              = " << gvd_PPLN(ls,Temperature) << " ps²/\u03BCm" << std::endl;
	std::cout << "\u0394k                      = " << dk << " \u03BCm⁻¹" << std::endl;
	std::cout << "GDD                     = " << GDD*1e3 << " fs²"  << std::endl;
	std::cout << "Cavity net dispersion   = " << (gvd_PPLN(ls,Temperature)*Lcr+GDD)*1e3 << " fs²"  << std::endl;
	std::cout << "deff                    = " << deff*1e6 << " pm/V"  << std::endl;
	std::cout << "\u039B                       = " << Lambda << " \u03BCm"  << std::endl;
	std::cout << "\u03B1p                      = " << alphap << " \u03BCm⁻¹"  << std::endl;
	std::cout << "\u03B1s                      = " << alphas << " \u03BCm⁻¹" << std::endl;
	std::cout << "Crystal length          = " << Lcr*1e-3 << " mm"  << std::endl;
	std::cout << "Cavity  length          = " << Lcav*1e-3 << " mm"  << std::endl;
	std::cout << "\u0394z                      = " << dz << " \u03BCm"  << std::endl;
	std::cout << "Reflectivity            = " << R*100 << " %"  << std::endl;	
	std::cout << "Round-trip time         = " << t_rt << " ps"  << std::endl;	
	std::cout << "FSR                     = " << FSR*1e3 << " GHz"  << std::endl;
	std::cout << "Cavity detuning (\u03B4)     = " << delta << "\u03C0"  << std::endl;	
	std::cout << "Using N                 = 2^" << ex << " = " << N_ps << " points" << std::endl;
	std::cout << "dT                      = " << dT << " ps" << std::endl;
	std::cout << "SIZEL                   = " << SIZEL << std::endl;
	std::cout << "dF_p                    = " << dF_p << " THz" << std::endl;
	std::cout << "Max frequency           = " << F_p[SIZEL-1] << " THz" << std::endl;
	
	short unsigned int save_vectors = atoi(argv[1]);
	if (save_vectors == 1){
		std::cout << "\nSaving time and frequency vectors...\n" << std::endl;
		Filename = "Tp"; SaveFileVectorReal (Tp, SIZEL, Filename+Extension);
 		Filename = "freq"; SaveFileVectorReal (F_p, SIZEL, Filename+Extension);
		Filename = "T"; SaveFileVectorReal (T, SIZE, Filename+Extension);
	}
	else{ std::cout << "\nTime and frequency were previuosly save...\n" << std::endl;
	}
	
	typefl_t *F_ext = (typefl_t*) malloc(sizeof(typefl_t) * SIZE);
	inic_vector_F(F_ext, SIZE, dF); // extended freq. grid [THz]

	typefl_t *w_ext = (typefl_t*) malloc(sizeof(typefl_t) * SIZE);
	fftshift(w_ext,F_ext, SIZE);
	for (int i=0; i<SIZE; i++)
		w_ext[i] = 2*PI*w_ext[i];  // ang freq [2*pi*THz]

	// Host vectors //
	int nBytes =  sizeof(CC)*SIZE;
	CC *As_total   = (CC*)malloc(sizeof(CC) * SIZEL);

	std::cout << "Defining input field..." << std::endl;
	/* Define input pump */
// 	CC *Ap_total             = (CC*)malloc(sizeof(CC) * SIZE);
	CC *Ap_in                = (CC*)malloc(nBytes);
// 	CC *Ap_total_out         = (CC*)malloc(sizeof(CC) * SIZEL);
	typefl_t FWHM            = 8000; // input pump with fwhm [ps]
	typefl_t sigmap          = FWHM/(2*sqrt(2*log(2))); // standar deviation for gaussian pulse [ps]
	typefl_t waist           = 55; // beam waist radius [um]
	typefl_t spot            = PI*waist*waist; // spot area [μm²]
	typefl_t Power           = atof(argv[9]);
	typefl_t Ap0             = sqrt(2*Power/(spot*n_PPLN(ls,Temperature)*EPS0*C)) ; // input field amplitud [V/μm] 	

	std::cout << "Ap0                     = " << Ap0 << " V/um" << std::endl; 
	std::cout << "FWHM                    = " << FWHM*1e-3 << " ns" << std::endl; 
	std::cout << "\u03C3                       = " << sigmap*1e-3 << " ns" << std::endl; 
	std::cout << "waist                   = " << waist << " \u03BCm" << std::endl;
	std::cout << "spot                    = " << spot << " \u03BCm²" << std::endl;
	std::cout << "Power                   = " << Power << " W" << std::endl;
	std::cout << "Intracavity power       = " << (1-R)*Power*1e3 << " mW" << std::endl;

	char m = 'c'; // Wave form: 'c' = CW, 'g' = gaussian pulse, 's' = soliton//

	// 	input_field_T(Ap_total, Tp, SIZEL, sigmap*sqrt(2), Ap0*Ap0, m); // input pump
	input_field_T(Ap_in, T, SIZE, sigmap*sqrt(2), Ap0*Ap0, m); // input pump
	
	CC *As = (CC*)malloc(nBytes);
	NoiseGeneratorCPU ( As, SIZE ); // inicial noisy signal and idler


	// PHASE MODULATION
	bool using_phase_modulator = true;
	typefl_t mod_depth, fact, OMEGA;
	if(using_phase_modulator){
		mod_depth       = 0.5;//atof(argv[11])/100;
		OMEGA           = 0.004653930664063;
		std::cout << "Using a phase modulator:" << std::endl;
		std::cout << "Modulation depth (\u03B2)         = " << mod_depth << std::endl;
		std::cout << "Modulator frequency (\u03A9m)     = " << OMEGA << std::endl;
	}
	else{std::cout << "No phase modulator" << std::endl;}
	
	// Device vectors	
	// Parameters for kernels
	int dimx = 1 << 7; //ceil(SIZE/1024);
	dim3 block(dimx);
	int N = SIZE;
	dim3 grid((N + block.x - 1) / block.x);
	printf("\nKernels dimensions:\n<<<grid_dim, block_dim>>> = <<<(%d,%d), (%d,%d)>>>\n", grid.x, grid.y, block.x, block.y);
	
	std::cout << "Setting constants and vectors in device..." << std::endl;
	typefl_t *w_ext_gpu;
	CHECK(cudaMalloc((void **)&w_ext_gpu, sizeof(typefl_t) * SIZE ));

	typefl_t *T_gpu;
	CHECK(cudaMalloc((void **)&T_gpu, sizeof(typefl_t) * SIZE ));    
	CHECK(cudaMemcpy(T_gpu, T, sizeof(typefl_t)*SIZE, cudaMemcpyHostToDevice));    
    
	CC *As_gpu, *Ap_gpu, *Ap_in_gpu, *As_total_gpu, *Asw_gpu, *Apw_gpu;
	
	CHECK(cudaMalloc((void **)&As_gpu, nBytes ));
	CHECK(cudaMalloc((void **)&Ap_gpu, nBytes ));
	CHECK(cudaMemset(Ap_gpu, 0, nBytes));
	CHECK(cudaMalloc((void **)&Ap_in_gpu, nBytes ));
	CHECK(cudaMemset(Ap_in_gpu, 0, nBytes));
	CHECK(cudaMalloc((void **)&As_total_gpu, sizeof(CC) * SIZEL ));
	CHECK(cudaMalloc((void **)&Asw_gpu, nBytes ));
	CHECK(cudaMalloc((void **)&Apw_gpu, nBytes ));

	CHECK(cudaMemcpy(As_gpu, As, nBytes, cudaMemcpyHostToDevice));	
	CHECK(cudaMemcpy(Ap_in_gpu, Ap_in, nBytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(w_ext_gpu, w_ext, sizeof(typefl_t) * SIZE , cudaMemcpyHostToDevice));
    
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
	
	// Set plan for cuFFT 1D and 2D//
	cufftHandle plan1D; 
	#ifdef DOUBLEPRECISION
		cufftPlan1d(&plan1D, SIZE, CUFFT_Z2Z, 1);
	#else
		cufftPlan1d(&plan1D, SIZE, CUFFT_C2C, 1);
	#endif	

	std::cout << "Starting main loop on CPU & GPU...\n" << std::endl;
	// START MAIN LOOP //	
	unsigned int mm = 0;
	for (int nn = 0; nn < N_rt; nn++){
		if( nn%500 == 0 or nn == N_rt-1 )
			std::cout << "#round trip: " << nn << std::endl;

		CHECK(cudaMemcpy(Ap_gpu, Ap_in_gpu, nBytes, cudaMemcpyDeviceToDevice));
		
		EvolutionInCrystal( w_ext_gpu, grid, block, Ap_gpu, As_gpu, Apw_gpu, Asw_gpu, k1p_gpu, k1s_gpu, k2p_gpu, k2s_gpu, k3p_gpu, k3s_gpu, k4p_gpu, k4s_gpu, auxp_gpu, auxs_gpu, lp, ls, Temperature, alphap, alphas, deff, Lambda, rho, dz, steps_z, SIZE, nBytes );
		
		AddPhase<<<grid,block>>>(As_gpu, auxs_gpu, R, delta, nn, SIZE);
		CHECK(cudaDeviceSynchronize());
				
		if(GDD!=0){
			#ifdef DOUBLEPRECISION
				cufftExecZ2Z(plan1D, (CC *)As_gpu, (CC *)Asw_gpu, CUFFT_INVERSE);
				CHECK(cudaDeviceSynchronize());
				CUFFTscale<<<grid,block>>>(Asw_gpu, SIZE, SIZE);
				CHECK(cudaDeviceSynchronize());
				AddGDD<<<grid,block>>>(Asw_gpu, auxs_gpu, w_ext_gpu, GDD, SIZE);
				CHECK(cudaDeviceSynchronize());
				cufftExecZ2Z(plan1D, (CC *)Asw_gpu, (CC *)As_gpu, CUFFT_FORWARD);
				CHECK(cudaDeviceSynchronize());
			#else
				cufftExecC2C(plan1D, (CC *)As_gpu, (CC *)Asw_gpu, CUFFT_INVERSE);
				CHECK(cudaDeviceSynchronize());
				CUFFTscale<<<grid,block>>>(Asw_gpu, SIZE, SIZE);
				CHECK(cudaDeviceSynchronize());
				AddGDD<<<grid,block>>>(Asw_gpu, auxs_gpu, w_ext_gpu, GDD, SIZE);
				CHECK(cudaDeviceSynchronize());
				cufftExecC2C(plan1D, (CC *)Asw_gpu, (CC *)As_gpu, CUFFT_FORWARD);
				CHECK(cudaDeviceSynchronize());
			#endif
		}
		if( using_phase_modulator ){
// 			std::cout << "Using modulator intracavity." << std::endl;
			PhaseModulatorIntraCavity<<<grid,block>>>(As_gpu, auxs_gpu, mod_depth, OMEGA, T_gpu, SIZE);
			CHECK(cudaDeviceSynchronize());
		}
		if (nn >= N_rt - Nrts){                
			WriteField<<<grid,block>>>(As_total_gpu, As_gpu, mm, N_ps, extra_win, Nrts);
			CHECK(cudaDeviceSynchronize());
			mm += 1;
		}
	}
	CHECK(cudaMemcpy(As_total, As_total_gpu, sizeof(CC) * SIZEL, cudaMemcpyDeviceToHost));
	std::cout << "Saving outputs..." << std::endl;
	Filename = "signal_total_delta_", SAux = std::to_string(delta);
	SaveFileVectorComplex (As_total, SIZEL, Filename+SAux);


	// Deallocating memory and destroying plans //
	free(Tp);
	free(T);
	free(F_p);
	free(w_ext);
	free(F_ext);
	free(Z);
	free(As);
	//free(Ap);	
	free(As_total);
	free(Ap_in);


	CHECK(cudaFree(As_gpu));
	CHECK(cudaFree(Ap_gpu));
	CHECK(cudaFree(As_total_gpu));   
	CHECK(cudaFree(Ap_in_gpu));	
	CHECK(cudaFree(T_gpu));
	CHECK(cudaFree(w_ext_gpu));
	CHECK(cudaFree(k1p_gpu));        CHECK(cudaFree(k2p_gpu));
	CHECK(cudaFree(k3p_gpu));        CHECK(cudaFree(k4p_gpu));
	CHECK(cudaFree(k1s_gpu));        CHECK(cudaFree(k2s_gpu));
	CHECK(cudaFree(k3s_gpu));        CHECK(cudaFree(k4s_gpu));	
	CHECK(cudaFree(auxs_gpu));       CHECK(cudaFree(auxp_gpu));

	
	// Destroy CUFFT context //
	cufftDestroy(plan1D); 
	cudaDeviceReset();
    
	// finish timing
	double iElaps = seconds() - iStart;
	if(iElaps>60){printf("\n\n...time elapsed %.3f min \n\n\n", iElaps/60);}
	else{printf("\n\n...time elapsed %.3f sec \n\n\n", iElaps);}

	time(&current_time);
	printf("%s", ctime(&current_time));

	return 0;
}


/**
 Letter   Description  Escape-Sequence
-------------------------------------
A        Alpha        \u0391
B        Beta         \u0392
Γ        Gamma        \u0393
Δ        Delta        \u0394
Ε        Epsilon      \u0395
Ζ        Zeta         \u0396
Η        Eta          \u0397
Θ        Theta        \u0398
Ι        Iota         \u0399
Κ        Kappa        \u039A
Λ        Lambda       \u039B
Μ        Mu           \u039C
Ν        Nu           \u039D
Ξ        Xi           \u039E
Ο        Omicron      \u039F
Π        Pi           \u03A0
Ρ        Rho          \u03A1
Σ        Sigma        \u03A3
Τ        Tau          \u03A4
Υ        Upsilon      \u03A5
Φ        Phi          \u03A6
Χ        Chi          \u03A7
Ψ        Psi          \u03A8
Ω        Omega        \u03A9 
-------------------------------------
Letter   Description  Escape-Sequence
-------------------------------------
α        Alpha        \u03B1
β        Beta         \u03B2
γ        Gamma        \u03B3
δ        Delta        \u03B4
ε        Epsilon      \u03B5
ζ        Zeta         \u03B6
η        Eta          \u03B7
θ        Theta        \u03B8
ι        Iota         \u03B9
κ        Kappa        \u03BA
λ        Lambda       \u03BB
μ        Mu           \u03BC
ν        Nu           \u03BD
ξ        Xi           \u03BE
ο        Omicron      \u03BF
π        Pi           \u03C0
ρ        Rho          \u03C1
σ        Sigma        \u03C3
τ        Tau          \u03C4
υ        Upsilon      \u03C5
φ        Phi          \u03C6
χ        Chi          \u03C7
ψ        Psi          \u03C8
ω        Omega        \u03C9
-------------------------------------
*/
