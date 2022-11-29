// Compilation line:        nvcc cwOPO.cu --gpu-architecture=sm_75 -lcufftw -lcufft -o cuOPO
//
// Consider changing the flag sm_75 by the proper one based in the GPU architecture



/**
 * This main file computes the evolution of the electric field
 * at the signal and pump frequency for a degenerate optical 
 * parametric oscillator. The code includes chirped mirrors
 * comensating for the group-velocity dispersion as well as
 * an intracavity electro-optical modulator.
 * 
 * The code receives a set of parameteres passed as an 
 * external arguments. For this propose, it is convenient to use
 * a shell script file in order to nicely manupulate the data.
 */

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
#include <thrust/host_vector.h>
#include <thrust/complex.h>
#include <cuComplex.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/generate.h>
#include <thrust/transform.h>
#include <thrust/random.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

#include "SaveFiles.cuh"
#include "common.cuh"
#include "fft.cuh"
#include "refindex.cuh"
#include "functors.cuh"
#include "operators.cuh"
#include "functions.cuh"




/** 
 * Complex data type: a set of datatypes are
 * defined to make the code more readable.
 *
 * Definitions for numbers
 * real_t : datatype for real numbers
 * complex_t     : datatype for complex numbers
 * 
 * Definitions for vectors:
 * 
 * rVech_t  : real vector host
 * rVecd_t  : real vector device
 * cVech_t  : complex vector host
 * cVecd_t  : complex vector device
 */

using real_t = float;
using complex_t = cuFloatComplex;
using rVech_t = thrust::host_vector<real_t>;
using rVecd_t = thrust::device_vector<real_t>;
using cVech_t = thrust::host_vector<complex_t>;
using cVecd_t = thrust::device_vector<complex_t>;	
	

int main(int argc, char *argv[]){
	
	
	const real_t PI   = 3.141592653589793238462643383279502884;	
	const real_t C    = 299792458*1E6/1E12;              // speed of ligth in vacuum [um/ps]
	const real_t EPS0 = 8.8541878128E-12*1E12/1E6;       // vacuum pertivity [W.ps/V²μm] 

	std::cout << "\n\n\n#######---Welcome to OPO calculator---#######\n\n\n" << std::endl;

	time_t current_time;
	time(&current_time);
	std::cout << ctime(&current_time) << std::endl;
	
	// timing the code
	double iStart = seconds();

	std::string Filename, SAux, Extension = ".dat";
	
	
	// Set up device //
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	std::cout << "\n\nUsing Device " << dev << ": GPU " << deviceProp.name << std::endl;
	CHECK(cudaSetDevice(dev));

	std::cout << "Setting constants and vectors in host..." << std::endl;
	
	// Set parameters and constants
	int N_rt                 = atoi(argv[8]); // number of round trips to cover the input pulse

	// Define wavelengths
	real_t lp              = 0.532;    // pump wavelength [μm]
	real_t ls              = 2*lp;     // signal wavelength [μm]

	real_t Temperature     = 27;       // Crystal temperature [ºC]
	real_t deff            = 14.77e-6; // effective d [um/V]
	real_t Lambda          = 6.97;     // grating period [um]
	real_t alpha_crp       = 0.025e-4; // pump linear absorption [1/μm]
	real_t alpha_crs       = 0.002e-4; // signal linear absorption [1/μm]

	real_t np              = n_PPLN(lp, Temperature); // refractive index at pump wavelength
	real_t vp              = group_vel_PPLN(lp, Temperature); // group velocity at pump wavelength
	real_t b2p             = gvd_PPLN(lp, Temperature);  // GVD at pump wavelength
	real_t b3p             = 0.*TOD_PPLN(lp, Temperature);       // pump TOD [ps³/μm]	
	real_t kp              = 2*PI*deff/(n_PPLN(lp, Temperature)*lp); // kappa pump [1/V]
	
	real_t ns              = n_PPLN(ls, Temperature);  // refractive index at signal wavelength
	real_t vs              = group_vel_PPLN(ls, Temperature); // group velocity at signal wavelength
	real_t b2s             = gvd_PPLN(ls, Temperature); // GVD at signal wavelength
	real_t b3s             = 0.*TOD_PPLN(ls, Temperature);       // signal TOD [ps³/μm]	
	real_t ks              = 2*PI*deff/(n_PPLN(ls, Temperature)*ls); // kappa signal [1/V]
	real_t dk              = 2*PI*( np/lp - 2*ns/ls - 1/Lambda ); // mismatch factor
	
	real_t Lcr             = 5e3;  // crystal length [um]
	real_t Lcav            = atof(argv[4]) * Lcr;  // cavity length [um]
	real_t R               = atof(argv[5])*0.01;  // net reflectivity 
	real_t t_rt            = (Lcav+Lcr*(n_PPLN(ls, Temperature)-1))/C;
	real_t FSR             = 1/t_rt; // free spectral range
	real_t delta           = atof(argv[6]); if(R<=0.5){delta *= 0.01;} else{delta *= 0.001;}
	real_t epsilon         = atof(argv[7])*0.01;
	real_t GDD             = -epsilon*b2s*Lcr; // GDD [ps²]
	real_t TOD             = -0.01*atof(argv[13])*b3s*Lcr; // TOD [ps³]

	real_t alphas          = 0.5*((1-R)+alpha_crs*Lcr);    // Total losses for threshold condition signal
	
	int steps_z              = atoi(argv[3]); // number of z step inside the crystal
	real_t dz              = Lcr/steps_z;   // z-step size
	
	// Time and frequency discretization
	
	unsigned int ex          = atoi(argv[2]);
	int N_ps                 = 1 << ex;  // points per time slice
	real_t dT              = t_rt/N_ps; // time step in [ps]
	real_t dF              = 1/t_rt; // frequency step in [THz]
	int SIZE                 = N_ps;
	unsigned int Nrts        = 32; // number of round trips to save
	int SIZEL                = N_ps*Nrts;
	real_t T_width         = (real_t ) (Nrts*t_rt); // total time for input ns-pulse
    
	/* vector T for one round trip */
	rVech_t T(SIZE); 
	linspace( T, -0.5*t_rt, 0.5*t_rt);

	/* vectors Tp for the complete time */
	rVech_t  Tp(SIZEL); 
	linspace( Tp, -0.5*T_width, 0.5*T_width);
	
	/* vector F_p for the complete pump frequency */
	real_t dF_p = 1/T_width;
	rVech_t F_p(SIZEL) ; inic_vector_F(F_p, dF_p);

	short unsigned int save_vectors = atoi(argv[1]);
	if (save_vectors == 1){
		std::cout << "\nSaving time and frequency vectors...\n" << std::endl;
		Filename = "Tp"; SaveFileVectorReal (Tp, Filename+Extension);
 		Filename = "freq"; SaveFileVectorReal (F_p, Filename+Extension);
		Filename = "T"; SaveFileVectorReal (T, Filename+Extension);
	}
	else{ std::cout << "\nTime and frequency were previuosly save...\n" << std::endl;}
	
	rVech_t F_ext(SIZE); inic_vector_F(F_ext, dF); // extended freq. grid [THz]

	rVech_t w(SIZE);  // angular frequency Ω
	cVech_t w_DISPp_h(SIZE); // e^(i.dz.((1/vs-1/vp).Ω + ½.β.Lcr.Ω²))
	cVech_t w_DISPs_h(SIZE); // i.½.β.Ω².dz
	cVech_t w_Comp_h(SIZE);  // e^(-i.½.ε.β.Lcr.Ω²)
	
	fftshift(w, F_ext);   // define ang ref for FFTs  [2*pi*THz]
	
	for ( int i = 0; i < w.size(); i++ ){
		w[i] *= 2*PI;
		w_DISPp_h[i].x = +cosf( dz * w[i] * ((1/vs-1/vp)+0.5*b2p*w[i]+b3p*w[i]*w[i]/6) );
		w_DISPp_h[i].y = +sinf( dz * w[i] * ((1/vs-1/vp)+0.5*b2p*w[i]+b3p*w[i]*w[i]/6) );
		w_DISPs_h[i].x = +cosf( dz * w[i] * ((1/vs-1/vs)+0.5*b2s*w[i]+b3s*w[i]*w[i]/6) );
		w_DISPs_h[i].y = +sinf( dz * w[i] * ((1/vs-1/vs)+0.5*b2s*w[i]+b3s*w[i]*w[i]/6) );
		w_Comp_h[i].x = +cosf( 0.5*GDD*w[i]*w[i] + TOD*w[i]*w[i]*w[i]/6 );
		w_Comp_h[i].y = +sinf( 0.5*GDD*w[i]*w[i] + TOD*w[i]*w[i]*w[i]/6 );
	}
	
	cVecd_t w_DISPp_d = w_DISPp_h, w_DISPs_d = w_DISPs_h, w_Comp_d = w_Comp_h; // copy to device the angular frequency vectors
	
		
	// Pumping parameters //
	
	std::string pump_regime = "cw";					// continuous wave pump
	// Define input pump parameters
	real_t waist = 55;             // beam waist radius [um]
	real_t spot  = PI*waist*waist; // spot area [μm²]
	#ifdef THREE_EQS
	// Power and intensity threshold non-degenerate DRO 
	real_t Ith   = EPS0*C*np*ns*ni*ls*li*pow((1/deff/Lcr/PI),2)*alphas*alphai/8;
	real_t Pth   = Ith*spot;
	#else
	// Power and intensity threshold degenerate DRO 
	real_t Ith   = EPS0*C*np*pow((ns*ls*alphas/deff/Lcr/PI), 2)/8;
	real_t Pth   = Ith*spot;
	#endif
	// Times over the threshold
	real_t Nth   = atof(argv[9]); 	
	// Pump intensity and power, times over the threshold in [W]
	real_t Inten = atof(argv[9])*Ith;
	real_t Power = Inten*spot; 
	// Input pump field strength [V/μm]
	real_t Ap0   = sqrt(2*Inten/(np*EPS0*C)) ;
	
	cVech_t Ap_in(SIZE);	
	InputField(Ap_in, Ap0, pump_regime);
	
	// Signal vector (is a complex noisy vector)
	cVech_t As(SIZE);	NoiseGeneratorCPU ( As );
	
	Filename = "signal_input";
	SaveFileVectorComplex(As, Filename);
	
	/********************************/
	// PHASE MODULATION
	bool using_phase_modulator = atoi(argv[10]);
	real_t mod_depth, fpm, df;
	cVech_t T_PM_h(SIZE);
	if(using_phase_modulator){		
		mod_depth       = atof(argv[11])*PI;
		df              = atof(argv[12])*sqrtf(Nth-1)*alphas/(PI*mod_depth)*FSR;
		fpm             = FSR - df;
	}
	else{std::cout << "No phase modulator" << std::endl;}
	
	for (int i = 0; i < T_PM_h.size(); i++){
		T_PM_h[i].x = cosf(mod_depth*sinf(2*PI*fpm*T[i]));
		T_PM_h[i].y = sinf(mod_depth*sinf(2*PI*fpm*T[i]));
	}
	
	cVecd_t T_PM_d = T_PM_h;		
	/********************************/
	

	bool prt_param_onscreen = true;
	if( prt_param_onscreen ){
		// Print parameters
		std::cout << "\n\nSimulation parameters:\n\n " << std::endl;
		std::cout << "Number of round trips   = " << N_rt  << std::endl;
		std::cout << "Pump wavelength         = " << lp*1e3 << " nm" << std::endl;
		std::cout << "Signal wavelength       = " << ls*1e3 << " nm" << std::endl;
		std::cout << "Temperature             = " << Temperature << " ºC" << std::endl;
		std::cout << "np                      = " << np << std::endl;
		std::cout << "ns                      = " << ns << std::endl;
		std::cout << "\u03BD⁻¹ pump                = " << 1.0/vp << " ps/\u03BCm" << std::endl;
		std::cout << "\u03BD⁻¹ signal              = " << 1.0/vs << " ps/\u03BCm" << std::endl;
		std::cout << "\u0394k                      = " << dk << " \u03BCm⁻¹" << std::endl;
		std::cout << "GVD pump                = " << b2p << " ps²/\u03BCm" << std::endl;
		std::cout << "GVD signal              = " << b2s << " ps²/\u03BCm" << std::endl;
		std::cout << "TOD pump                = " << b3p << " ps³/\u03BCm" << std::endl;
		std::cout << "TOD signal              = " << b3s << " ps³/\u03BCm" << std::endl;
		std::cout << "Net GVD                 = " << (1-epsilon)*b2s << " ps²/\u03BCm" << std::endl;
		std::cout << "Net TOD                 = " << (1-0.01*atof(argv[13]))*b3s << " ps³/\u03BCm" << std::endl;
		std::cout << "GVD compensation        = " << atof(argv[7]) << " %"  << std::endl;
		std::cout << "TOD compensation        = " << atof(argv[13]) << " %"  << std::endl;
		std::cout << "Cavity net dispersion   = " << (1-epsilon)*b2s*Lcr*1e6 << " fs²"  << std::endl;
		std::cout << "deff                    = " << deff*1e6 << " pm/V"  << std::endl;
		std::cout << "\u039B                       = " << Lambda << " \u03BCm"  << std::endl;
		std::cout << "\u03B1_cp                    = " << alpha_crp << " \u03BCm⁻¹"  << std::endl;
		std::cout << "\u03B1_cs                    = " << alpha_crs << " \u03BCm⁻¹" << std::endl;
		std::cout << "\u03B1s                      = " << alphas << " \u03BCm⁻¹" << std::endl;		
		std::cout << "Crystal length          = " << Lcr*1e-3 << " mm"  << std::endl;
		std::cout << "Cavity  length          = " << Lcav*1e-3 << " mm"  << std::endl;
		std::cout << "\u0394z                      = " << dz << " \u03BCm"  << std::endl;
		std::cout << "Reflectivity            = " << R*100 << " %"  << std::endl;	
		std::cout << "Round-trip time         = " << std::setprecision(15) << t_rt << " ps"  << std::endl;	
		std::cout << "FSR                     = " << std::setprecision(15) << FSR*1e3 << " GHz"  << std::endl;
		std::cout << "Cavity detuning (\u03B4)     = " << delta << "\u03C0"  << std::endl;	
		std::cout << "Using N                 = 2^" << ex << " = " << N_ps << " points" << std::endl;
		std::cout << "dT                      = " << dT << " ps" << std::endl;
		std::cout << "SIZEL                   = " << SIZEL << std::endl;
		std::cout << "dF_p                    = " << dF_p << " THz" << std::endl;
		std::cout << "Ap0                     = " << Ap0 << " V/um" << std::endl; 
		std::cout << "waist                   = " << waist << " \u03BCm" << std::endl;
		std::cout << "spot                    = " << spot << " \u03BCm²" << std::endl;
		std::cout << "Power threshold         = " << Pth << " W" << std::endl;
		std::cout << "Power                   = " << Power << " W" << std::endl;
		std::cout << "Times above the thres.  = " << Nth << std::endl;
		if(using_phase_modulator){
			std::cout << "\n\nUsing a phase modulator:\n" << std::endl;
			std::cout << "Mod. depth (\u03B2)          = " << atof(argv[11]) << "\u03C0" << std::endl;
			std::cout << "Mod. frequency (fpm)    = " << fpm*1e3 << " GHz" << std::endl;
			std::cout << "Frequency detuning (\u03B4f) = " << df*1e6 << " MHz" << std::endl;
		}
		else{std::cout << "No phase modulator" << std::endl;}
	}

	
	/********************************/
	// Device vectors	//    
	cVecd_t As_d = As, 	Asw_d(SIZE),	As_total_d;   // signal device vectors
	cVecd_t Ap_d(SIZE),	Apw_d(SIZE);	              // pump device vectors

	
	// Auxiliar device vectors //
	cVecd_t k1p(SIZE), k2p(SIZE), k3p(SIZE), k4p(SIZE), auxp(SIZE);
	cVecd_t k1s(SIZE), k2s(SIZE), k3s(SIZE), k4s(SIZE), auxs(SIZE);
	/********************************/
	
	
	// Set plan for cuFFT 1D and 2D//
	cufftHandle plan;		cufftPlan1d(&plan, SIZE, CUFFT_C2C, 1);
	
	std::cout << "Starting main loop on GPU...\n" << std::endl;
	for (int nn = 0; nn < N_rt; nn++){
		if( nn%500 == 0 or nn == N_rt-1 )
			std::cout << "#round trip: " << nn << std::endl;

		Ap_d = Ap_in; // In every round trip, Ap <- Bin

		// Evolution along the nonlinear crystal
		SinglePass( plan, w_DISPp_d, w_DISPs_d, Ap_d,
				As_d, Apw_d, Asw_d, k1p, k1s,
				k2p, k2s, k3p, k3s, k4p, k4s,
				auxp, auxs,	dk, alpha_crp, alpha_crs,
				kp, ks, dz, steps_z );

		ifft ( As_d, Asw_d, plan );
// 		thrust::transform( w_Comp_d.begin(), w_Comp_d.end(), Asw_d.begin(), Asw_d.begin(), ComplexMult());
		Asw_d *= w_Comp_d;		
		fft ( Asw_d, As_d, plan );
		
		AddPhase( As_d, R, delta, nn );	// Add phase and loss (R)
		
		if( using_phase_modulator )		// Add intracavity phase modulator
			As_d *= T_PM_d;
		
		if (nn >= N_rt - Nrts)			// Accumulates several round trips to save them
			As_total_d.insert(As_total_d.end(),
						As_d.begin(), As_d.end());
	}
	
	// Destroy CUFFT context //
	cufftDestroy(plan); 
	
	
	
	// Save outputs //
	cVech_t As_total = As_total_d;
	std::cout << "Saving outputs..." << std::endl;
	Filename = "signal_total_delta_", SAux = std::to_string(delta);
	SaveFileVectorComplex (As_total, Filename+SAux);
	
	As = As_d;
	Filename = "signal_last_rt";
	SaveFileVectorComplex (As, Filename);
	
	cVech_t Ap = Ap_d;
	Filename = "pump_last_rt";
	SaveFileVectorComplex (Ap, Filename);
	
	
	
	// Finish timing
	double iElaps = seconds() - iStart;
	if(iElaps>60){std::cout << "\n\n...time elapsed " <<  iElaps/60.0 << " min\n\n " << std::endl;}
	else{std::cout << "\n\n...time elapsed " <<  iElaps << " min\n\n " << std::endl;}

	time(&current_time);
	std::cout << ctime(&current_time) << std::endl;
	
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
