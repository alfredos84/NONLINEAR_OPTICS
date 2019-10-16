// Compile with "nvcc SSFM.cu -DSELFSTEEPENING -DRAMAN -DNOISE functions.cu --gpu-architecture=sm_30 -lcufftw -lcufft -lcurand -o SSFMcu" for GeForce 750 (Kepler)
// Compile with "nvcc SSFM.cu -DSELFSTEEPENING -DRAMAN -DNOISE functions.cu --gpu-architecture=sm_35 -lcufftw -lcufft -lcurand -o SSFMcu" for Tesla K40 (Kepler)
// Compile with "nvcc SSFM.cu -DSELFSTEEPENING -DRAMAN -DNOISE functions.cu --gpu-architecture=sm_52 -lcufftw -lcufft -lcurand -o SSFMcu" for GTX 980 Ti (Maxwell)

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

const double PI2 = 2.0 * 3.14159265358979323846;    //2*pi
const double C = 299792458*1E9/1E12;                // speed of ligth in vacuum [nm/ps]
#define NELEMS(x)  (sizeof(x) / sizeof((x).x))      // number of elements of an array

int main(int argc, char *argv[]){
    
    int n_realiz  = atoi(argv[1]);
    char filename1[30],filename2[30],filename5[30],filename3[30],filename4[30];    
    double factores[] = { 1.1, 1.3, 4.0 };
    double distancias[] = { 0.6, 1.3, 8.3};    
//     double factores[] = { 1.1, 1.2, 1.3, 1.5, 2, 3, 4 };    
//     double distancias[] = { 0.6, 0.9, 1.3, 2.0, 3.8, 7.4, 11 };    
    size_t n_fact = sizeof(factores)/sizeof(factores[0]);
    for (int pp = 0; pp < (int)n_fact; pp++){        
        for ( int nr = 0; nr < n_realiz; nr++ ){
            printf("p = %.2f - Realisation #%d/%d\n", factores[pp], nr+1, n_realiz);
            int i; 
            int N = 1<<15; // number of points

            // Parameters for kernels //
            int dimx = 1 << 5;
            dim3 block(dimx);
            dim3 grid((N + block.x - 1) / block.x);
            printf("Kernels dims: <<<(%d,%d), (%d,%d)>>>\n", grid.x, grid.y, block.x, block.y);
            // Set up device //
            int dev = 0;
            cudaDeviceProp deviceProp;
            CHECK(cudaGetDeviceProperties(&deviceProp, dev));
            printf("Using Device %d: %s\n\n", dev, deviceProp.name);
            CHECK(cudaSetDevice(dev));

            int nBytes =  sizeof(CC)*N;
            double dT = 0.001; // time step [ps]
            double t_width = (double )(N * dT ); // time window size
            double dF = 1/t_width; // frequency step [ps]
            double T0 = 0.1; //temporal width of pulses [ps]
            double lambda_0 = 5000; // central wavelength [nm]
            double w0 = PI2 * C / lambda_0; // angular frequency in 2*pi*[THz]
            double betas[3] = {-1E-3, 0*0.004E-3,-0*0.0016E-3}; // betas [ps^i / m]
            //double betas[3] = {-0.020E-3, 0, 0}; // betas [ps^i / m]
            int lb = 3; // number of betas that are included
            double sol_ord, P0, factor; // soliton order, power and factor for anomalous dispersion

            double gamma = 0.10; // nonlinear parameter gamma [1/W/m]
            double tau1 = 0.0155, tau2 = 0.2305; // Raman times [ps] 

            #if defined(RAMAN)
                double fr = 0.031; // fractional Raman contribution
            #endif
            #if !defined(RAMAN)
                double fr = 0.00; // fractional Raman contribution
            #endif

            char kindpower = 'p'; // select among different kind of power
            switch(kindpower) {
                case 'n': // select soliton order and then associated power will be computed
                    sol_ord = 1; // soliton order
                    P0 = pow(sol_ord,2) * fabs(betas[0])/(gamma*pow(T0,2));
                    break;
                case 'p': // select power and then soliton order will be computed                
                    factor = factores[pp]; // normalized power from cutoff
                    P0 = (fabs(betas[0])* w0 * w0/gamma)*factor; // peak power of input [W]
                    sol_ord = sqrt(P0*gamma*pow(T0,2)) / fabs(betas[0]);
                    break;
                case 'a': // arbitrary power
                    P0 =200; // peak power of input [W]
                    factor = factores[pp];
                    break;
            }

            // Distances //
//             double LD = pow(T0,2) / fabs(betas[0]);  // dispersion lenght
//             double LD3 = pow(T0,3) / fabs(betas[1]); // third order dispersion length
//             double LNL = 1/gamma/P0; // nonlinear length
//             double Zfiss = LD/sol_ord; // soliton fission length
//             double Zsol = 0.5 * 3.14159265358979323846 * LD; // soliton period            
            double flength = distancias[pp];
            double h = flength/100000; // z step

            int steps_z = (int )floor(flength/h); // number of steps in Z
            
            // Set plan for cuFFT //
            cufftHandle plan_1;
            cufftPlan1d(&plan_1, N, CUFFT_Z2Z, 1);
            
            // Host vectors //
            CC *u1 = (CC*)malloc(nBytes);	CC *u1_W = (CC*)malloc(nBytes);
            CC *D_OP = (CC*)malloc(nBytes);  // Linear operator exp(Dh/2)
            CC *hR = (CC*)malloc(nBytes);  // Raman response in time domain
            CC *hR_W = (CC*)malloc(nBytes);  // Raman response in frequency domain
            CC *self_st = (CC*)malloc(nBytes);  // Self-steepening
            CC *V_ss = (CC*)malloc(nBytes);
                        
            /* Time, frequency and Z vectors*/
            double *T;    
            T = (double*) malloc(sizeof(double) * N);
            inic_vector_T(T, N, t_width, dT);
            
            double *TT;    
            TT = (double*) malloc(sizeof(double) * N);
            inic_vector_Traman(TT, N, t_width);
            
            double *V;    
            V = (double*) malloc(sizeof(double) * N);
            inic_vector_F(V, N, dF);
            
            freq_shift( V_ss, V, N ); //frequecy used in DOP and self-steepening
            
            double *Z;
            Z = (double*) malloc(sizeof(double) * steps_z);
            inic_vector_Z(Z, steps_z, h);
            
            if( (nr == 0) && (pp == 0) ){
                /* Saving some vectors */
                FILE *uno;	
                uno = fopen("T.txt", "w+");
                for ( int i = 0; i < N; i++ )
                    fprintf(uno, "%15.10f\n", T[i]);// writing data into file
                fclose(uno);//closing file
                    
                FILE *dos;
                dos = fopen("V.txt", "w+");
                for ( int i = 0; i < N; i++ )
                    fprintf(dos, "%15.10f\n", V[i]);// writing data into file
                fclose(dos);//closing file             
            }
            
            // Device vectors //
            CC *d_self_st, *d_D_OP, *d_u_ip, *d_alpha1, *d_alpha2, *d_alpha3, *d_alpha4, *d_u1_W, *d_u1, *d_u2_W, *d_u2, *d_u3_W, *d_u3, *d_u4_W, *d_u4, *d_hR, *d_hR_W;
            
            // Raman //
            RAMAN_RESP(hR, N, tau1, tau2, TT);
            CHECK(cudaMalloc((void **)&d_hR, nBytes));
            CHECK(cudaMalloc((void **)&d_hR_W, nBytes));
            CHECK(cudaMemset(d_hR, 0, nBytes));
            CHECK(cudaMemset(d_hR_W, 0, nBytes));
            CHECK(cudaMemcpy(d_hR, hR, nBytes, cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_hR_W, hR_W, nBytes, cudaMemcpyHostToDevice));
            cufftExecZ2Z(plan_1, (CC *)d_hR, (CC *)d_hR_W, CUFFT_INVERSE);
            CHECK(cudaDeviceSynchronize());
            scale<<<grid,block>>>(d_hR_W, N, dT);
            CHECK(cudaDeviceSynchronize());            
            CHECK(cudaMemcpy(hR_W, d_hR_W, nBytes, cudaMemcpyDeviceToHost));
            
            #if defined(SELFSTEEPENING)
                inic_selfst(self_st, V_ss, w0, gamma, N);
            #endif
            #if !defined(SELFSTEEPENING)
                inic_selfst(self_st, V_ss, 0, gamma, N);
            #endif
            
            // Print parameters // 
            printf("Parameters\n\nN = %i points\nbeta2 = %f ps^2/km\nbeta3 = %f ps^3/km\ngamma = %.3f 1/W/km\ntwidth = %f ps\ndT = %f ps\ndF = %f THz\nfr = %f\ntau1 = %f ps\ntau2 = %f ps\nw0 = %.2f THz\nlambda0 = %.1f nm\nstep size = %f m\nDistance = %f m\nNumber of steps %d\nPower = %.2f W\n",N, betas[0]*1000, betas[1]*1000, gamma*1000, t_width, dT, dF, fr, tau1, tau2, w0, lambda_0, h,flength, steps_z, P0);
            
            
            // Input field and envelope expressed in the interaction picture
            linear_operator(D_OP, V_ss, betas, lb, N, h); //set exp(D*h/2) as a function of omega = 2*pi*f
            
            // Wave form: 'c' = CW, 'g' = gaussian pulse, 's' = soliton, 't' = step funcion with a defined step //
            char m = 'c';
            int step = 1; //to modulate the CW in step steps

            #if !defined(NOISE)    
                input_field_T(u1, T, N, T0, P0, m, step); // signal without noise
                printf("Sin ruido\n");
            #endif     
            #if defined(NOISE)         
                input_field_T(u1, T, N, T0, P0, m, step);
                CC *h_noise = (CC *)malloc(nBytes);
                double SNR = 50; // Signal-to-Noise ratio
                noise_generator(h_noise, SNR, N, P0 );
                for (int j = 0; j < N; j++){
                    u1[j].x = u1[j].x + h_noise[j].x;
                    u1[j].y = u1[j].y + h_noise[j].y;
                }
                free(h_noise);
                printf("Con ruido\n");
            #endif
                
            #if defined(SELFSTEEPENING)
                printf("Con self-stepeening\n");
            #endif
            #if !defined(SELFSTEEPENING)
                printf("Sin self-stepeening\n");
            #endif
            #if defined(RAMAN)
                printf("Con Raman\n\n");
            #endif
            #if !defined(RAMAN)
                printf("Sin Raman\n\n");
            #endif                

            // Define dV, d_self_st, d_D_OP, du1 and d_u1_W, vectos on GPU //
            CHECK(cudaMalloc((void **)&d_self_st, nBytes));             CHECK(cudaMemset(d_self_st, 0, nBytes));
            CHECK(cudaMemcpy(d_self_st, self_st, nBytes, cudaMemcpyHostToDevice));                        
            CHECK(cudaMalloc((void **)&d_D_OP, nBytes));                CHECK(cudaMemset(d_D_OP, 0, nBytes));
            CHECK(cudaMemcpy(d_D_OP, D_OP, nBytes, cudaMemcpyHostToDevice));
            CHECK(cudaMalloc((void **)&d_u1_W, nBytes));                CHECK(cudaMemset(d_u1_W, 0, nBytes));
            CHECK(cudaMalloc((void **)&d_u1, nBytes));                  CHECK(cudaMemset(d_u1, 0, nBytes));
            CHECK(cudaMemcpy(d_u1, u1, nBytes, cudaMemcpyHostToDevice));    

            cufftExecZ2Z(plan_1, (CC *)d_u1, (CC *)d_u1_W, CUFFT_INVERSE); // d_u1
            CHECK(cudaDeviceSynchronize());
            CUFFTscale<<<grid,block>>>(d_u1_W, N, N);
            CHECK(cudaDeviceSynchronize());

            // Allocating memory on GPU //
            CHECK(cudaMalloc((void **)&d_u_ip, nBytes));    CHECK(cudaMemset(d_u_ip, 0, nBytes));
            CHECK(cudaMalloc((void **)&d_alpha1, nBytes));  CHECK(cudaMemset(d_alpha1, 0, nBytes));
            CHECK(cudaMalloc((void **)&d_alpha2, nBytes));  CHECK(cudaMemset(d_alpha2, 0, nBytes));
            CHECK(cudaMalloc((void **)&d_alpha3, nBytes));  CHECK(cudaMemset(d_alpha3, 0, nBytes));
            CHECK(cudaMalloc((void **)&d_alpha4, nBytes));  CHECK(cudaMemset(d_alpha4, 0, nBytes));
            CHECK(cudaMalloc((void **)&d_u2_W, nBytes));    CHECK(cudaMemset(d_u2_W, 0, nBytes));
            CHECK(cudaMalloc((void **)&d_u2, nBytes));      CHECK(cudaMemset(d_u2, 0, nBytes));
            CHECK(cudaMalloc((void **)&d_u3_W, nBytes));    CHECK(cudaMemset(d_u3_W, 0, nBytes));
            CHECK(cudaMalloc((void **)&d_u3, nBytes));      CHECK(cudaMemset(d_u3, 0, nBytes));
            CHECK(cudaMalloc((void **)&d_u4_W, nBytes));    CHECK(cudaMemset(d_u4_W, 0, nBytes));
            CHECK(cudaMalloc((void **)&d_u4, nBytes));      CHECK(cudaMemset(d_u4, 0, nBytes));
            // Aux device vectors //
            CC *d_aux;  CHECK(cudaMalloc((void **)&d_aux, nBytes));   CHECK(cudaMemset(d_aux, 0, nBytes));
            CC *d_aux1; CHECK(cudaMalloc((void **)&d_aux1, nBytes));  CHECK(cudaMemset(d_aux1, 0, nBytes));
            CC *d_aux2; CHECK(cudaMalloc((void **)&d_aux2, nBytes));  CHECK(cudaMemset(d_aux2, 0, nBytes));
            CC *d_aux3; CHECK(cudaMalloc((void **)&d_aux3, nBytes));  CHECK(cudaMemset(d_aux3, 0, nBytes));

            // Device vectors for COMPUTE_TFN //
            CC *d_op1, *d_op1_W, *d_op2, *d_op2_W, *d_op3, *d_op3_W, *d_op4_W;
            CHECK(cudaMalloc((void **)&d_op1, nBytes));               CHECK(cudaMemset(d_op1, 0, nBytes));
            CHECK(cudaMalloc((void **)&d_op1_W, nBytes));             CHECK(cudaMemset(d_op1_W, 0, nBytes));
            CHECK(cudaMalloc((void **)&d_op2, nBytes));               CHECK(cudaMemset(d_op2, 0, nBytes));
            CHECK(cudaMalloc((void **)&d_op2_W, nBytes));             CHECK(cudaMemset(d_op2_W, 0, nBytes));
            CHECK(cudaMalloc((void **)&d_op3, nBytes));               CHECK(cudaMemset(d_op3, 0, nBytes));
            CHECK(cudaMalloc((void **)&d_op3_W, nBytes));             CHECK(cudaMemset(d_op3_W, 0, nBytes));
            CHECK(cudaMalloc((void **)&d_op4_W, nBytes));             CHECK(cudaMemset(d_op4_W, 0, nBytes));

            sprintf(filename1, "outputTreal_%.2f_%d.txt", factor, nr);
            sprintf(filename2, "outputTimag_%.2f_%d.txt", factor, nr);
            sprintf(filename3, "outputWreal_%.2f_%d.txt", factor, nr);
            sprintf(filename4, "outputWimag_%.2f_%d.txt", factor, nr);
            sprintf(filename5, "Z.txt");
            FILE *cincor;
            FILE *cincoi;
            FILE *seisr;
            FILE *seisi;
            FILE *zvec;
            cincor = fopen(filename1, "w+");
            cincoi = fopen(filename2, "w+");
            seisr = fopen(filename3, "w+");
            seisi = fopen(filename4, "w+");
            zvec = fopen(filename5, "w+");

            float num;
            printf("Starting main loop on CPU & GPU...\n");
            double iStart = seconds();

            // START MAIN LOOP //
            for (int s = 0; s < steps_z; s++){
                cpx_prod_GPU<<<grid,block>>>( d_D_OP, d_u1_W, d_u_ip, N ); // d_u_ip = d_D_OP * d_u1_W.  
                CHECK(cudaDeviceSynchronize());                 
                COMPUTE_TFN( d_alpha1, d_u1, d_u1_W, d_hR_W, d_self_st, N, fr, nBytes, d_op1, d_op1_W, d_op2, d_op2_W, d_op3, d_op3_W, d_op4_W ); // d_alpha1.
                cpx_prod_GPU<<<grid,block>>>( d_alpha1, d_D_OP, d_aux ,N );
                CHECK(cudaDeviceSynchronize());
                equal<<<grid,block>>>( d_alpha1, d_aux, N ); // d_alpha1 = d_D_OP * d_alpha1.
                CHECK(cudaDeviceSynchronize());                     
                lineal<<<grid,block>>>( d_u_ip, d_alpha1, d_u2_W, N, h/2 ); // d_u2_W = d_u_ip + h/2*d_alpha1
                CHECK(cudaDeviceSynchronize());                 
                cufftExecZ2Z( plan_1, (CC *)d_u2_W, (CC *)d_u2, CUFFT_FORWARD ); // d_u2.
                CHECK(cudaDeviceSynchronize());        
                COMPUTE_TFN( d_alpha2, d_u2, d_u2_W, d_hR_W, d_self_st, N, fr, nBytes, d_op1, d_op1_W, d_op2, d_op2_W, d_op3, d_op3_W, d_op4_W ); // d_alpha2
                lineal<<<grid,block>>>( d_u_ip, d_alpha2, d_u3_W, N, h/2 );  
                CHECK(cudaDeviceSynchronize());
                cufftExecZ2Z( plan_1, (CC *)d_u3_W, (CC *)d_u3, CUFFT_FORWARD );
                CHECK(cudaDeviceSynchronize());
                COMPUTE_TFN( d_alpha3, d_u3, d_u3_W, d_hR_W, d_self_st, N, fr, nBytes, d_op1, d_op1_W, d_op2, d_op2_W, d_op3, d_op3_W, d_op4_W ); // d_alpha3.
                lineal<<<grid,block>>>(d_u_ip, d_alpha3, d_aux1, N, h); // d_u_ip + h*d_alpha3.
                CHECK(cudaDeviceSynchronize());                
                cpx_prod_GPU<<<grid,block>>>( d_D_OP, d_aux1, d_u4_W , N ); // d_u4_W.
                CHECK(cudaDeviceSynchronize());
                cufftExecZ2Z(plan_1, (CC *)d_u4_W, (CC *)d_u4, CUFFT_FORWARD);
                CHECK(cudaDeviceSynchronize());
                COMPUTE_TFN( d_alpha4, d_u4, d_u4_W, d_hR_W, d_self_st, N, fr, nBytes, d_op1, d_op1_W, d_op2, d_op2_W, d_op3, d_op3_W, d_op4_W ); // d_alpha4.
                final<<<grid,block>>>(d_u_ip, d_alpha1, d_alpha2, d_alpha3, d_aux2, h, N); 
                CHECK(cudaDeviceSynchronize());
                cpx_prod_GPU<<<grid,block>>>( d_D_OP, d_aux2, d_aux3 , N ); // d_aux3 = d_D_OP * d_aux2.
                CHECK(cudaDeviceSynchronize());                 
                lineal<<<grid,block>>>(d_aux3, d_alpha4, d_u1_W, N, h/6); // d_u1_W = d_aux3 + h/6*d_alpha4.
                CHECK(cudaDeviceSynchronize());
                cufftExecZ2Z(plan_1, (CC *)d_u1_W, (CC *)d_u1, CUFFT_FORWARD); // d_u1.
                CHECK(cudaDeviceSynchronize());
                
//                 if(s == 0 || s%5000==0){
                if(s == 0 || s == steps_z-1){                
                    //printf("\nVALOR = %d\n",s );
                    CHECK(cudaMemcpy(u1, d_u1, nBytes, cudaMemcpyDeviceToHost));
                    CHECK(cudaMemcpy(u1_W, d_u1_W, nBytes, cudaMemcpyDeviceToHost));
                    for ( i = 0; i < N; i++ ){
                        fprintf(cincor, "%15.20f\t", u1[i].x);// writing data into file
                        fprintf(cincoi, "%15.20f\t", u1[i].y);// writing data into file
                        fprintf(seisr, "%15.20f\t", u1_W[i].x);// writing data into file
                        fprintf(seisi, "%15.20f\t", u1_W[i].y);// writing data into file
                    }    
                    fprintf(zvec, "%15.20f\t", s*h/flength);// writing data into file
                    fprintf(cincor, "\n");// writing data into file
                    fprintf(cincoi, "\n");// writing data into file
                    fprintf(seisr,  "\n");// writing data into file
                    fprintf(seisi,  "\n");// writing data into file
                }

                num = (float) s*200/(steps_z-1);
                if ( (abs(num - 1.00) <= 0.001) || (abs(num - 10.00) <= 0.001) || (abs(num - 25.00) <= 0.001) || (abs(num - 50.00) <= 0.001) || (abs(num - 75.00) <= 0.001) || (abs(num - 90.00) <= 0.001) || (abs(num - 100.00) <= 0.001) ){ 
                    printf("%.2f %% completed...\n", num);
                }          
            }
            fclose(cincor);//closing file	
            fclose(cincoi);//closing file
            fclose(seisr);//closing file
            fclose(seisi);//closing file                        
            fclose(zvec);//closing file 
            
            double iElaps = seconds() - iStart;
            if(iElaps>60){
                printf("...time elapsed %.3f min in realisation #%d\n\n", iElaps/60, nr+1);
            }
            else{
                printf("...time elapsed %.3f sec in realisation #%d\n\n", iElaps, nr+1);
            }
                                    
            // Deallocating memory and destroying plans //
            free(u1); free(u1_W);      free(D_OP);
            free(self_st);             free(hR);
            free(hR_W);                free(V_ss);                
            free(T); free(TT);         free(V); free(Z);
            CHECK(cudaFree(d_D_OP));   CHECK(cudaFree(d_self_st));
            CHECK(cudaFree(d_u_ip));   CHECK(cudaFree(d_alpha1));
            CHECK(cudaFree(d_alpha2)); CHECK(cudaFree(d_alpha3));
            CHECK(cudaFree(d_u1));     CHECK(cudaFree(d_u1_W));
            CHECK(cudaFree(d_u2));     CHECK(cudaFree(d_u2_W));
            CHECK(cudaFree(d_u3));     CHECK(cudaFree(d_u3_W));
            CHECK(cudaFree(d_u4));     CHECK(cudaFree(d_u4_W));
            CHECK(cudaFree(d_aux));    CHECK(cudaFree(d_aux1));
            CHECK(cudaFree(d_aux2));   CHECK(cudaFree(d_aux3));
            CHECK(cudaFree(d_op1_W));  CHECK(cudaFree(d_op2_W));
            CHECK(cudaFree(d_op1));    CHECK(cudaFree(d_op2));
            CHECK(cudaFree(d_op3));    CHECK(cudaFree(d_op3_W));
            CHECK(cudaFree(d_op4_W));
            
            // Destroy CUFFT context //
            cufftDestroy(plan_1);            
            cudaDeviceReset();
        }
    }
    return 0;
}
