#ifndef _SAVEFILESCUH
#define _SAVEFILESCUH


/**
 * This file contains two funtions used to save real and complex vectors
 * 
 * SaveFileVectorReal:    takes a real vector a returns one-column .dat file
 * 
 * SaveFileVectorComplex: takes a complex vector a returns two-column .dat files (real and imag)
 * 
 * */



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


// Complex data type
using CC_t = cuFloatComplex;
using typefl_t = float;

using rVech_t = thrust::host_vector<typefl_t>;
using rVecd_t = thrust::device_vector<typefl_t>;
using cVech_t = thrust::host_vector<CC_t>;
using cVecd_t = thrust::device_vector<CC_t>;	


void SaveFileVectorReal (rVech_t &Vector, std::string Filename){
	std::ofstream myfile;
	myfile.open(Filename);
	for (int iy = 0; iy < Vector.size(); iy++)
		myfile << std::setprecision(20) << Vector[iy] << "\n";
	myfile.close();
	
	return;
}


void SaveFileVectorComplex (cVech_t &Vector, std::string Filename){
	std::ofstream myfile;
	std::string extension_r = "_r.dat", extension_i = "_i.dat";
	myfile.open(Filename+extension_r);
	for (int iy = 0; iy < Vector.size(); iy++)
		myfile << std::setprecision(20) << Vector[iy].x << "\n";
	myfile.close();
	myfile.open(Filename+extension_i);
	for (int iy = 0; iy < Vector.size(); iy++)
	    myfile << std::setprecision(20) << Vector[iy].y << "\n";
	myfile.close();
	
	return ;
}

#endif // -> #ifdef _SAVEFILESCUH
