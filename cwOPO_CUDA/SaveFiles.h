#pragma once

#include <iostream>
#include <string>
#include <random>
#include <fstream>
#include <iomanip>
#include <typeinfo>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <cuda_runtime.h>
#include <cufft.h>

#include "common.h"

// Complex data type
#ifdef DOUBLEPRECISION
    typedef cufftDoubleComplex CC;
    typedef double typefl;
#else
    typedef cufftComplex CC;
    typedef float typefl;
#endif


template <class T>
void SaveFileVectorReal (T *Vector, const int N, std::string Filename){
	std::ofstream myfile;
	myfile.open(Filename);
	for (int iy = 0; iy < N; iy++)
		myfile << std::setprecision(20) << Vector[iy] << "\n";
	myfile.close();
	return;
}


template <class T>
void SaveFileVectorComplex (T *Vector, const int N, std::string Filename){
	std::ofstream myfile;
	std::string extension_r = "_r.dat", extension_i = "_i.dat";
	myfile.open(Filename+extension_r);
	for (int iy = 0; iy < N; iy++)
		myfile << std::setprecision(20) << Vector[iy].x << "\n";
	myfile.close();
	myfile.open(Filename+extension_i);
	for (int iy = 0; iy < N; iy++)
	    myfile << std::setprecision(20) << Vector[iy].y << "\n";
	myfile.close();
}


template <class T>
void SaveFileMatrixReal (T *Vector, const int nx, const int ny, std::string Filename){
    std::ofstream myfile;
    std::string extension = ".dat";
    myfile.open(Filename+extension);
    for (int iy = 0; iy < ny; iy++){
		for (int ix = 0; ix < nx; ix++)
		    myfile << std::setprecision(20) << Vector[iy*nx+ix] << "\t";
		myfile << "\n"; 
    }
    myfile.close();

	return;
}


template <class T>
void SaveFileMatrixComplex (T *Vector, const int nx, const int ny, std::string Filename)
{
	std::ofstream myfile;
	std::string filenamer = "_r.dat", filenamei = "_i.dat";
	myfile.open(Filename+filenamer);
	for (int iy = 0; iy < ny; iy++){
		for (int ix = 0; ix < nx; ix++)
		    myfile << std::setprecision(20) << Vector[iy*nx+ix].x << "\t";
		myfile << "\n"; 
	}
	myfile.close();
	myfile.open(Filename+filenamei);
	for (int iy = 0; iy < ny; iy++){
		for (int ix = 0; ix < nx; ix++)
		    myfile << std::setprecision(20) << Vector[iy*nx+ix].y << "\t";
		myfile << "\n"; 
	}
	myfile.close();
	
	
    return;
}
