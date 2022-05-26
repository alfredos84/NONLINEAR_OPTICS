#ifndef _OPERATORSCUH
#define _OPERATORSCUH



/**
 * This file contains the operators overloading required for several 
 * complex vectors operatios in both CPU and GPU.
 * 
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


//////////////////////////////
// Returns C = A+B
cVech_t  operator+(cVech_t &lhs, const cVech_t &rhs) {
    thrust::transform(rhs.begin(), rhs.end(),
                      lhs.begin(), lhs.begin(), ComplexSum());
    return lhs;
}


cVecd_t operator+(cVecd_t &lhs, const cVecd_t &rhs) {
    thrust::transform(rhs.begin(), rhs.end(),
                      lhs.begin(), lhs.begin(), ComplexSum());
    return lhs;
}
//////////////////////////////




//////////////////////////////
// Returns B += A
cVech_t  operator+=(cVech_t &lhs, const cVech_t &rhs) {
    thrust::transform(rhs.begin(), rhs.end(),
                      lhs.begin(), lhs.begin(), ComplexSum());
    return lhs;
}


cVecd_t operator+=(cVecd_t &lhs, const cVecd_t &rhs) {
    thrust::transform(rhs.begin(), rhs.end(),
                      lhs.begin(), lhs.begin(), ComplexSum());
    return lhs;
}
//////////////////////////////




//////////////////////////////
// Returns C = A-B
cVech_t  operator-(cVech_t &lhs, const cVech_t &rhs) {
    thrust::transform(rhs.begin(), rhs.end(),
                      lhs.begin(), lhs.begin(), ComplexSubstract());
    return lhs;
}


cVecd_t operator-(cVecd_t &lhs, const cVecd_t &rhs) {
    thrust::transform(rhs.begin(), rhs.end(),
                      lhs.begin(), lhs.begin(), ComplexSubstract());
    return lhs;
}
//////////////////////////////




//////////////////////////////
// Returns B -= A
cVech_t  operator-=(cVech_t &lhs, const cVech_t &rhs) {
    thrust::transform(rhs.begin(), rhs.end(),
                      lhs.begin(), lhs.begin(), ComplexSubstract());
    return lhs;
}


cVecd_t operator-=(cVecd_t &lhs, const cVecd_t &rhs) {
    thrust::transform(rhs.begin(), rhs.end(),
                      lhs.begin(), lhs.begin(), ComplexSubstract());
    return lhs;
}
//////////////////////////////




//////////////////////////////
// Returns C = A*B
cVech_t  operator*(cVech_t &lhs, const cVech_t &rhs) {
    thrust::transform(rhs.begin(), rhs.end(),
                      lhs.begin(), lhs.begin(), ComplexMult());
    return lhs;
}


cVecd_t operator*(cVecd_t &lhs, const cVecd_t &rhs) {
    thrust::transform(rhs.begin(), rhs.end(),
                      lhs.begin(), lhs.begin(), ComplexMult());
    return lhs;
}
//////////////////////////////




//////////////////////////////
// Returns B *= A
cVech_t  operator*=(cVech_t &lhs, const cVech_t &rhs) {
    thrust::transform(rhs.begin(), rhs.end(),
                      lhs.begin(), lhs.begin(), ComplexMult());
    return lhs;
}


cVecd_t operator*=(cVecd_t &lhs, const cVecd_t &rhs) {
    thrust::transform(rhs.begin(), rhs.end(),
                      lhs.begin(), lhs.begin(), ComplexMult());
    return lhs;
}
//////////////////////////////




//////////////////////////////
// Returns A *= c -> A = c*A, with c as a real constant
cVech_t  operator*=(cVech_t &rhs, const typefl_t realscalar) {
    thrust::transform(rhs.begin(), rhs.end(),
                      rhs.begin(), RealScale(realscalar));
    return rhs;
}


cVecd_t  operator*=(cVecd_t &rhs, const typefl_t realscalar) {
    thrust::transform(rhs.begin(), rhs.end(),
                      rhs.begin(), RealScale(realscalar));
    return rhs;
}
//////////////////////////////




//////////////////////////////
// Returns A *= c -> A = c*A, with c as a complex constant
cVech_t  operator*=(cVech_t &rhs, const CC_t complexscalar) {
    thrust::transform(rhs.begin(), rhs.end(),
                      rhs.begin(), ComplexScale(complexscalar));
    return rhs;
}


cVecd_t  operator*=(cVecd_t &rhs, const CC_t complexscalar) {
    thrust::transform(rhs.begin(), rhs.end(),
                      rhs.begin(), ComplexScale(complexscalar));
    return rhs;
}
//////////////////////////////



#endif // -> #ifdef _OPERATORSCUH
