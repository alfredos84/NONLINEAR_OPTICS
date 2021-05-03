#!/bin/bash

rm *.txt

nvcc SSFM.cu -DSELFSTEEPENING -DRAMAN -DNOISE functions.cu --gpu-architecture=sm_30 -lcufftw -lcufft -lcurand -o SSFMcu

realizaciones=50
echo "Se ejecutarán $realizaciones realizaciones"

./SSFMcu $realizaciones

carpeta=simulaciones

mkdir carpeta
mv *.txt carpeta
