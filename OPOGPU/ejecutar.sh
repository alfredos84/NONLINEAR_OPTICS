#!/bin/bash

#rm *.dat
rm OPO

COMPILER="nvcc"
MAIN_FILE="OPO_base.cu"
FUNCTIONS="functions.cu"


EXEC_FILE="OPO"
ARG1=14 #exponent of N. N=2^(ARG1)
ARG2=100 #number of crystal partitions
ARG3=1024 #number of round trip. Set as a power of 2 
ARG4=20 #how much times is the cavity to the crystal length

FOLDER="PPLN_Lcav-$ARG4-xLcr"
printf "Checking if compilation is needed...\n"
if [ -e $EXEC_FILE ]
then
    printf "   File exists!\n"
else
    printf "compiling..."
	$COMPILER $MAIN_FILE $FUNCTIONS --gpu-architecture=sm_60 -lcufftw -lcufft -lcurand -o $EXEC_FILE
    printf "   OK!\n\n"
fi

mkdir $FOLDER
for DELTAS in {-90..0..10}  
do 

printf "Execution...\n"
printf "Procesando delta=$DELTAS ...\n\n"
./OPO $ARG1 $ARG2 $ARG3 $ARG4 $DELTAS
printf "   OK!\n\n"

done


#############################

#############################
### COMPRIMIR ARCHIVOS DE SALIDA

mv *.dat $FOLDER"/"
cp graficar.m $FOLDER"/"
#cd $FOLDER
#ls -a
#printf "Comprimiendo archivos...\n\n"

#mkdir "$newfolder"
#for DATs in *.dat
#do

#zip -m $files".zip" $DATs
#done
