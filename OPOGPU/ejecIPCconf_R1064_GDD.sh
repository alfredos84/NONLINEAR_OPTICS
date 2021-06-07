#!/bin/bash

clear
rm *.dat
rm OPO_GDD

COMPILER="nvcc"
MAIN_FILE="OPO_GDD.cu"
FUNCTIONS="functions.cu"

EXEC_FILE="OPO_GDD"
ARG1=13 #exponent of N. N=2^(ARG1)
ARG2=50 #number of crystal partitions
ARG3=400 #number of round trip. Set as a power of 2 
ARG4=4 #how much times is the cavity to the crystal length
ARG6=0 # set 1 to save time and frequency vectors
WL=1064


R=30
FOLDER="BBO_R_0.${R}_pump_at_${WL}_Si_GDD"
mkdir $FOLDER 
P=500000
for (( c=-35; c<=0; c+=5 ))
do
DELTAS=$c
for (( gdd=0; gdd<=100; gdd+=20 ))
do
	GDD=$gdd
	printf "\nChecking if compilation is needed...\n"
	if [ -e $EXEC_FILE ]
	then
		printf "   File exists!\n"
	else
		printf "compiling...   "
		$COMPILER $MAIN_FILE $FUNCTIONS --gpu-architecture=sm_60 -lcufftw -lcufft -lcurand -o $EXEC_FILE
		printf "   OK!\n\n"
	fi
	
	printf "Bash execution...\n\n"
	./$EXEC_FILE $ARG1 $ARG2 $ARG3 $ARG4 $DELTAS $ARG6 $WL $R $P $GDD
	printf "Bash finished!!\n\n"
	mv *.dat $FOLDER"/"
done
done


R=99
FOLDER="BBO_R_0.${R}_pump_at_${WL}_Si_GDD"
mkdir $FOLDER 
P=4000
for (( c=-5; c<=0; c++ ))
do
DELTAS=$c
for (( gdd=20; gdd<=100; gdd+=20 ))
do
	GDD=$gdd
	printf "\nChecking if compilation is needed...\n"
	if [ -e $EXEC_FILE ]
	then
		printf "   File exists!\n"
	else
		printf "compiling...   "
		$COMPILER $MAIN_FILE $FUNCTIONS --gpu-architecture=sm_60 -lcufftw -lcufft -lcurand -o $EXEC_FILE
		printf "   OK!\n\n"
	fi
	
	printf "Bash execution...\n\n"
	./$EXEC_FILE $ARG1 $ARG2 $ARG3 $ARG4 $DELTAS $ARG6 $WL $R $P $GDD
	printf "Bash finished!!\n\n"
	mv *.dat $FOLDER"/"
done
done
