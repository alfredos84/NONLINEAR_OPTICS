#!/bin/bash

# clear
rm *.dat
rm *.txt
rm cuOPO

COMPILER="nvcc"
MAIN_FILE="cuOPO.cu"

REG="cw"                             # Set regime (ns or cw)
EQS="2"                              # Set number of equations to solve (2 or 3)

EXEC_FILE="cuOPO"

printf "Compiling...   "
$COMPILER $MAIN_FILE -DCW_OPO --gpu-architecture=sm_75 -lcufftw -lcufft -o $EXEC_FILE
# $COMPILER $MAIN_FILE -DCW_OPO -DTHREE_EQS --gpu-architecture=sm_75 -lcufftw -lcufft -o $EXEC_FILE
printf "   OK!\n\n"


EXEC_FILE="cuOPO"
ARG1=1                               # Set 1 to save time and frequency vectors                   (ARG1)
ARG2=14                              # Grid size = 2^(ARG2)                                       (ARG2)
ARG3=150                             # Number of crystal partitions                               (ARG3)
ARG4=(5)                             # Lcav = ARG5*Lcr (cavity length in terms of crystal length) (ARG4)
RE=(98)                              # Reflectivity at signal wl (in percent %)                   (ARG5)
D=(0)                                # Net cavity detuning (in rad)                               (ARG6)
GD=(0)                             # GDD compensation (in percent %)                            (ARG7)
ARG8=10000                           # Number of round trips per simulation                       (ARG8)
NN=(4)                             # N = Power/Pth                                              (ARG9)
UPM=(1)                              # Using phase modulator: OFF/ON = 0/1                        (ARG10)
MD=(0.8)                             # EOM: β (modulation depth in π rads)                        (ARG11)
FM=(2)                               # δf = FSR - fpm [MHz] Frequency detuning for EOM            (ARG12)
TD=(0)                               # TOD compensation (in percent %)                            (ARG13)
SPM=(0)                              # Using self-phase modulation: OFF/ON = 0/1                  (ARG14)
# REAL=(1 2)
# for (( k=0; k<${#REAL[@]}; k++ ))
# do

for (( u=0; u<${#UPM[@]}; u++ ))
do  
	for (( n=0; n<${#NN[@]}; n++ ))
	do  
		for (( m=0; m<${#MD[@]}; m++ ))
		do  
			for (( f=0; f<${#FM[@]}; f++ ))
			do
				for (( i=0; i<${#D[@]}; i++ ))
				do  
					for (( r=0; r<${#RE[@]}; r++ ))
					do	
						for (( g=0; g<${#GD[@]}; g++ ))
						do
							for (( t=0; t<${#TD[@]}; t++ ))
							do
								U=${UPM[$u]} 
								printf "\nUsing phase mod  = ${UPM}\n" 
								
								N=${NN[$n]} 
								printf "\nPower/Pth        = ${N} \n" 
								
								GDD=${GD[$g]} 
								printf "\nGDD compensation = ${GDD}%%\n"
								
								TOD=${TD[$t]}
								printf "\nTOD compensation = ${TOD}%%\n"
								
								R=${RE[$r]} 
								printf "\nLcav/Lc          = ${ARG4} \n"
								
								DELTAS=${D[$i]}
								printf "\ndelta            = ${DELTAS}\n"
								
								MODDEP=${MD[$m]}
								printf "\nModul depth      = ${MODDEP}\n"
								
								FREQMOD=${FM[$f]}
								printf "\nFreqMod          = ${FREQMOD} \n"
								
# 								REA=${REAL[$k]}
								
								printf "\nMaking directory...\n"
								FOLDER="${REG}_${EQS}eqs_PPLN_beta_${MODDEP}_N_${N}_GDD_${GDD}"
								FILE="${REG}_${EQS}eqs_PPLN_beta_${MODDEP}_N_${N}_GDD_${GDD}.txt"
								
								printf "Bash execution and writing output file...\n\n"
								./$EXEC_FILE $ARG1 $ARG2 $ARG3 $ARG4 $R $DELTAS $GDD $ARG8 $N $U $MODDEP $FREQMOD $TOD $SPM | tee -a $FILE
						# 			
								printf "Bash finished!!\n\n" 
								mkdir $FOLDER
								mv *.dat $FOLDER"/"
								mv *.txt $FOLDER"/"
							done
						done
					done
				done
			done
		done
	done
done

# done
