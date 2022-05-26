#!/bin/bash

clear
rm *.dat
rm *.txt
rm cwOPO

COMPILER="nvcc"
MAIN_FILE="cwOPO.cu"

EXEC_FILE="cwOPO"
ARG1=0                        # Set 1 to save time and frequency vectors (ARG1)
ARG2=14                       # Exponent of N. N=2^(ARG2)
ARG3=150                      # Number of crystal partitions (ARG3)
ARG4=(5)                      # How much times is the cavity to the crystal length (ARG4)
RE=(98)                       # Net cavity reflectivity (ARG5)
D=(0)                         # Net cavity detuning (ARG6)
GD=(100) 	                  # GDD compensation (ARG7)
ARG8=10000                    # Number of round trips (ARG8)
POWER=(400)                   # Power [mW] (ARG9)
UPM=(1)                       # Using phase modulator: OFF=0/ON=1 (ARG10)
MD=(20)	                  # Modulation depth (ARG11)
FM=(150)                      # δf = FSR - fm, frequency detuning modulation for EOM [MHz] (ARG12)

for (( u=0; u<${#UPM[@]}; u++ ))
do  
	for (( p=0; p<${#POWER[@]}; p++ ))
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
							U=${UPM[$u]} 
							printf "\nUsing phase mod  = ${UPM}\n" 
							
							P=${POWER[$p]} 
							printf "\nPower            = ${P} mW\n" 
							
							GDD=${GD[$g]} 
							printf "\nGDD compensation = ${GDD}%%\n" 
							R=${RE[$r]} 
							printf "\nLcav/Lc          = ${ARG4} \n" 
							DELTAS=${D[$i]}
							printf "\ndelta            = ${DELTAS}\n"
							
							MODDEP=${MD[$m]}
							printf "\nModul depth      = ${MODDEP}\n"
							
							FREQMOD=${FM[$f]}
							printf "\nFreqMod          = ${FREQMOD}\n"
							
							printf "\nMaking directory...\n"
# 							FOLDER="cw_PPLN_R_0.${R}_Realz_${p}"
# 							FILE="cw_PPLN_R_0.${R}_Realz_${p}.txt"
							FOLDER="cw_PPLN_R_0.${R}_delta_${DELTAS}_Power_${P}_W_GDD_${GDD}_UPM_${U}_ModDepth_${MODDEP}_FreqMod_${FREQMOD}"
							FILE="cw_PPLN_R_0.${R}_delta_${DELTAS}_Power_${P}_W_GDD_${GDD}_UPM_${U}_ModDepth_${MODDEP}_FreqMod_${FREQMOD}.txt"
							
							mkdir $FOLDER		
							printf "\nChecking if compilation is needed...\n"
							if [ -e $EXEC_FILE ]
							then
								printf "   File exists!\n"
							else
								printf "compiling...   "
								$COMPILER $MAIN_FILE --gpu-architecture=sm_75 -lcufftw -lcufft -o $EXEC_FILE		
								printf "   OK!\n\n"
							fi

							printf "Bash execution and writing output file...\n\n"
							./$EXEC_FILE $ARG1 $ARG2 $ARG3 $ARG4 $R $DELTAS $GDD $ARG8 $P $U $MODDEP $FREQMOD | tee -a $FILE
					# 				./$EXEC_FILE #| tee -a $FILE
							printf "Bash finished!!\n\n" 
							mv *.dat $FOLDER"/"
							mv *.txt $FOLDER"/"
						done
					done
				done
			done
		done
	done
done
