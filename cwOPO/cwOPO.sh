#!/bin/bash

clear
rm *.dat
rm *.txt
rm cwOPO

COMPILER="nvcc"
MAIN_FILE="cwOPO.cu"

EXEC_FILE="cwOPO"


$COMPILER $MAIN_FILE --gpu-architecture=sm_75 -lcufftw -lcufft -o $EXEC_FILE
printf "   OK!\n\n"
printf "Bash execution and writing output file...\n\n"

ARG1=0                               # Set 1 to save time and frequency vectors (ARG1)
ARG2=14                              # Exponent of N. N=2^(ARG2)
ARG3=150                             # Number of crystal partitions (ARG3)
ARG4=(5)                             # How much times is the cavity to the crystal length (ARG4)
RE=(98)                              # Net cavity reflectivity (ARG5)
D=(0)                                # Net cavity detuning (ARG6)
GD=(100)                             # GDD compensation (ARG7)
ARG8=10000                           # Number of round trips (ARG8)
NN=(4)                               # Power/Pth (ARG9)
UPM=(1)                              # Using phase modulator: OFF=0/ON=1 (ARG10)
MD=(0.8)                             # Modulation depth (ARG11)
FM=(1)                             # δf = FSR - fpm [MHz] Frequency modulation for EOM (ARG12)
TD=(0)                               # TOD compensation (ARG13) 

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
								
								DELTAS=${D[$i]}
								printf "\ndelta            = ${DELTAS}\n"
								
								MODDEP=${MD[$m]}
								printf "\nModul depth      = ${MODDEP}\n"
								
								FREQMOD=${FM[$f]}
								printf "\nFreqMod          = ${FREQMOD} MHz\n"
								
								printf "\nMaking directory...\n"

								FOLDER="MgOPPLN_N_${N}_beta_${MODDEP}_df_LB_${FREQMOD}"
								FILE="MgOPPLN_N_${N}_beta_${MODDEP}_df_LB_${FREQMOD}.txt"
										
								./$EXEC_FILE $ARG1 $ARG2 $ARG3 $ARG4 $R $DELTAS $GDD $ARG8 $N $U $MODDEP $FREQMOD $TOD | tee -a $FILE
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
