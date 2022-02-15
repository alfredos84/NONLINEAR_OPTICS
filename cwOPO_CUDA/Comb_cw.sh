#!/bin/bash

clear
rm *.dat
rm *.txt
rm Combcw

COMPILER="nvcc"
MAIN_FILE="Combcw.cu"

EXEC_FILE="Combcw"
ARG1=1                        # Set 1 to save time and frequency vectors (ARG1)
ARG2=14                       # Exponent of N. N=2^(ARG2)
ARG3=150                      # Number of crystal partitions (ARG3)
CAV=(5)                       # How much times is the cavity to the crystal length (ARG4)
R=98                          # Net cavity reflectivity (ARG5)
D=(0)                         # Net cavity detuning (ARG6)
GD=(0)                        # GDD compensation (ARG7)
ARG8=10000                     # Number of round trips
P=2                           # Power [W]


for (( i=0; i<${#D[@]}; i++ ))
do  
	for (( j=0; j<${#CAV[@]}; j++ ))
	do	
		for (( k=0; k<${#GD[@]}; k++ ))
		do	
			GDD=${GD[$k]} 
			printf "\nGDD compensation = ${GDD}%%\n" 
			ARG4=${CAV[$j]} 
			printf "\nLcav/Lc          = ${ARG4} \n" 
			DELTAS=${D[$i]}
			printf "\ndelta            = ${DELTAS}\n"
			
			printf "\nMaking directory...\n"
			FOLDER="cw_PPLN_R_0.${R}_cav_length_${ARG4}_delta_${DELTAS}_Power_${P}_W_GDD_${GDD}"
			FILE="cw_PPLN_R_0.${R}_cav_length_${ARG4}_delta_${DELTAS}_Power_${P}_W_GDD_${GDD}.txt"
			mkdir $FOLDER		
			printf "\nChecking if compilation is needed...\n"
			if [ -e $EXEC_FILE ]
			then
				printf "   File exists!\n"
			else
				printf "compiling...   "
	# 			$COMPILER $MAIN_FILE -DDOUBLEPRECISION -gencode=arch=compute_60,code=sm_60 -lcufftw -lcufft -lcurand -o $EXEC_FILE 
 				$COMPILER $MAIN_FILE --gpu-architecture=sm_60 -lcufftw -lcufft -o $EXEC_FILE		
				printf "   OK!\n\n"
			fi

			printf "Bash execution and writing output file...\n\n"
			./$EXEC_FILE $ARG1 $ARG2 $ARG3 $ARG4 $R $DELTAS $GDD $ARG8 $P | tee -a $FILE
	# 				./$EXEC_FILE #| tee -a $FILE
			printf "Bash finished!!\n\n" 
			mv *.dat $FOLDER"/"
			mv *.txt $FOLDER"/"
		done
	done
done
#     done
# done
