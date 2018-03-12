#!/bin/bash
declare -a range
range=(1 2 3 4 5 6)

for i in "${range[@]}"
do
    python run_simulations_server_normal.py $i

done 
# first mc version
#python main.py --epochs $numepochs --l-samples 1
#python main.py --epochs $numepochs --l-samples 5
#python main.py --epochs $numepochs --l-samples 10
#python main.py --epochs $numepochs --l-samples 20
#python main.py --epochs $numepochs --l-samples 50

# second rqmc version
#python main.py --epochs $numepochs --rqmc True --l-samples 1
#python main.py --epochs $numepochs --rqmc True --l-samples 5
#python main.py --epochs $numepochs --rqmc True --l-samples 10
#python main.py --epochs $numepochs --rqmc True --l-samples 20
#python main.py --epochs $numepochs --rqmc True --l-samples 50

