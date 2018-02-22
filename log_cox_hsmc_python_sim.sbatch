#!/bin/bash
#
# state_space_sim.sbatch
#
#SBATCH -p serial_requeue # Partition serial_requeue
#SBATCH -n 8              # one CPU
#SBATCH -N 1              # on one node
#SBATCH -t 2-0:00         # Running time of 2 hours
#SBATCH --mem 8000        # Memory request
#SBATCH -o log_cox_%j.out # File to which STDOUT will be written
#SBATCH -e log_cox_%j.err # File to which STDERR will be written
#SBATCH --array=1-4

module load python/2.7.11-fasrc01
source activate my_root
module load R/3.2.0-fasrc01
export R_LIBS_USER=$HOME/apps/R:$R_LIBS_USER




python run_simulations_server_log_cox.py $SLURM_ARRAY_TASK_ID
