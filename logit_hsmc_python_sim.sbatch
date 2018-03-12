#!/bin/bash
#
# state_space_sim.sbatch
#
#SBATCH -p serial_requeue # Partition serial_requeue
#SBATCH -n 8              # one CPU
#SBATCH -N 1              # on one node
#SBATCH -t 2-0:00         # Running time of 2 hours
#SBATCH --mem 8000        # Memory request
#SBATCH -o info_logit_probit_%j.out # File to which STDOUT will be written
#SBATCH -e info_logit_probit_%j.err # File to which STDERR will be written
#SBATCH --array=1-40

module load python/2.7.11-fasrc01
source activate my_root
module load R/3.2.0-fasrc01
export R_LIBS_USER=$HOME/apps/R:$R_LIBS_USER




#srun R CMD BATCH --no-save $input out/$input.$SLURM_ARRAY_TASK_ID.out
#python test_batch.py $SLURM_ARRAY_TASK_ID

python run_simulations_server_logit.py $SLURM_ARRAY_TASK_ID

