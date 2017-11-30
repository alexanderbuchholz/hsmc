#!/bin/bash
#
# state_space_sim.sbatch
#
#SBATCH -p serial_requeue # Partition
#SBATCH -n 8              # one CPU
#SBATCH -N 1              # on one node
#SBATCH -t 7-0:00         # Running time of 2 hours
#SBATCH --mem 4000        # Memory request
#SBATCH -o hostname_%j.out # File to which STDOUT will be written
#SBATCH -e hostname_%j.err # File to which STDERR will be written
#SBATCH --array=1-8

module load python/2.7.11-fasrc01
source activate my_root
module load R/3.2.0-fasrc01
export R_LIBS_USER=$HOME/apps/R:$R_LIBS_USER




#srun R CMD BATCH --no-save $input out/$input.$SLURM_ARRAY_TASK_ID.out
#python test_batch.py $SLURM_ARRAY_TASK_ID

python run_simulation.py $SLURM_ARRAY_TASK_ID
