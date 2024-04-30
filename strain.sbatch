#!/bin/bash
#SBATCH -JDLTRAIN                    # Job name
#SBATCH -N1 --ntasks-per-node=1                 # Number of nodes and cores per node required
#SBATCH --mem-per-cpu=16G                        # Memory per core
#SBATCH -t1:00:00                                    # Duration of the job (Ex: 15 mins)
#SBATCH -oReport-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=tkwok7@gatech.edu        # E-mail address for notifications
cd $SLURM_SUBMIT_DIR                            # Change to worki`ng directory

module load anaconda3                           # Load module dependencies
conda activate DL_ASLR_ENV
srun python train_model.py                             # Example Process
