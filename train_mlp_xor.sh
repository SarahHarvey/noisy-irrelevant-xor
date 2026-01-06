#!/bin/bash
#SBATCH --job-name=trainnetwork          # Job name
#SBATCH --output=.out/result_%j.out      # Standard output file (%j expands to jobId)
#SBATCH --error=.out/result_%j.err       # Standard error file
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=1                # Number of CPU cores per task
#SBATCH --mem=2GB                        # Memory per node
#SBATCH --time=0-01:00:00                # Time limit days-hrs:min:sec
#SBATCH --gres=gpu:1                     # Request 1 GPU
#SBATCH --partition=gpu                  # GPU partition 

N_HIDDEN=${1:-6}   # Default to 6 if not provided
LOAD_DATA=${2:-false}  # Default to false if not provided

module load python
srun hostname

# Build the command with optional flags
CMD="python train_mlp_xor.py --n_hidden $N_HIDDEN"

# Add --load_data flag if LOAD_DATA is true/True
if [ "$LOAD_DATA" = "true" ] || [ "$LOAD_DATA" = "True" ]; then
    CMD="$CMD --load_data"
fi

# Add --seed if $3 is provided
if [ -n "$3" ]; then
    CMD="$CMD --seed $3"
fi

srun $CMD