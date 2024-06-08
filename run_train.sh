#!/bin/bash
#SBATCH --job-name=sft_train
#SBATCH --partition=batch
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gpus=1

# only use the following on partition with GPUs

#SBATCH --output=sbatch/logs/%j.out

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

# Load environment
eval "$(conda shell.bash hook)"
conda activate cs336_alignment

which python3
python3 --version

nvidia-smi

# Print command
echo "Running the following command: python cs336_alignment/train.py"

# Run command
python cs336_alignment/train.py --run sft_train