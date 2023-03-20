#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=moco_common
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=100GB
#SBATCH --output=/home/users/krave/logs/mocologs/%x.%j.out
#SBATCH --open-mode=append
#SBATCH --partition=trc

# Uses command-line (basic, non-python) ANTs to motion correct
# All params, file lists etc handled in two_photon_analysis/scripts/batch_common_moco.py
# USAGE: sbatch medulla_moco_to_common.sh

date

ml python/3.6 ants-base

# Run moco python script
python3 -u /home/users/krave/github_repos/two_photon_analysis/scripts/batch_common_moco.py
