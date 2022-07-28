#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=medulla_moco
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --output=/home/users/mhturner/two_photon_analysis/scripts/%x.%j.out
#SBATCH --open-mode=append

date

directory=$1
brain_master=$2

echo $directory
echo $brain_master

# Optional params
type_of_transform=${3:-"SyN"}
output_format=${4:-"nii"}
meanbrain_n_frames=${5:-"100"}
echo $type_of_transform
echo $output_format
echo $meanbrain_n_frames

args="{\"directory\":\"$directory\",\"brain_master\":\"$brain_master\",\"meanbrain_n_frames\":\"$meanbrain_n_frames\","\
"\"type_of_transform\":\"$type_of_transform\",\"output_format\":\"$output_format\"}"

ml py-ants/0.3.2_py36

python3 -u /home/users/mhturner/brainsss/scripts/motion_correction.py $args
