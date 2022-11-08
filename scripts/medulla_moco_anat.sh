#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=medulla_moco
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --output=/home/users/krave/logs/mocologs/%x.%j.out
#SBATCH --open-mode=append
#SBATCH --partition=trc

date

directory=$1
brain_master=$2
brain_mirror=$3

echo $directory
echo $brain_master

# Optional params
type_of_transform=${4:-"SyN"}
output_format=${5:-"nii"}
meanbrain_n_frames=${6:-"100"}
echo $type_of_transform
echo $output_format
echo $meanbrain_n_frames

args="{\"directory\":\"$directory\",\"brain_master\":\"$brain_master\",\"brain_mirror\":\"$brain_mirror\",\"meanbrain_n_frames\":\"$meanbrain_n_frames\","\
"\"type_of_transform\":\"$type_of_transform\",\"output_format\":\"$output_format\"}"

#ml gcc/6.3.0 py-numpy/1.14.3_py36 py-pandas/0.23.0_py36 viz py-scikit-learn/0.19.1_py36 py-ants/0.3.2_py36
ml py-ants/0.3.2_py36

python3 -u /home/users/krave/github_repos/brainsss/scripts/motion_correction.py $args