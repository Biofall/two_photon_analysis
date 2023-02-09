#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=moco_common
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/users/mhturner/mocologs/%x.%j.out
#SBATCH --open-mode=append
#SBATCH --partition=trc

# Params: (1) base directory, (2) name of master brain for reference (3) name of mirror brain 4... optional registration params
# USAGE: sbatch medulla_moco_to_common.sh /oak/stanford/groups/trc/data/Max/ImagingData/Bruker/.../ TSeries-2022MMDD-001 TSeries-2022MMDD-002

date

directory=$1

brain_master=$2  # Brain to use to compute meanbrain to motion correct to (reference brain)
brain_mirror=$3  # Brain to motion correct

# Optional params
type_of_transform=${4:-"Rigid"}
output_format=${5:-"nii"}
total_sigma=${6:-'0'}
meanbrain_n_frames=${7:-"100"}
aff_metric=${8:-"mattes"}


ml python/3.6 py-ants/0.3.2_py36

# Make meanbrain from master
args="{\"directory\":\"$directory\",\"files\":\"$brain_master\",\"meanbrain_n_frames\":\"$meanbrain_n_frames\",\"logfile\":\"meanbrain_log.txt\"}"

echo $args
python3 -u /home/users/mhturner/brainsss/scripts/make_mean_brain.py $args

brain_master_base=${brain_master:0: -4}  # strip off the .nii
master_mean_name="${brain_master_base}_mean.nii"

# # Motion correction on mirror
echo $type_of_transform
echo $output_format
echo $meanbrain_n_frames
echo $total_sigma
echo $aff_metric

# Motion correct brain_mirror using meanbrain_target as reference 
args="{\"directory\":\"$directory\",\"brain_master\":\"$brain_mirror\",\"meanbrain_target\":\"$master_mean_name\",\"total_sigma\":\"$total_sigma\",\"aff_metric\":\"$aff_metric\","\
"\"type_of_transform\":\"$type_of_transform\",\"output_format\":\"$output_format\",\"aff_metric\":\"$aff_metric\"}"

python3 -u /home/users/mhturner/brainsss/scripts/motion_correction.py $args