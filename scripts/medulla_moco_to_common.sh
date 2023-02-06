#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=moco_common
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/users/mhturner/mocologs/%x.%j.out
#SBATCH --open-mode=append
#SBATCH --partition=trc

# Params: (1) base directory, (2) name of master brain (3) name of mirror brain 4... optional registration params
# USAGE: sbatch medulla_moco_to_common.sh /oak/stanford/groups/trc/data/Max/ImagingData/Bruker/.../ TSeries-2022MMDD-00n

date

directory=$1

brain_master=$2
brain_mirror=$3

echo $directory
echo $brain_master
echo $brain_mirror

# Optional params
type_of_transform=${4:-"Rigid"}
output_format=${5:-"nii"}
total_sigma=${6:-'0'}
meanbrain_n_frames=${7:-"100"}
aff_metric=${8:-"mattes"}

echo $type_of_transform
echo $output_format
echo $meanbrain_n_frames
echo $total_sigma
echo $aff_metric

args="{\"directory\":\"$directory\",\"brain_master\":\"$brain_master\",\"brain_mirror\":\"$brain_mirror\",\"total_sigma\":\"$total_sigma\",\"aff_metric\":\"$aff_metric\","\
"\"type_of_transform\":\"$type_of_transform\",\"output_format\":\"$output_format\",\"meanbrain_n_frames\":\"$meanbrain_n_frames\",\"aff_metric\":\"$aff_metric\"}"

ml python/3.6 py-ants/0.3.2_py36

python3 -u /home/users/mhturner/brainsss/scripts/motion_correction.py $args
