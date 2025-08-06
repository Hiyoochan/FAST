#!/bin/bash
#SBATCH -J CTunet       # job name, optional
#SBATCH -N 1          # number of computing node
#SBATCH -c 6          # number of cpus, for multi-thread programs
#SBATCH --gres=gpu:1  # number of gpus allocated on each node
#SBATCH -t 14400
#SBATCH -w inspur0
#SBATCH --qos=super

train_flemme --config /home/NA/workspace/flASAP/jobs/CT/target/MSD/unet/train_local_MSDliver.yaml
train_flemme --config /home/NA/workspace/flASAP/jobs/CT/target/MSD/unet/train_local_MSDpancreas.yaml
train_flemme --config /home/NA/workspace/flASAP/jobs/CT/target/MSD/unet/train_local_MSDspleen.yaml
train_flemme --config /home/NA/workspace/flASAP/jobs/CT/target/MSD/unet/train_local_MSDvessel.yaml

train_flemme --config /home/NA/workspace/flASAP/jobs/CT/target/THOR/unet/train_local_THORaorta.yaml
train_flemme --config /home/NA/workspace/flASAP/jobs/CT/target/THOR/unet/train_local_THORheart.yaml
train_flemme --config /home/NA/workspace/flASAP/jobs/CT/target/THOR/unet/train_local_THORtrachea.yaml
train_flemme --config /home/NA/workspace/flASAP/jobs/CT/target/THOR/unet/train_local_THOResophagus.yaml




