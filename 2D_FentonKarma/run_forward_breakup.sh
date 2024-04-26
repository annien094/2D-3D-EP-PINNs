#!/bin/bash
#PBS -l select=1:ncpus=28:mem=240gb:ngpus=1:gpu_type=RTX6000
#PBS -l walltime=15:00:00
#PBS -m ae
#PBS -N FK_2D_breakup_forward
#PBS -o ./
#PBS -e ./


#Modules!/Applications
module load anaconda3/personal
source activate /rds/general/user/ch720/home/anaconda3/envs/PINNs
ulimit -s unlimited


cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR
python main_2D_FK_spiral_breakup.py -f "FKbreakup.mat" -m "2DFK_breakup" -ep 150000 -rg 3

