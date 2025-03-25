#!/bin/bash
#PBS -q gpu_dgx@pbs-m1.metacentrum.cz
#PBS -l walltime=30:0:0
#PBS -l select=1:ncpus=7:ngpus=1:mem=50gb
#PBS -N run_pretrain_exp4_one_src_token

cd /storage/brno2/home/ahajek/Spektro/MassGenie/clean_paper
source /storage/brno2/home/ahajek/miniconda3/bin/activate trainSpectus
echo $CONDA_PREFIX
./config_runners/run_pretrain_exp4_one_src_token.sh

exit

##### 1x A100 ######

##### 1x A40 ######
#PBS -q gpu@meta-pbs.metacentrum.cz
#PBS -l walltime=24:0:0
#PBS -l select=1:ncpus=8:ngpus=1:mem=50gb:scratch_local=400mb:cl_galdor=True
#PBS -N debug_train__GALDOR


##### DGXko ########
# nnn #PBS -q gpu_dgx@meta-pbs.metacentrum.cz
# nnn #PBS -l walltime=24:0:0
# nnn #PBS -l select=1:ncpus=16:ngpus=4:mem=80gb
# nnn #PBS -N GPU_test_ft_scratch_capy