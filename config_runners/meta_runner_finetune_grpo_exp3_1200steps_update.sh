
#!/bin/bash
#PBS -q default@pbs-m1.metacentrum.cz
#PBS -l walltime=20:0:0
#PBS -l select=1:ncpus=4:ngpus=1:mem=50gb:scratch_local=400mb:cl_bee=True
#PBS -N run_finetune_grpo

cd /storage/brno2/home/ahajek/Spektro/MassGenie
source /storage/brno2/home/ahajek/miniconda3/bin/activate trl
echo $CONDA_PREFIX
./config_runners/run_finetune_grpo_exp3_1200steps_update.sh

exit

# # !/bin/bash
# # PBS -q gpu_dgx@pbs-m1.metacentrum.cz
# # PBS -l walltime=30:0:0
# # PBS -l select=1:ncpus=5:ngpus=1:mem=50gb:scratch_local=400mb
# # PBS -N run_finetune_cot