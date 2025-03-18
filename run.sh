#!/bin/bash

#- Job parameters

# (TODO)
# Please modify job name

#SBATCH -J spec_val             # The job name
#SBATCH -o ret-spec_val.out        # Write the standard output to file named 'ret-<job_number>.out'
#SBATCH -e ret-spec_val.err        # Write the standard error to file named 'ret-<job_number>.err'


#- Resources

# (TODO)
# Please modify your requirements

#SBATCH -p r8nv-gpu-hw               # Submit to 'r8nv-gpu-hw' Partition
#SBATCH -t 0-8:00:00                 # Run for a maximum time of 0 days, 12 hours, 00 mins, 00 secs
#SBATCH --nodes=1                    # Request N nodes
#SBATCH --gres=gpu:8                 # Request M GPU per node
#SBATCH --gres-flags=enforce-binding # CPU-GPU Affinity
#SBATCH --qos=gpu-long             # Request QOS Type

###
### The system will alloc 8 or 16 cores per gpu by default.
### If you need more or less, use following:
### #SBATCH --cpus-per-task=K            # Request K cores
###
### 
### Without specifying the constraint, any available nodes that meet the requirement will be allocated
### You can specify the characteristics of the compute nodes, and even the names of the compute nodes
###
### #SBATCH --nodelist=r8a30-a0          # Request a specific list of hosts 
### #SBATCH --constraint="A30|A100"      # Request GPU Type: A30 or A100_40GB
###

#- Log information

echo "Job start at $(date "+%Y-%m-%d %H:%M:%S")"
echo "Job run at:"
echo "$(hostnamectl)"
echo "$(df -h | grep -v tmpfs)"

#- Important setting!!!
##  otherwise it will cause an error of insufficient RDMA resources:
ulimit -l unlimited
##  otherwise it will result in an insufficient virtual memory size error, especially when loading LLM:
ulimit -v unlimited

#- Load environments
source /tools/module_env.sh
module list                       # list modules loaded

##- Tools
module load cluster-tools/v1.0
module load slurm-tools/v1.0

##- language
module load python3/3.8.16

##- CUDA
module load cuda-cudnn/11.6-8.4.1

##- virtualenv
# source xxxxx/activate

echo $(module list)              # list modules loaded
echo $(which gcc)
echo $(which python)
echo $(which python3)

#- Other
module load slurm-tools/v1.0
module load cluster-tools/v1.0
module load cuda-cudnn/11.6-8.4.1

cluster-quota                    # nas quota

nvidia-smi --format=csv --query-gpu=name,driver_version,power.limit # gpu info

#- WARNING! DO NOT MODIFY your CUDA_VISIBLE_DEVICES
#- in `.bashrc`, `env.sh`, or your job script
echo "Using GPU(s) ${CUDA_VISIBLE_DEVICES}"                         # which GPUs
#- The CUDA_VISIBLE_DEVICES variable is assigned and specified by SLURM
echo "This job is assigned the following resources by SLURM:"
scontrol show jobid $SLURM_JOB_ID -dd | awk '/IDX/ {print $2, $4}'

##- Monitor
# The script continues executing other tasks while the following command will execute after a while
module load slurm-tools/v1.0
(sleep 3h && slurm-gpu-atop-log-stats $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES) &
echo "Main program continues to run. Monitoring information will be exported after three hours."

#- Main program execution

# ##- Job step TODO
python -m OpenAGI.lambda_return_bs_train --batch_size 32 --lr 1e-4 --lambda_ 1
python -m OpenAGI.lambda_return_bs_train --batch_size 16 --lr 1e-4 --lambda_ 1
python -m OpenAGI.lambda_return_bs_train --batch_size 8 --lr 1e-4 --lambda_ 1

# python -m OpenAGI.lambda_return_train --batch_size 16 --lr 5e-5 --lambda_ 1
# python -m OpenAGI.lambda_return_train --batch_size 8 --lr 5e-5 --lambda_ 1
# python -m OpenAGI.predict_k.sft_ppo_distilbert_train --batch_size 32 --lr 1e-4
# python -m OpenAGI.predict_k.sft_ppo_distilbert_train --batch_size 16 --lr 1e-4
# python -m OpenAGI.predict_k.sft_ppo_distilbert_train --batch_size 8 --lr 1e-4
# python -m OpenAGI.predict_k.sft_ppo_distilbert_train --batch_size 16 --lr 5e-5
# python -m OpenAGI.predict_k.sft_ppo_distilbert_train --batch_size 8 --lr 5e-5

# sleep 3h
# python simulation_new_stepwise.py --acc 0.1
# python simulation_new_stepwise.py --acc 0.3
# python simulation_new_stepwise.py --acc 0.5
# python simulation_new_stepwise.py --acc 0.65
# python simulation_new_stepwise.py --acc 0.8
# python simulation_new_stepwise.py --acc 0.9
# python simulation_new_stepwise.py --acc 0.95

# python simulation_new.py --acc 0.1
# python simulation_new.py --acc 0.3
# python simulation_new.py --acc 0.5
# python simulation_new.py --acc 0.65
# python simulation_new.py --acc 0.8
# python simulation_new.py --acc 0.9
# python simulation_new.py --acc 0.95

# python mixed_predict_k.py --alpha 0.2
# python mixed_predict_k.py --alpha 0.4
# python mixed_predict_k.py --alpha 0.6
# python mixed_predict_k.py --alpha 0.8
# python mixed_predict_k.py --alpha 1.0

# python mixed_stepwise.py

# Insert your main job operation. 
# This can include running other scripts, executing Python programs, C++ binaries, or any relevant task.
# [Example: python my_script.py or ./my_program]

#- End
slurm-gpu-atop-log-stats $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES
echo "Job end at $(date "+%Y-%m-%d %H:%M:%S")"
# This will overwrite any existing atop logs from previous runs.
# WARNING: If your program times out or is terminated by scancel,
#          the above script part might not execute correctly.
