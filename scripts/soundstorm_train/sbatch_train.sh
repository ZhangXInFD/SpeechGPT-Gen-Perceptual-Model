#!/bin/bash

#SBATCH --partition=x090
#SBATCH --time=72:00:00

#SBATCH --job-name=cnf_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G



head_node_ip=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)

echo Node IP: $head_node_ip nodes_array: $SLURM_NODELIST

# export NCCL_SOCKET_IFNAME=eth0
export LOGLEVEL=ERROR
export NCCL_DEBUG=INFO #TRACE # 可以改成 ERROR，减少输出量
#export CUDA_LAUNCH_BLOCKING=1

srun bash -c 'echo $SLURMD_NODENAME-$SLURM_JOB_GPUS' # 打印出不同机器上分配的显卡编号

# srun \
# torch-run --nnodes 2 --nproc_per_node 2 \
# --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29549 \
# elastic_ddp.py

srun accelerate launch scripts/soundstorm_train/hier_dac_train.py \
    -c config/SoundStorm/hier/hubert_dac.yaml \
    --continue_train
# torchrun --nnodes 1 \
#     --nproc_per_node 4 \
#     --rdzv_id $RANDOM \
#     --rdzv_backend c10d \
#     --rdzv_endpoint $head_node_ip:29549\
#     # --master_port 50025 \
#     scripts/train.py \
#     -c config/uconformer/spt_snake_cfg.yaml

sync && echo "success"