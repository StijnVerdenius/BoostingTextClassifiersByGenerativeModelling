#!/bin/bash
#SBATCH --job-name=train_lstm
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=60000M
#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1
module purge
module load eb
module load Python/3.6.3-foss-2017b
module load cuDNN/7.0.5-CUDA-9.0.176
module load NCCL/2.0.5-CUDA-9.0.176
export LD_LIBRARY_PATH=/hpc/eb/Debian9/cuDNN/7.1-CUDA-8.0.44-GCCcore-5.4.0/lib64:$LD_LIBRARY_PATH


for lr in 0.05 0.005 0.0005
do
    for hidden_dim in 64 128
    do
        srun python3 -u main.py --classifier LSTMClassifier --dataset_class LyricsDataset --loss CrossEntropyLoss --train-classifier --learning_rate $lr --hidden_dim $hidden_dim >> 'output/tune-lstm-lr-'$lr'-hd-'$hidden_dim'-seed-42.out'
    done
done
