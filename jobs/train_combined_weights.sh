#!/bin/bash
#SBATCH --job-name=train_comb_w
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


srun python3 main.py --classifier CombinedClassifier --joint_training --dataset_class LyricsDataset --num_classes 5 --embedding_size 256 --hidden_dim_vae 256 --hidden_dim 128 --z_dim 64 --learning_rate 0.01 --batch_size 1 --combined_classification --generator SentenceVAE --analysis --loss CrossEntropyLoss --dataset_class_sentencevae LyricsRawDataset --combination learn_sum --train-classifier --eval_freq 500 --classifier_dir full_lstm --vaes_dir full_vae/country,full_vae/hip-hop,full_vae/metal,full_vae/pop,full_vae/rock --classifier_name model_best --vaes_names model_best,model_best,model_best,model_best,model_best --run_name 'learncombine'
