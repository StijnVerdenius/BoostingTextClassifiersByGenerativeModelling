#!/bin/bash
#Set job requirements
#SBATCH --job-name=face_synth_with_landmarks  # todo: job name
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=47:00:00
#SBATCH --mem=60000M
#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1

echo "scratch_dir"
echo $TMPDIR

#Loading modules
module load Miniconda3/4.3.27
module load CUDA/9.0.176
module load cuDNN/7.3.1-CUDA-9.0.176

export LD_LIBRARY_PATH=/hpc/eb/Debian9/cuDNN/7.1-CUDA-8.0.44-GCCcore-5.4.0/lib6$
export PYTHONIOENCODING=utf8

echo "copy directory"
## TODO: FIX FOR THIS PROJECT
#mkdir $TMPDIR/lgpu0386
#cp -r $HOME/DeepFakes $TMPDIR/lgpu0386

echo "cd inwards"
## TODO: FIX FOR THIS PROJECT
#cd $TMPDIR/lgpu0386/DeepFakes


echo "activate env"
source activate dl4nlp

echo " ------ Job is started ------- "
echo "dir: "

echo $TMPDIR
echo $(pwd)

for GENRE in #todo: add genres here
do
	srun python3 main.py --generator BaseVAE --dataset_class LyricsDatasetVAE --loss ELBO --embedding_size 256 --genre $GENRE
done


## TODO: FIX FOR THIS PROJECT
#cp -r $TMPDIR/lgpu0386/DeepFakes/results/output/* $HOME/DeepFakes/results/output

echo " "
echo " ------ Job is finished -------"
