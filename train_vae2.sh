mkdir ./jobs/training/
mkdir ./jobs/training/vae/

for loss in MSE_ELBO NormalELBO CombinedELBO
do
    for genre in Rock Pop HipHop Metal Country
    do
        echo 'train-vae-loss-'$loss'-genre-'$genre'-seed-42.out'
        python3 -u main.py --generator BaseVAE --run_name '_'$loss'_'$genre'_' --dataset_class LyricsDatasetVAE --loss $loss --max_training_minutes 10 --genre $genre --embedding_size 256 --z_dim 16 --learning_rate 0.0025 --hidden_dim 64 >> './jobs/training/vae/train-vae-loss-'$loss'-genre-'genre'-seed-42.out'
    done
done

for loss in MSE_ELBO NormalELBO CombinedELBO
do
    echo 'train-vae-ALL-genre-'$genre'-seed-42.out'
    python3 -u main.py --generator BaseVAE --run_name '_'$loss'_ALL' --dataset_class LyricsDataset --loss $loss --max_training_minutes 10 --embedding_size 256 --z_dim 16 --learning_rate 0.0025 --hidden_dim 64 >> './jobs/training/vae/train-vae-ALL-genre-'$genre'-seed-42.out'
done