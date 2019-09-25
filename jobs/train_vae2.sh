
for loss in MSE_ELBO NormalELBO CombinedELBO
do
    for genre in Rock
    do
        for z_dim in 16 32 64
        do
            echo 'tune-vae-lr-'$lr'-hd-'$hidden_dim'-z_dim-'$z_dim'seed-42.out'
            python3 -u main.py --generator BaseVAE --dataset_class LyricsDataset --loss ELBO --max_training_minutes 10 --embedding_size 256 --z_dim 16 --learning_rate 0.005 --hidden_dim 128 >> './jobs/tune-vae-lr-'$lr'-hd-'$hidden_dim'-z_dim-'$z_dim'seed-42.out'
        done
    done
done

