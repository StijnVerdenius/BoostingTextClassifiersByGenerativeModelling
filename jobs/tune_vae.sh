



for lr in 0.05 0.005 0.0005
do
    for hidden_dim in 64 128
    do
        for z_dim in 16 32 64
        do
            echo 'tune-vae-lr-'$lr'-hd-'$hidden_dim'-z_dim-'$z_dim'seed-42.out'
            python3 -u main.py --generator BaseVAE --dataset_class LyricsDataset --loss ELBO --max_training_minutes 10 --embedding_size 256 --z_dim $z_dim --learning_rate $lr --hidden_dim $hidden_dim >> './jobs/tune-vae-lr-'$lr'-hd-'$hidden_dim'-z_dim-'$z_dim'seed-42.out'
        done
    done
done