# Boosting Text Classifiers by Generative Modelling

The aim of this project is to boost a simple classifier using per-class generative models that leverage the underlying per-class data marginal distribution. 
For this purpose we have implemented a LSTM for classification and VAEs for each class in our dataset. 
We demonstrate the promising results of this ensemble approach to boosting in our research paper. -MAYBE GIVE A LINK for paper in repo(if u wanna put it in)?-

![alt text](arch.png)

# Usage
 The implementation offers many different models, loss functions etc. to pick from, hence there are many configurations. 
 To run training or testing you need our pre-processed data sets which take up quite some space so they're not provided in this repository. 
 
## Training
Here you can find our final training preferences:
 -IF ITS BETTER ADD A LINK TO THE JOB THING- 
    
## Testing
Our final testing preferences:
 -IF ITS BETTER ADD A LINK TO THE JOB THING- 

### Loading already acquired results
We also provide a pickle file which loads a dictionary of our test logs consisting of combined, LSTM and VAE-Classifier models score results. 
These can be directly loaded and processed if run the test preferences with --skip_test.

# Configurations
#### List of parameters (this is konstantins) (maybe we can skip this:D dont bother)

| Parameter     | type          | default value  | description |
| ------------- |:-------------:| --------------:|-------------|
| `--z_dim` | int | 32 | Latent space dimensionality|
| `--hidden_dim` | int | 64 | Hidden dimension of a network|
| `--hidden_dim_vae` | int | 0 | Hidden dim of VAE for cases where we need both models|
| `--num_classes` | int | 5 | Number of classes|
| `--embedding_size` | int | 256 | Size of Embeddings|
| `--batch_size` | int | 64 | Batch size to use for the dataset |
| `--epochs` | int | 500 | Number of max epochs of the training|
| `--learning_rate` | float | 1e-3 | Learning rate |
| `--optimizer` | str | `Adam` | Optimizer|
| `--loss` | str | `CrossEntropyLoss` | Loss preference `CrossEntropyLoss`, `VAELoss`, `CombinedClassifier`|
| `--classifier` | str | `LSTMClassifier` | Model type for classifier
| `--generator` | str | `BaseVAE` | Model type for generator: `BaseVAE`, `SentenceVAE`|
| `--dataset_class` | str | `LyricsDataset` | Dataset type to use `LyricsDataset`, `LyricsRawDataset`|
| `--dataset_class_sentencevae` | str | `None` | To tell whether to datasets are necessary|
| `--genre` | str | `None` | Genre type for a class-specific VAE|
| `--test-mode` | action | `store_true` | Testing mode|
| `--analysis` | action | `store_true` | Whether to do analysis on test logs|
| `--train-classifier` | action | `store_true` | Classifier training (rather than sth else)|
| `--combined_classification` | action | `store_true` | Are we running the combined model (CombinedClassifier)|
| `--patience` | int | `30` | how long will the model wait for improvement before stopping training|
| `--combination` | str | `joint` | Combination heuristic to use in CombinedClassifier: `joint/learn_sum/learn_classifier`|
| `--classifier_dir` | str | ` ` | Classifier state-dict directory to load weights from|
| `--classifier_name` | str | ` ` | State dict file under `classifier_dir/models/`|
| `--vaes_dir` | str | ` ` | VAE state-dict directories to load weights from (split by comma)|
| `--vaes_names` | str | ` ` | State dict file under respective `vaes_dir/models/`(split by comma)|


#### Training Example (LSTM)

```
python3 main.py 
--classifier LSTMClassifier 
--dataset_class LyricsDataset 
--loss CrossEntropyLoss 
--train-classifier
```

#### Training Example (VAE)

```
python3 main.py 
--generator SentenceVAE 
--dataset_class LyricsRawDataset 
--loss VAELoss 
--batch_size 16 --eval_freq 100
--embedding_size 256 --hidden_dim 64 
--genre <GenreName> 
--run_name 'sentence-vae-genre-'<GenreName> 
```

#### Testing Example


```
python main.py
--test-mode
--analysis 
--classifier CombinedClassifier
--dataset_class LyricsDataset 
--dataset_class_sentencevae LyricsRawDataset
--generator SentenceVAE
--loss VAELoss
--num_classes 5 --embedding_size 256  --learning_rate 0.005 
--hidden_dim 128 --hidden_dim_vae 256 --z_dim 64 --batch_size 1 
--combined_classification 
--classifier_dir full_lstm 
--vaes_dir full_vae\country,full_vae\hip-hop,full_vae\metal,full_vae\pop,full_vae\rock 
--classifier_name model_best --vaes_names model_best,model_best,model_best,model_best,model_best 
--combination learn_sum
--combined_weights combine
```

LINK TO DATASET MAYBE?

## Links: (delete these at the end)

https://drive.google.com/open?id=12zTpLuKhGhmM5Ql2QvxoM0J3OJExbAo6

https://onedrive.live.com/?authkey=%21AERWc2QlWN0yqKE&id=5574F751815D9FB1%211766753&cid=5574F751815D9FB1

https://www.overleaf.com/2949321739vycbcgjmcddj

https://drive.google.com/drive/folders/1fj0jnOnTZAzYuimKFLfoqKLt2B8c7KZ4?usp=sharing


## sources:

https://arxiv.org/abs/1804.03599

http://proceedings.mlr.press/v70/yang17d/yang17d.pdf

https://github.com/kefirski/contiguous-succotash

https://arxiv.org/pdf/1809.03259.pdf

https://arxiv.org/pdf/1511.06349.pdf

https://arxiv.org/pdf/1809.03259.pdf
