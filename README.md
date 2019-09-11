# DL4NLP


## folders:

#### local_data

- gitignored
- meant for actual dataset and run-outputs (big or not-group-shared files)
- please don't save the above anywhere else
- for each executed-run, a folder with the start-time-datestamp will be created with in there: subdirectories for that executed-run to write to (including the current codebase at that time for debug purposes)

#### models

- all model classes (classifiers, generators whatnot)
- loss functions are models too
- also data-loader-classes are here
- make sure they all inherit from general model (except data-loaders of course)

#### preprocessing
 
- for all offline data manipulation
- for data exploration (maybe jupyter notebook)
- plotting perhaps

#### utils

- general helper functions
    - functionality for saving models
    - functionality for loading model-classes
    - functionality for building up directory structure
    - you name it
- constants that are shared project wide



## main.py

class handles the high-level functionality. Designed so that it can be controlled purely by parsed arguments.
you can choose whether you would like to run train or testing and load models in by string-reference, whilst still be able to pass keyword arguments to them.

## train.py

class handles any training process, regardless of model. therefore its set up rather abstractly.
If your model needs a different implementation of this shared trainer, please inherit from the trainer and override the functions you require done differently

## test.py

- nothing yet

## environment.yml

- env for project under the name dl4nlp. build with:

###### # conda env create -f environment.yml

## .gitignore

- all paths that are not shared among everyone through git but are personal

