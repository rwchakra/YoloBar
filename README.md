# VISUM 2022 Project
Official repository of the Project of the VISUM Summer School 2022.

## Create a SSH key for GitHub
First, you should create a SSH key for GitHub. You can go to the official GitHub tutorial [here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent).

## Clone repository
Then, to clone this repository you have to open a Terminal and run the following command:
```bash
$ git clone git@github.com:visum-summerschool/visum-competition2022.git
```

## Download database
Before training the models, we must be sure that the database structure is built. To download the data and build the structure, run the following command.
```bash
$ bash download_data.sh
```

## Install the requirements to set up your Python environment
To intall the minimum requirements you just need to run
```bash
$ pip install -r requirements.txt
```

## Train model
To train the baseline model, you have to run the following command (please be sure that your current directory is the root directory of the repository):
```bash
$ python code/model_train.py
```

This command has several constant variables you can change:
```
BATCH_SIZE - the batch size for the DataLoader
NUM_EPOCHS - the number of epochs for the trainin
IMG_SIZE - the image size you will use (H, W)
VAL_MAP_FREQ - the frequency you want the training loop to print the mAP values
```

## Creating and changing your own models
To do this, you can go to [model_utilities.py](code/model_utilities.py) and edit the `LoggiBarcodeDetectionModel`.

## Plot results
If you want to check and visualise some of you results on validation (or training) you can run the following command:
```bash
$ python code/plot_results.py
```

## Creating a submission
After the training of your models you have to create a submission file. This file will be .ZIP file containing: 1) the `code/` directory, the `results/models/visum2022.pt` directory, and the `Dockerfile`. Be sure that you fill the requirement needs of your Dockerfile, and the run the following command:
```bash
$ bash create_submission_file.sh
```
