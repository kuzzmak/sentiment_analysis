# Sentiment analysis

Goal of this project is to predict sentiment of user messages by finetuning `BERT` model. Data on which finetuning will be done is located in `data\rn_data.csv.gz`.

## Environment preparation

One of the simpler ways to have working python environment is to install and use [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). After `conda` in installed run following command to init environment

```bash
conda create -n sentiment_env python=3.9
```

This created new conda environment named `sentiment_env` along with the python 3.9 interpreter for that environment. Activate the created environment by executing

```bash
conda activate sentiment_env
```

Install pytorch with following command

```bash
pip install torch  --index-url https://download.pytorch.org/whl/cu118
```

NOTE: this torch installation uses cuda 11.8, if you'd like to install pytorch with some other configuration, consult [pytorch](https://pytorch.org/) homepage.

Install all the other required packages with

```bash
pip install -r requirements.txt
```

Now your environment should be complete.

## Data preparetion

Data for finetuning can be downloaded on [this](https://drive.google.com/file/d/12TRJWiUT69hfffJBfEs_DYh4_CX41mCP/view?usp=sharing) link. It needs to be put into `data` folder inside this repository. It first needs to be prepared in order to be used for model finetuning. This can be done by executing following command

```bash
python preprocess_data.py
```

- Process extracts data in `data\rn_data.csv`
- Data is filtered in a way that corrupt lines are discarded. Raw filtered data is located in `data\preprocessed_data.txt`. Corrupt data is located in `data\corrupt_data.txt`.
- Preprocessed data is split into train, val and test datasets. Ratio for train data is 0.7, val 0.2 and test 0.1. Train data is stored in `data\train_data.txt`.

## Finetuning

In order to fintune BERT, run following command

```bash
python train.py --epochs 3 --train_samples 100000 --val_samples 10000 --test_samples 10000 --batch_size 128
```

It is recommended by the [BERT](https://arxiv.org/pdf/1810.04805) paper authors that finetuning is done for 3 epochs and learning rate is in the range of e-5.

This process generates new model weights which can be found in `checkpoints/data_and_time_folder/best`. There is also a `Tensorboard` file generated in `runs/sentiment_analysis/data_and_time_folder` which can be used to analyze finetuning process. `Tensorboard` can be launched like

```bash
tensorboard --logdir runs\sentiment_analysis
```

which spawns local server on `http://localhost:6006`

## Inference

In order to run inference using `BERT` it's necessary to have at least one checkpoint in `checkpoints/data_and_time_folder`.

Weights of the finetuned model can be downloaded on [this](https://drive.google.com/drive/folders/12Rz8HtjgOZnWTmJ69MG3UCcwgiXgxr6A?usp=sharing) link. Place the downloaded `checkpoints` in the root of this project.

Inference can be ran using following command

```bash
python infer.py --run_name 2024-09-22_22-31-42 --text "I hate this product!"
```

`run_name` is the foler name in `checkpoints` folder, i.e. there needs to exist `checkpoints/2024-09-22_22-31-42` folder with model weights.

Mentioned command outputs something like

```bash
Model predicted that the sentiment of the message:

        "staying @ home today"
is
        "Negative"
```
