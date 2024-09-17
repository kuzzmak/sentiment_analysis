# Environment preparation

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
