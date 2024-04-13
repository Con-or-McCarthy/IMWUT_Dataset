# IMWUT_Dataset

Anonymous GitHub reopsitory containing the Dataset contribution of the paper "In- and Out-of-Distribution Multi-sensor Human Activity Recognition"

Create the environment using:
```
conda env create -f environment.yml
```
The Dataset files are stored in /Dataset/

Running
```
python data_preprocessing.py
```
Will convert the data into 10s and 30s windows with 50% overlap, stored in data/clean/w10 and data/clean/w30, respectively.

The code and model weights used in the paper will be made public upon its acceptance. 