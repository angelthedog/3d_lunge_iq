# 3d_lunge_iq

## 1Preprocess.py
* Input: input.txt with list of bvh filenames and 3 scores
* Output: output.csv with all the data stored. Row 1 is the header.

## 2TrainMain.py
* Input: output.csv
* Output: the lstm_model.pth pytorch model

## How to run
```
poetry shell # activate .venv
python 1Preprocess.py # generate training dataframe
python 2TrainMain.py # train model
```