## Problem Statement
Experiment with developing and training a Handwriting recognition model using Convolutional
Neural Network and sequential model like LSTM together which could predict what is written in
an image by first extracting  its image features and feeding the image features to a sequential
model to further make prediction of character sequences.
## Dataset and Preprocessing
For training dataset is a simple one taken from Kaggle. 

Dataset Link: https://www.kaggle.com/datasets/landlord/handwriting-recognition

**Dataset summary**: 
The data is composed of first and last name of people written in grayscale images.
Train, validation and test set are separately provided.

Some pre-processing and cleaning steps are done to make the data more appropriate to train
an effective model:
The steps taken to clean are:
1. Remove labels having NaN values
2. Small letters are very few compared to capital letters, so removed them to avoid severe imbalance problem
3. Some samples have very large length, which are very few, thus removed all samples having characters greater than 20
4. Deciding what all images can be resized to
5. Removing wrong label images with identity "UNREADABLE"

All these steps can be visualized in the jupyter notebook.


## Modelling
Information about model architecture used can be found in modelling.py and training mechanism
using pytorch lightning in training.py and training_modules.py

A model with set of 4 convolutional layers are followed by 2 GRU layers are used to model the Neural Network

## Evaluation
The metric is evaluated using character error rate and exact match percentage.
Evaluation metrics can be visualized in weights and biases link [here](https://wandb.ai/nikhilsalodkar/handwriting_recognition_kaggle?workspace=user-nikhilsalodkar).

## Demo
A demo link build using streamlit and deployed on huggingface is [here](https://huggingface.co/spaces/niks-salodkar/Handwriting-Recog-Demo).
You might have to restart the huggingface space and wait a short while to try the demo.

The code for the demo could be found in app.py and inference.py and in huggingface repo.

## Requirements
The required packages can be viewed in reqs.txt. This file could include extra packages
which might not be necessary. A new conda environment is recommended if you want to
test out on your own.
