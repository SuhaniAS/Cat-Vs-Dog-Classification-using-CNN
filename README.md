# Cat-Vs-Dog-Classification-using-CNN

## Objective:
### Develop a machine learning model that can classify images into two categories: 'cat' and 'dog'.
#
## Tasks:
### 1. Select a dataset that includes images of cats and dogs labelled as 0 and 1 or 'cats' and 'dogs'.
### 2. Train a machine learning model to classify images into the two main categories.
### 3. Evaluate the model's performance using classification metrics such as accuracy.
#
## Deliverables:
### A Jupyter notebook or Python script with the code for data preprocessing, model training, and evaluation.
### A report summarizing the methodology, model performance, and insights gained from the analysis.
#
#
#
# Report
## CNN: Convolutional Neural Network is a special kind of neural network used for processing data that has a known grid-like topology like image.
## CNN has three main layers:
### 1. Convolution layer: extract features. (note: it is location dependent)
### 2. Pooling layer: location independent
### 3. Fully connected layer: just like ANN
#
#
### Now lets begin.
####### I am taking the data directly from kaggle website: **https://www.kaggle.com/datasets/salader/dogs-vs-cats**
####### Before uploading the dataset, we will have to run some codes after uploading kaggle.json file.
**!mkdir -p ~/.kaggle
  !cp kaggle.json ~/.kaggle/**
####### Once the above code is executed, copy the API command from kagle website and paste is accordingly.
**!kaggle datasets download -d salader/dogs-vs-cats**
####### Now the dataset is in the zipfile format. To unzip, we use te following command:
**from zipfile import ZipFile
  






















#### Please go trough the belove link and come back to README.md
https://github.com/SuhaniAS/Cat-Vs-Dog-Classification-using-CNN/blob/main/Cat_vs_Dog_Classification_using_CNN.ipynb

#### You can see that, overfitting is happening.
#### There are multiple ways to reduce overfitting. Some of them are:
##### 1. Add more data
##### 2. Droupout
##### 3. Batch Normalization
##### 4. Data Augmentation
#
##### Adding more data is not possible if we have used all the data we have.
##### I have used Droupout and Batch Normalization to reduce overfitting. please go through the belove link.
https://github.com/SuhaniAS/Cat-Vs-Dog-Classification-using-CNN/blob/main/Cat_vs_Dog_Classification_using_CNN(1).ipynb

