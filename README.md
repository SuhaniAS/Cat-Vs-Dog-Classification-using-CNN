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
###### I am taking the data directly from kaggle website: **https://www.kaggle.com/datasets/salader/dogs-vs-cats**
###### Before uploading the dataset, we will have to run some codes after uploading kaggle.json file.
**!mkdir -p ~/.kaggle
  !cp kaggle.json ~/.kaggle/**
###### Once the above code is executed, copy the API command from kagle website and paste is accordingly.
**!kaggle datasets download -d salader/dogs-vs-cats**
###### Now the dataset is in the zipfile format. To unzip, we use te following command:
![Screenshot 2024-04-21 212053](https://github.com/SuhaniAS/Cat-Vs-Dog-Classification-using-CNN/assets/137792301/e19b0cdd-dbe5-4e0b-8735-a9033dba3084)
###### Import train and test dataset
![Screenshot 2024-04-21 212220](https://github.com/SuhaniAS/Cat-Vs-Dog-Classification-using-CNN/assets/137792301/20df148e-6637-4c32-822b-6948a691838c)
###### Normalize the data
![Screenshot 2024-04-21 212502](https://github.com/SuhaniAS/Cat-Vs-Dog-Classification-using-CNN/assets/137792301/7c0582d4-6f73-4b63-b449-c69189e6657a)
##### Build the CNN model using the following library
**from keras import Sequential
  from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten**
![Screenshot 2024-04-21 212809](https://github.com/SuhaniAS/Cat-Vs-Dog-Classification-using-CNN/assets/137792301/01f85a3e-fea9-4ccf-bac7-c8f8d9f450c5)
##### Now compile and fir the model
![Screenshot 2024-04-21 212950](https://github.com/SuhaniAS/Cat-Vs-Dog-Classification-using-CNN/assets/137792301/11f57d98-4dd3-4b66-bc46-0f0c28fcc69c)
![Screenshot 2024-04-21 213022](https://github.com/SuhaniAS/Cat-Vs-Dog-Classification-using-CNN/assets/137792301/3fc254ed-3cce-41c9-b699-c9b60569e92b)
##### Check the accuracy and loss of the model graphically
![Screenshot 2024-04-21 213149](https://github.com/SuhaniAS/Cat-Vs-Dog-Classification-using-CNN/assets/137792301/54a6a10c-78cf-4f5f-899a-fc4abe6e58d5)
![Screenshot 2024-04-21 213234](https://github.com/SuhaniAS/Cat-Vs-Dog-Classification-using-CNN/assets/137792301/b7ac74d6-54a7-42d0-bb60-b286876d8663)
![Screenshot 2024-04-21 213259](https://github.com/SuhaniAS/Cat-Vs-Dog-Classification-using-CNN/assets/137792301/3d79c68f-92e3-4f1d-abf0-f82c7afcf214)
![Screenshot 2024-04-21 213333](https://github.com/SuhaniAS/Cat-Vs-Dog-Classification-using-CNN/assets/137792301/4ae55345-e52e-42f4-aa59-6f12dc99fff5)
##### From bothe the graphs, we can see that the model is performing well with the training dataset but not with test dataset. This indicates **Overfitting**.
#
##### We have multiple ways to avoid overfitting.
##### Some of them are:
##### 1. Increace te data size.
##### 2. Droupout method
##### 3. L1/L2 regularization
##### 4. BatchNormalization
##### 5. Reduce complexity
#
##### In this case, I have used *Dropout* and *BatchNormalization*.
![Screenshot 2024-04-21 213943](https://github.com/SuhaniAS/Cat-Vs-Dog-Classification-using-CNN/assets/137792301/9af676bf-845d-4870-91ac-f3ac28356756)
![Screenshot 2024-04-21 214014](https://github.com/SuhaniAS/Cat-Vs-Dog-Classification-using-CNN/assets/137792301/d3b2cbb7-bb58-418c-a420-69a19023498d)
![Screenshot 2024-04-21 214036](https://github.com/SuhaniAS/Cat-Vs-Dog-Classification-using-CNN/assets/137792301/d558188c-896a-49bd-bf69-d1f68e4d9fd2)
![Screenshot 2024-04-21 214113](https://github.com/SuhaniAS/Cat-Vs-Dog-Classification-using-CNN/assets/137792301/b70833c2-aed6-4edf-bde6-027ff6b9a04b)
![Screenshot 2024-04-21 214127](https://github.com/SuhaniAS/Cat-Vs-Dog-Classification-using-CNN/assets/137792301/cb4b3598-b59c-4b78-b4ab-91429b8878f8)
![Screenshot 2024-04-21 214210](https://github.com/SuhaniAS/Cat-Vs-Dog-Classification-using-CNN/assets/137792301/62461079-d419-4786-a9b3-0ffc3f1472d6)
![Screenshot 2024-04-21 214227](https://github.com/SuhaniAS/Cat-Vs-Dog-Classification-using-CNN/assets/137792301/afa998ee-9609-41b0-9fcd-ee8b41ed6da5)
##### **We can see the that the overfitting has reduced**
##### Now we will predict for an unknown data
![Screenshot 2024-04-21 214356](https://github.com/SuhaniAS/Cat-Vs-Dog-Classification-using-CNN/assets/137792301/fd60ea67-eda3-43fc-bd9d-ba8c391ba3d3)
![Screenshot 2024-04-21 214407](https://github.com/SuhaniAS/Cat-Vs-Dog-Classification-using-CNN/assets/137792301/3dd44282-b755-487e-adb4-d454817c4d78)
![Screenshot 2024-04-21 214417](https://github.com/SuhaniAS/Cat-Vs-Dog-Classification-using-CNN/assets/137792301/53530267-bba2-41aa-9e96-5e6dd02e5a41)
![Screenshot 2024-04-21 214427](https://github.com/SuhaniAS/Cat-Vs-Dog-Classification-using-CNN/assets/137792301/341f1c84-e3bd-41c9-8417-f81a9a9aa6bf)
**Thus the model has predicted the image as 1 (i.e., dog)**
