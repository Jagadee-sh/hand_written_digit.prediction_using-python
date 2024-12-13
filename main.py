                                                                                  Hand Written Digit Prediction - Classification Analysis
#Objective:
Handwritten Digit Prediction - Classification Analysis is to develop a highly accurate machine learning model that can classify 
handwritten digits (0-9) with precision and recall. The model should generalize well to new data, avoid overfitting, and 
be deployable in real-world applications, such as optical character recognition systems.

#Data Source
sklearn.datasets

#Import Library

import pandas as pd
     

import numpy as np
     

import matplotlib.pyplot as plt
     
#Import Data

from sklearn.datasets import load_digits
     

df = load_digits()
     

_, axes = plt.subplots(nrows = 1,ncols =4, figsize = (10,3))
for ax, image, label in zip(axes,df.images,df.target):
  ax.set_axis_off()
  ax.imshow(image, cmap=plt.cm.gray_r, interpolation = "nearest")
  ax.set_title("Training: %i" % label)
     

#Data Preprocessing

df.images.shape
     
(1797, 8, 8)

df.images[0]
     
array([[ 0.,  0.,  5., 13.,  9.,  1.,  0.,  0.],
       [ 0.,  0., 13., 15., 10., 15.,  5.,  0.],
       [ 0.,  3., 15.,  2.,  0., 11.,  8.,  0.],
       [ 0.,  4., 12.,  0.,  0.,  8.,  8.,  0.],
       [ 0.,  5.,  8.,  0.,  0.,  9.,  8.,  0.],
       [ 0.,  4., 11.,  0.,  1., 12.,  7.,  0.],
       [ 0.,  2., 14.,  5., 10., 12.,  0.,  0.],
       [ 0.,  0.,  6., 13., 10.,  0.,  0.,  0.]])

df.images[0].shape
     
(8, 8)

len(df.images)
     
1797

n_samples = len(df.images)
data = df.images.reshape((n_samples, -1))
     

data[0]
     
array([ 0.,  0.,  5., 13.,  9.,  1.,  0.,  0.,  0.,  0., 13., 15., 10.,
       15.,  5.,  0.,  0.,  3., 15.,  2.,  0., 11.,  8.,  0.,  0.,  4.,
       12.,  0.,  0.,  8.,  8.,  0.,  0.,  5.,  8.,  0.,  0.,  9.,  8.,
        0.,  0.,  4., 11.,  0.,  1., 12.,  7.,  0.,  0.,  2., 14.,  5.,
       10., 12.,  0.,  0.,  0.,  0.,  6., 13., 10.,  0.,  0.,  0.])

data[0].shape
     
(64,)

data.shape
     
(1797, 64)
Scaling Image Data

data.min()
     
0.0

data.max()
     
16.0

data = data/16
     

data.min()
     
0.0

data.max()
     
1.0

data[0]
     
array([0.    , 0.    , 0.3125, 0.8125, 0.5625, 0.0625, 0.    , 0.    ,
       0.    , 0.    , 0.8125, 0.9375, 0.625 , 0.9375, 0.3125, 0.    ,
       0.    , 0.1875, 0.9375, 0.125 , 0.    , 0.6875, 0.5   , 0.    ,
       0.    , 0.25  , 0.75  , 0.    , 0.    , 0.5   , 0.5   , 0.    ,
       0.    , 0.3125, 0.5   , 0.    , 0.    , 0.5625, 0.5   , 0.    ,
       0.    , 0.25  , 0.6875, 0.    , 0.0625, 0.75  , 0.4375, 0.    ,
       0.    , 0.125 , 0.875 , 0.3125, 0.625 , 0.75  , 0.    , 0.    ,
       0.    , 0.    , 0.375 , 0.8125, 0.625 , 0.    , 0.    , 0.    ])
Define Target Variable (y) and Feature Variables (X)


     
# Train Test Split

from sklearn.model_selection import train_test_split
     

X_train, X_test, y_train, y_test = train_test_split(data, df.target,test_size = 0.3)
     

X_train.shape, X_test.shape, y_train.shape, y_test.shape
     
((1257, 64), (540, 64), (1257,), (540,))
Random Forest Model

from sklearn.ensemble import RandomForestClassifier
     

rf = RandomForestClassifier()
     

rf.fit(X_train, y_train)
     
RandomForestClassifier()
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
Predict Test Data

y_pred = rf.predict(X_test)
     

y_pred
     
array([7, 1, 9, 4, 3, 0, 2, 8, 5, 7, 8, 7, 2, 0, 1, 2, 6, 8, 7, 3, 6, 9,
       1, 7, 4, 3, 3, 7, 5, 6, 0, 7, 4, 0, 0, 3, 9, 4, 0, 1, 4, 4, 2, 0,
       3, 0, 4, 4, 6, 5, 3, 8, 1, 4, 5, 5, 9, 4, 5, 0, 2, 4, 9, 6, 5, 2,
       6, 6, 7, 3, 1, 2, 5, 8, 0, 8, 3, 2, 9, 3, 1, 5, 9, 3, 1, 3, 5, 0,
       6, 4, 4, 5, 3, 5, 1, 1, 0, 2, 0, 4, 5, 6, 3, 3, 6, 0, 5, 5, 5, 1,
       0, 9, 1, 8, 7, 0, 7, 1, 7, 0, 2, 2, 6, 3, 1, 6, 9, 3, 2, 4, 8, 0,
       1, 2, 9, 9, 2, 7, 0, 3, 2, 6, 0, 1, 1, 0, 7, 2, 3, 4, 3, 6, 0, 1,
       7, 7, 9, 8, 8, 5, 7, 6, 2, 7, 8, 6, 4, 8, 8, 5, 0, 7, 9, 3, 4, 7,
       0, 0, 9, 3, 5, 6, 7, 7, 2, 4, 7, 6, 8, 8, 6, 4, 9, 3, 5, 3, 7, 3,
       3, 3, 6, 1, 2, 9, 1, 4, 8, 3, 1, 1, 2, 9, 9, 8, 5, 6, 4, 8, 6, 1,
       9, 1, 0, 0, 4, 3, 4, 8, 5, 6, 1, 8, 0, 2, 6, 9, 9, 2, 4, 8, 7, 3,
       0, 8, 1, 7, 7, 3, 6, 7, 1, 3, 7, 3, 4, 8, 1, 5, 5, 1, 1, 8, 8, 1,
       9, 3, 8, 2, 2, 7, 5, 7, 7, 6, 9, 4, 5, 1, 8, 2, 5, 0, 3, 6, 7, 4,
       2, 3, 2, 1, 6, 7, 7, 8, 3, 6, 4, 6, 1, 9, 7, 9, 5, 8, 1, 7, 5, 8,
       0, 1, 8, 9, 2, 2, 0, 4, 1, 4, 3, 5, 5, 9, 9, 6, 7, 4, 7, 9, 0, 0,
       8, 2, 1, 0, 4, 6, 9, 0, 7, 3, 2, 9, 8, 1, 7, 8, 0, 9, 0, 8, 5, 9,
       0, 2, 8, 3, 3, 3, 1, 6, 9, 9, 5, 5, 5, 1, 0, 5, 4, 1, 8, 0, 7, 1,
       6, 8, 5, 2, 3, 2, 9, 5, 7, 7, 5, 7, 5, 8, 1, 9, 2, 2, 3, 7, 8, 6,
       5, 8, 1, 8, 4, 5, 1, 4, 3, 6, 9, 7, 5, 5, 7, 0, 1, 5, 4, 3, 2, 5,
       2, 4, 0, 2, 0, 5, 7, 0, 2, 7, 5, 3, 4, 6, 2, 2, 9, 5, 2, 8, 0, 6,
       9, 4, 2, 0, 4, 9, 9, 0, 4, 0, 1, 2, 4, 0, 7, 1, 0, 2, 4, 7, 2, 7,
       0, 7, 1, 1, 6, 7, 8, 3, 3, 2, 3, 2, 8, 5, 0, 4, 7, 9, 8, 4, 3, 0,
       5, 1, 0, 8, 3, 4, 3, 6, 6, 7, 8, 0, 5, 0, 5, 1, 2, 2, 6, 2, 5, 5,
       7, 3, 9, 1, 1, 8, 9, 9, 3, 8, 4, 5, 9, 3, 8, 6, 6, 8, 3, 1, 9, 3,
       9, 6, 3, 6, 2, 3, 7, 4, 2, 2, 7, 7])
Model Accuracy

from sklearn.metrics import confusion_matrix, classification_report
     

confusion_matrix(y_test,y_pred)
     
array([[55,  0,  0,  0,  1,  0,  0,  0,  0,  0],
       [ 0, 53,  0,  0,  0,  0,  0,  0,  0,  0],
       [ 1,  0, 55,  0,  0,  0,  0,  0,  0,  0],
       [ 0,  0,  0, 59,  0,  0,  0,  0,  3,  0],
       [ 0,  0,  0,  0, 48,  0,  0,  2,  0,  0],
       [ 0,  0,  0,  0,  0, 55,  0,  0,  0,  2],
       [ 1,  0,  0,  0,  0,  1, 46,  0,  0,  0],
       [ 0,  0,  0,  0,  0,  0,  0, 57,  0,  0],
       [ 0,  2,  0,  0,  0,  0,  0,  1, 48,  0],
       [ 0,  1,  0,  1,  0,  0,  0,  0,  1, 47]])

print(classification_report(y_test, y_pred))
     
              precision    recall  f1-score   support

           0       0.96      0.98      0.97        56
           1       0.95      1.00      0.97        53
           2       1.00      0.98      0.99        56
           3       0.98      0.95      0.97        62
           4       0.98      0.96      0.97        50
           5       0.98      0.96      0.97        57
           6       1.00      0.96      0.98        48
           7       0.95      1.00      0.97        57
           8       0.92      0.94      0.93        51
           9       0.96      0.94      0.95        50

    accuracy                           0.97       540
   macro avg       0.97      0.97      0.97       540
weighted avg       0.97      0.97      0.97       540




#Explaination:
Firstly i import three libraries as follows: panadas,numpy,matplotib.pyplot

Secondly imported the data from datasource sklearn.datasets and load it.

After that performs Image Preprocessing.

Scale the imported image data.

5.From the imported library import classifiers.

Predict the test data.

Check the model Accuracy by import confusion matrix and classification report.
