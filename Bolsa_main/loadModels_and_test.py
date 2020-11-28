#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file loads the pretrained models and test them.
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from tensorflow import keras
import cv2

def loadModel(modelName , imgToClassify = None):
    """
    

    Parameters
    ----------
    modelName : TYPE string
        DESCRIPTION. it's the models path
    imgToClassify : TYPE, optional :string
        DESCRIPTION. The default is None.
        It's the path of the image to be classify'

    Returns
    -------
    None.
    Prints in the console the classifier accuracy for test data 
    And classifies the imgToClassify if it was given to the argparser.
    """
    
    num_classes=10
    mnist = tf.keras.datasets.mnist

    #Split data into training sets & test sets
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    
    train_images = train_images.astype("float32") / 255
    test_images = test_images.astype("float32") / 255
    
    train_images = np.expand_dims(train_images, -1)
    test_images = np.expand_dims(test_images, -1)
    
    
    train_labels = keras.utils.to_categorical(train_labels, num_classes)
    test_labels = keras.utils.to_categorical(test_labels, num_classes)

    
    
    ###LOAD THE MODEL:
    loaded_model= keras.models.load_model(modelName)

    
    #loaded_model.summary()
    
    loss, acc= loaded_model.evaluate(test_images,test_labels,verbose=0)
    print()
    print(">>The accuracy of the loaded model is: ",acc)

    if imgToClassify != None:
        print("We're going to classify your image...")
        preprocessImg_Classify(loaded_model,imgToClassify)




def preprocessImg_Classify(model, imgPath):
    #gray = cv2.imread("StockImg/img5.png",cv2.IMREAD_GRAYSCALE)
    
    gray = cv2.imread(imgPath,cv2.IMREAD_GRAYSCALE)
    #resize img and invert it (black background)
    gray = cv2.resize(255-gray , (28,28))
    (thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    #save the processed img
    #cv2.imwrite("StockImg/pro_img9.png",gray)
    cv2.imwrite(imgPath+"_pro",gray)
    


    flatten=gray.flatten()/255.0
    prediction = model.predict(flatten.reshape(1, 28, 28, 1))
    print()
    print(prediction)
    print(">>THE PREDICTION FOR YOUR IMAGE IS:",np.argmax(prediction))
