#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains the functions to train the models called by "mainBolsa.py"

"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from tensorflow import keras
import cv2
def train_default_mnist(saveFileName , imageToTest=None):
    """
    Parameters
    ----------
    saveFileName : TYPE string
        DESCRIPTION. name of the model to be saved.
    imageToTest : TYPE, optional
        DESCRIPTION. The default is None.
        name of the image to test after the training phase (path)
    Returns
    -------
    None.
    SAVES THE MODEL IN THE PATH GIVEN BY saveFileName
    """
    
    
    print("BEGINNING TRAINING...")
    mnist = keras.datasets.mnist
    num_classes=10
    input_shape = (28, 28, 1)
    #Split data into training sets & test sets
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    
    #Scale all values from [0;255] to [0;1] 
    
    #train_images = train_images / 255.0
    #test_images = test_images / 255.0
    # Scale images to the [0, 1] range
    train_images = train_images.astype("float32") / 255
    test_images = test_images.astype("float32") / 255
    
    # Make sure images have shape (28, 28, 1)
    train_images = np.expand_dims(train_images, -1)
    test_images = np.expand_dims(test_images, -1)
        
        
    # convert class vectors to binary class matrices
    train_labels = keras.utils.to_categorical(train_labels, num_classes)
    test_labels =  keras.utils.to_categorical(test_labels, num_classes)



    
    # "Sequential layers dynamically adjust the shape of input to a layer based the out of the layer before it"
    model = keras.Sequential([
    keras.Input(shape=input_shape),
    keras.layers.Flatten(),  # input layer (1)input_shape=(28, 28)
    keras.layers.Dense(128, activation='relu'),  # hidden layer (2)
    keras.layers.Dense(num_classes, activation='softmax') # output layer (3)
    ])
    
    
    #COMPILE MODELl
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',#'sparse_categorical_crossentropy',
              metrics=['accuracy'])
    
    #TRAINING:
    #fit the model to the training data
    model.fit(train_images, train_labels, epochs=10)
    
    #Test model
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1) 
    print("The accuracy of the classifier default is:",test_acc)
    
    #Saving the model .h5
    #tmpString = 'Pretrained_models'+saveFileName+'.h5'
    #model.save('Pretrained_models/'+saveFileName+'.h5')
    print()
    if saveFileName!=None:
        model.save(saveFileName+'.h5')
        print("TRAINING OF THE DEFAULT MODEL & SAVING PROCESS IS OVER...")
    
    else:
        
        print("TRAINING OF THE DEFAULT MODEL IS OVER...")

    if imageToTest!=None:
        print("We're going to classify your image...")
        preprocessImg_Classify(model,imageToTest)


###############################################################################
#CNN from keras;
def train_CNN_mnist(saveFileName , imageToTest=None):
    """
    Parameters
    ----------
    saveFileName : TYPE string
        DESCRIPTION. name of the model to be saved.
    imageToTest : TYPE, optional
        DESCRIPTION. The default is None.
        name of the image to test after the training phase (path)
    Returns
    -------
    None.
    SAVES THE MODEL IN THE PATH GIVEN BY saveFileName

    """
    #Model / data parameters
    num_classes = 10
    input_shape = (28, 28, 1)
    
    mnist = keras.datasets.mnist
    #Split data into training sets & test sets
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    
    # Scale images to the [0, 1] range
    train_images = train_images.astype("float32") / 255
    test_images = test_images.astype("float32") / 255
    
    # Make sure images have shape (28, 28, 1)
    train_images = np.expand_dims(train_images, -1)
    test_images = np.expand_dims(test_images, -1)
        
        
    # convert class vectors to binary class matrices
    train_labels = keras.utils.to_categorical(train_labels, num_classes)
    test_labels =  keras.utils.to_categorical(test_labels, num_classes)
    
    
    ##############
    # BUILD MODEL
    model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation="softmax"),
    ]
    )
    ##############
    #Compile & training
    batch_size = 128
    epochs = 15
    
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    
    #Eval
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1) 
    
    
    
    print("The accuracy of the classifier CNN is:",test_acc)
    
    
    
    
    #Saving the model .h5
    #tmpString = 'Pretrained_models'+saveFileName+'.h5'
    #model.save('Pretrained_models/'+saveFileName+'.h5')
    print()
    if saveFileName!=None:
        model.save(saveFileName+'.h5')
        print("TRAINING OF THE CNN & SAVING PROCESS IS OVER...")
    else:
        
        print("TRAINING OF THE CNN IS OVER...")

    
    if imageToTest!=None:

        print("We're going to classify your image...")
        preprocessImg_Classify(model,imageToTest)
    
    
    
    
    
    
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
    
    
    
    
    