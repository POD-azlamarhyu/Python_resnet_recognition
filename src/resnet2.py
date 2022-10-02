# -*- coding: utf-8 -*-

import os
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50,preprocess_input,decode_predictions
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,Model
from keras.layers import Input,Activation,Dropout,Flatten,Dense
from keras import optimizers
import numpy as np
import time
import glob
from PIL import Image
from pathlib import Path

tf.test.gpu_device_name()

def getdsample(cat,datadir):
    allfiles=[]
    for i,c in enumerate(cat):
        files=glob.glob(datadir+cat+"/*.png")
        print("category : {} sampledata : {}".format(cat,len(files)))
        for f in files:
            allfiles.append(f)

    
    return len(allfiles)

def main():
    classes=["ドライヤー","電気ケトル","オフィスチェア","ハサミ","コンセント"]
    classNums=len(classes)
    category=["dryer","kettle","officechair","scissors","outlet"]
    img_width,img_height=400,400

    traindatadir="./objdata/train/"
    testdatadir="./objdata/test/"

    trainSmp=getdsample(category,traindatadir)
    testSmp=getdsample(category,testdatadir)


    batchSize=100
    epoch=20

    trainDataset=ImageDataGenerator(
        rescale=1.0/224,
        zoom_range=0.2,
        horizontal_flip=True
        )
    validationDataset=ImageDataGenerator(rescale=1.0/224)

    traindata=trainDataset.flow_from_directory(
        traindatadir,
        target_size=(img_width,img_height),
        color_mode='rgb',
        classes=classes,
        class_mode='categorical',
        batch_size=batchSize,
        shuffle=True
    )

    validata=validationDataset.flow_from_directory(
        testdatadir,
        target_size=(img_width,img_height),
        color_mode='rgb',
        classes=classes,
        class_mode='categorical',
        batch_size=batchSize,
        shuffle=True
    )

    model=ResNet50(include_top=True,weights='imagenet',input_tensor=None,pooling=None)

    top_model=Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(256,actvation='relu'))
    top_model.add(Dropout(0.6))
    top_model.add(Dense(classNums,activation='softmax'))

    resnetmodel=Model(input=model.input,output=top_model(model.output))

    for layer in resnetmodel.layers[:15]:
        layer.trainable=False

    resnetmodel.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.SGD(lr='1e-3',momentum=0.9),
        metrics=['accuracy']
    )

    history=resnetmodel.fit_generator(
        traindata,
        samples_per_epoch=trainSmp,
        nb_epoch=epoch,
        validation_data=validata,
        nb_val_samples=testSmp
    )

    print(history.history['acc'])
    print(history.history['val_acc'])






