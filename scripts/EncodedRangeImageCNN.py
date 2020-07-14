#!/usr/bin/env python

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

from SimulationClass import *

sys.path.insert(1, '/home/satchel/ctrl_ws/src/ctrl_planner/scripts/AutoEncoder')

from CustomTrainingCallback import *

TrainingData = []

EncodedTrainingData = np.loadtxt('/home/satchel/ctrl_ws/src/ctrl_planner/training_data/TrainingDataEncodedRIs.txt').astype(float)

LabelData = np.loadtxt('/home/satchel/ctrl_ws/src/ctrl_planner/training_data/TrainingDataLabels.txt') #.astype(float)


done_filtering = False
i = 0
while not done_filtering:
    if (i == len(LabelData) - 1):
        done_filtering = True
    if (LabelData[i] == 0.0):
        LabelData = np.delete(LabelData, i)
        EncodedTrainingData = np.delete(EncodedTrainingData, i, axis=0)
    
    i+=1


for i in range (len(EncodedTrainingData)):
    TrainingData.append(EncodedTrainingData[i])



if __name__ == '__main__':

    input_dim = [len(TrainingData[0]), 1]
    model = models.Sequential()
    input_layer_model = layers.InputLayer(input_dim)
    model.add(input_layer_model) # Just a layer that acts as a placeholder for the input
    
    conv1 = layers.Conv1D( 
        filters=32, kernel_size=3, strides=1, activation='relu'
    )

    model.add( conv1 ) 

    conv2 = layers.Conv1D( 
        filters=64, kernel_size=4, strides=1, activation='relu'
    )

    model.add( conv2 ) 

    # maxpool1 = layers.MaxPool1D(
    #     pool_size=2
    # )

    # model.add(maxpool1)

    conv3 = layers.Conv1D( 
        filters=16, kernel_size=3, strides=1, activation='relu'
    )

    model.add( conv3 ) 

    conv4 = layers.Conv1D( 
        filters=8, kernel_size=2, strides=1, activation='sigmoid'
    )

    model.add( conv4 ) 
    
    model.add(layers.Flatten())
    model.add(layers.Dense(len(PossibleActions))) # Here the parameter is the dimensionality of the output space, i.e. possible outputs. This means
    # # that since we have, say 18 possible actions for our agent (acceleration vectors), we should put 18 here. We'll then use softmax on these 18 outputs like in the neural 
    # # network built in ML class. Softmax's purpose here would be to give us 18 probablites, one for each action (label):
    model.add(layers.Activation('softmax'))



    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.summary()


    TrainingData = np.expand_dims(TrainingData, -1)

    Encoded_train, Encoded_val = train_test_split(TrainingData, test_size=0.1, random_state=42)

    history = model.fit(x=TrainingData, y=LabelData, epochs=70, callbacks=[CustomTrainingCallback()])


    rospy.init_node('test')
    Sim = SimulationClass([5, 0, 2], 'Test', model)
