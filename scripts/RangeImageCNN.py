#!/usr/bin/env python

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

from SimulationClass import *

PossibleActions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0], 
        [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, -1.0, 0.0], [-1.0, 0.0, -1.0], [0.0, -1.0, -1.0], [-1.0, -1.0, -1.0]]

TrainingData = []

FlattenedTrainingData = np.loadtxt('/home/satchel/m100_ws/src/ctrl_planning/training_data/TrainingDataFVs.txt').astype(float)

LabelData = np.loadtxt('/home/satchel/m100_ws/src/ctrl_planning/training_data/TrainingDataLabels.txt') #.astype(float)

# Now filter out (state, action) pairs where the action is [0, 0, 0], i.e. index 0
done_filtering = False
i = 0
while not done_filtering:
    if (i == len(LabelData) - 1):
        done_filtering = True
    if (LabelData[i] == 0.0):
        LabelData = np.delete(LabelData, i)
        FlattenedTrainingData = np.delete(FlattenedTrainingData, i, axis=0)
    
    i+=1


for i in range (len(FlattenedTrainingData)):
    TrainingData.append(np.reshape(FlattenedTrainingData[i], (16, 90)))
    

if __name__ == '__main__':
    
    model = models.Sequential()
    conv1 = layers.Conv2D( 32, (3, 3), activation='relu', input_shape=(16, 90, 1) )
    model.add( conv1 ) # The first two parameters specify the number of filters and the size of each filter, respectively. s
    maxPool1 = layers.MaxPooling2D( pool_size=(3, 4) )
    model.add( maxPool1 ) 
    # At this point, each feature map has size 4 x 29, and there are 32 feature maps.
    model.add( layers.Conv2D( 64, (2, 3), activation='relu' ) )
    maxPool2 = layers.MaxPooling2D( pool_size=(2, 4) )
    model.add( maxPool2 ) 
    model.add(layers.Flatten())
    # # model.add( layers.Conv2D( 64, (3, 3), activation='relu' ) )
    model.add(layers.Dense(15)) # Here the parameter is the dimensionality of the output space, i.e. possible outputs. This means
    # that since we have, say 18 possible actions for our agent (acceleration vectors), we should put 18 here. We'll then use softmax on these 18 outputs like in the neural 
    # network built in ML class. Softmax's purpose here would be to give us 18 probablites, one for each action (label):
    model.add(layers.Activation('softmax'))



    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.summary()


    TrainingData = np.expand_dims(TrainingData, -1)

    history = model.fit(TrainingData, LabelData, epochs=20)

    # predictions = model.predict(TrainingData)
    # print(np.argmax(predictions[0]))

    

    plt.plot(history.history['acc'], label='acc')
    plt.xlabel('Epoch')
    plt.ylabel('acc')
    plt.ylim([0.7, 1])
    plt.legend(loc='lower right')
    plt.show()


    rospy.init_node('test')
    Sim = SimulationClass([35, 0, 2], 'Test', model)
    



