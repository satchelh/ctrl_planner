#!/usr/bin/env python
from depthmapper.msg import DepthMap_msg
import rospy

import sys
sys.path.insert(1, '/home/satchel/ctrl_ws/src/ctrl_planner/scripts/AutoEncoder')

from AutoEncoder import *
from ActorCritic import *


tf.compat.v1.disable_eager_execution()
session = tf.compat.v1.Session()
tf.compat.v1.keras.backend.set_session(session)
ActorCritic = ActorCritic(session)

AutoEncoder = AutoEncoder()
AutoEncoder.load_model()
# AutoEncoder.encoder._make_predict_function()

rospy.init_node('test')

TrainingData = []

def RangeImageCallback(data):
    # rospy.loginfo("Recieved Range Image")
    # print(data.map.data[0])
    range_image = data.map.data 

    range_image = np.reshape(range_image, (16, 90))

    range_image = np.expand_dims(range_image, -1)
    range_image = np.expand_dims(range_image, 0)

    # tf.compat.v1.enable_eager_execution()
    with session.as_default():
        with session.graph.as_default():
            encoded_RI = AutoEncoder.encoder.predict(range_image)
    
            encoded_RI_with_GP = np.append(encoded_RI[0], [0, 0, 1], axis=0)
            # encoded_RI_with_GP = np.expand_dims(encoded_RI_with_GP, -1)
            encoded_RI_with_GP = np.expand_dims(encoded_RI_with_GP, 0)
            
            action = ActorCritic.act(encoded_RI_with_GP)
            print(action)
    
            reward = 1
            # encoded_RI_with_GP = np.squeeze(encoded_RI_with_GP, 0)
            curr_state = encoded_RI_with_GP
            new_state = encoded_RI_with_GP

            TrainingData.append([curr_state, action, reward, new_state])
            print('\nRetraining with ' + str(len(TrainingData)) + ' samples.\n\n')
            ActorCritic.train(TrainingData)
            
    # tf.compat.v1.disable_eager_execution()
    # rospy.sleep(1)
    return
    

rospy.Subscriber(
    '/depth_map', DepthMap_msg, RangeImageCallback,
)

rospy.spin()

# import numpy as np

# List = [1, 2, 3]
# List = np.expand_dims(List, 0)
# print(List)
# print(np.squeeze(List, 0))