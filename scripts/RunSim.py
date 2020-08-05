#!/usr/bin/env python

import rospy
from SimulationClass import *
from nav_msgs.msg import Odometry
import numpy as np

if __name__ == '__main__':

    rospy.init_node('RunSim') 
    # rospy.sleep(7)

    # test = []
    # test.append([0, 1])
    # test.append([34, 0])
    # test.reverse()
    # print(np.min(test, axis=1))
    # test = np.array(test)
    # print(test[:, 1])
    # session = tf.compat.v1.Session()
    # tf.keras.backend.clear_session()
    # tf.compat.v1.keras.backend.set_session(session)

    Sim = SimulationClass([17, -8, 2], StartWithWeights=False)

    



