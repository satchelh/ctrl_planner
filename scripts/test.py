#!/usr/bin/env python

import rospy
from SimulationClass import *
from nav_msgs.msg import Odometry
import numpy as np



if __name__ == '__main__':

    rospy.init_node('test') 

    # test = []
    # test.append([0, 1])
    # test.append([34, 0])
    # test.reverse()
    # print(np.min(test, axis=1))
    # test = np.array(test)
    # print(test[:, 1])
    Sim = SimulationClass([17, 0, 2])

    



