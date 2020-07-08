#!/usr/bin/env python

# import heapq
# import random
# import time

# def createArray():
#     array = range( 10 * 1000 * 1000 )
#     random.shuffle( array )
#     return array

# def linearSearch( bigArray, k ):
#     return sorted(bigArray, reverse=True)[:k]

# def heapSort( List, k ):
#     heap = []
#     # Note: below is for illustration. It can be replaced by 
#     # return heapq.nlargest( List, k )
#     for item in List:
#         # If we have not yet found k items, or the current item is larger than
#         # the smallest item on the heap,
#         if len(heap) < k or item > heap[0]:
#             # If the heap is full, remove the smallest element on the heap.
#             if len(heap) == k: heapq.heappop( heap )
#             # add the current element as the new smallest.
#             heapq.heappush( heap, item )
#     return heap

# start = time.time()
# bigArray = createArray()
# print "Creating array took %g s" % (time.time() - start)

# start = time.time()
# print linearSearch( bigArray, 10 )    
# print "Linear search took %g s" % (time.time() - start)

# start = time.time()
# print heapSearch( bigArray, 10 )    
# print "Heap search took %g s" % (time.time() - start)

from sensor_msgs.msg import PointCloud2
from sensor_msgs.point_cloud2 import read_points
import numpy as np

def pcl_callback(pcl_msg):


def get_distance_to_closest_point(pc):
    
    points = np.array(list(read_points(pc)))
    xyz = np.array([(x, y, z) for x, y, z, _, _ in points]) # assumes XYZIR
    r = np.linalg.norm(xyz, axis=-1)
    return np.min(r)



rospy.Subscriber(
        '/velodyne_points', PointCloud2, pcl_callback,
    )