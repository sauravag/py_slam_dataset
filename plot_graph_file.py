#!/usr/bin/env python
#*********************************************************************
#
#  Copyright (c) 2017, Saurav Agarwal.
#  All rights reserved.
#
#*********************************************************************

# Authors: Saurav Agarwal

import os
import math as m
import numpy as np
import matplotlib.pyplot as plt
from base_utils import *

ARROW_LENGTH = 1.0

def compute_trajectory_length(traj):

    r,c = traj.shape

    d = 0

    for i in range(0,r-1):

        x1 = traj[i,:]

        x2 = traj[i+1,:]

        d_x1x2 = np.linalg.norm(x1-x2)

        d = d + d_x1x2

    return d

def read_graph(filename):

    print 'Reading SLAM solver output file'

    data_array = np.matrix([[0, 0, 0]])

    # ope file to read
    with open(filename, 'r') as fh:

        for line in fh:

            try:
                data_type, nodeID, x, y, theta = line.split()

                if data_type == "VERTEX2" or data_type == "VERTEX_SE2":

                    heading = normalize_angle_to_pi_range(float(theta))

                    data_array = np.append(data_array, [[float(x),float(y),heading]], axis=0)

            except:

                break

    return data_array

if __name__ == '__main__':
    
    print 'Plotting SLAM output.'

    robot_traj = read_graph('output.graph')

    fig1 = plt.figure()

    fig1.suptitle('Estimated Robot Trajectory', fontsize=15)

    ax1 = fig1.add_subplot(111)

    ax1.plot(robot_traj[:,0], robot_traj[:,1], label="Estimate")

    ax1.legend(bbox_to_anchor=(0., 1.0, 1., 0.05), loc=3, ncol=3, mode="expand", borderaxespad=0.)

    # draw arrow in direction of robot heading
    # for i in range(0, robot_traj.shape[0]):
    #     ax1.arrow(robot_traj[i,0], robot_traj[i,1], ARROW_LENGTH * m.cos(robot_traj[i,2]), ARROW_LENGTH * m.sin(robot_traj[i,2]), head_width=0.05, head_length=0.1, fc='k', ec='k')

    plt.ylabel('Y (m)')

    plt.xlabel('X (m)')

    print 'Trajectory Length = ', compute_trajectory_length(robot_traj), ' meters \n'

    plt.show()

    print 'Done!'