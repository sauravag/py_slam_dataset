#*********************************************************************
#
#  Copyright (c) 2017, Saurav Agarwal.
#  All rights reserved.
#
#*********************************************************************

# Authors: Saurav Agarwal

import numpy as np
import math as m

# normalize angle to between -pi to pi
def normalize_angle_to_pi_range(theta):

    while (theta > m.pi):

        theta -= 2 * m.pi

    while (theta < -m.pi):

        theta += 2 * m.pi

    return theta

# convert degrees to radian
def deg2rad(theta):

	return m.pi*theta / 180.0

# convert 
def rad2deg(theta):

	return 180.0 * theta / m.pi

