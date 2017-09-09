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
import matplotlib
from matplotlib import pyplot as plt
from base_utils import *

'''
Description: This utility simulates a SLAM scenario and generates a TORO / G2O style .graph / .g2o file with relative pose odometry measurements, loop closure relative
pose constraints and a .graph / .g2o ground truth file. It uses a bicycle motion model and gaussian noise model.
'''

### Global Parameters ###
dt = 0.5

wheel_base = 2.0

max_velocity = 5.0  # m/s

max_steering_angle = deg2rad(60)  # radians

max_omega = deg2rad(30)  # rad / s

eta_Wg = np.array([0.02, 0.02]) # the noise scaling factor

P_Wg = np.matrix([[(max_velocity*dt*eta_Wg[0])**2, 0.0, 0.0],
					[0.0, (max_velocity*dt*eta_Wg[0])**2, 0.0],
					[0.0, 0.0, (max_omega*dt*eta_Wg[1])**2]]);  # additive process noise covariance

P_Wg_inv = np.linalg.inv(P_Wg)

NUM_SIM_STEPS = 1000 # number of simulation steps

NUM_LC_CONSTRAINTS = max(5, NUM_SIM_STEPS/100) # number of loop closure constraints

MAX_RANGE_LC = 1.0 # max distance between two poses for loop closure to be valid (i.e., if distance < MAX_RANGE_LC loop closure will be computed for the pair of poses)

MIN_POSE_WINDOW_LC = 30 # MAX_RANGE_LC / (dt * max_velocity)

OUTPUT_FORMAT = 'g2o' # can change to g2o or toro

# convert yaw angle into a rotation matrix
def yaw2rotmat(yaw):

	R = np.matrix([[m.cos(yaw), -m.sin(yaw)], [m.sin(yaw), m.cos(yaw)]])

	return R

# generate process noise
def generate_process_noise():

	mean = np.array([0.0, 0.0, 0.0])

	w = np.random.multivariate_normal(mean, P_Wg)

	w_vec = np.array([[w[0]], [w[1]], [w[2]]])

	return w_vec


# discrete motion model equation
def evolve(x, u):

	global dt, wheel_base

	# v is linear velocity
	v = u[0, 0]

	# g is steering angle
	g = u[1, 0]

	# Change in position in local Coordinate frame of robot
	dp_local_frame = np.array([[v*dt*m.cos(g)], [v*dt*m.sin(g)]])

	# Get rotation matrix
	Rmat = yaw2rotmat(x[2])

	# Initialize the next state
	x_next = np.array([[0.0], [0.0], [0.0]])

	# The noise in local frame gets transformed from local frame to global
	x_next[0:2, :] = x[0:2, :] + np.dot(Rmat, dp_local_frame)

	# Update heading
	x_next[2, 0] = x[2] + v*dt*m.sin(g)/wheel_base

	# Wrap heading
	#x_next[2] = normalize_angle_to_pi_range(x_next[2])

	return x_next


# Generate odometry based on previous state and control using discrete
# motion model equation
def get_odometry(x, u):

	global dt, wheel_base

	# generate odometery noise
	additive_noise = generate_process_noise()

	if u[0] == 0:
		additive_noise = 0*additive_noise

	v = u[0, 0]

	g = u[1, 0]

	# OdoVal should be in local frame
	odoVal = np.array([[v*dt*m.cos(g)], 
					   [v*dt*m.sin(g)],
	                   [v*dt*m.sin(g)/wheel_base]]) + additive_noise

	x_next = np.array([[0.0], [0.0], [0.0]])

	x_next[0:2,:] = x[0:2,:] + np.dot(yaw2rotmat(x[2]), odoVal[0:2,:])

	x_next[2] = x[2] + odoVal[2]

	x_next[2] = normalize_angle_to_pi_range(x_next[2])

	return odoVal, x_next

def generate_loop_closure_constraint(x1, x2):

	additive_noise = generate_process_noise()

	delta_x = x2[0:2,0] - x1[0:2,0]

	R = yaw2rotmat(x1[2])

	C = np.transpose(R) # the direction cosine matrix

	delta_x_l = np.dot(C, delta_x) + additive_noise[0:2,0]

	d_theta = normalize_angle_to_pi_range(x2[2] - x1[2]) + additive_noise[2,0]

	DX = np.array([ [delta_x_l[0,0]], [delta_x_l[0,1]], [d_theta[0]] ])
	
	return DX

def check_loop_closure(curr_pose, x_hist, y_hist, indx):

	# calculate distance between current pose and history
	dx = x_hist[0:indx-MIN_POSE_WINDOW_LC] - curr_pose[0]

	dy = y_hist[0:indx-MIN_POSE_WINDOW_LC] - curr_pose[1]

	dx2 = np.square(dx)

	dy2 = np.square(dy)

	d = np.sqrt(dx2+dy2)

	check = np.greater(MAX_RANGE_LC, d)

	return check

# write slam data in TORO format and add it to buffer
def write_slam_data_formatted(buff, data_type, data_array, data_matrix, edge = (-1,-1)):

	to_write = ' '

	if data_type == 'EDGE2':

		if edge[0] < 0:

			print 'EDGE2 data is not in correct format'
			return

		else:
			to_write = '{} {} {} {} {} {} {} {} {} {} {} {}'.format(data_type, edge[0], edge[1],
						 data_array[0,0], data_array[1,0], data_array[2,0], 
						 data_matrix[0,0], data_matrix[0,1], data_matrix[1,1],
						 data_matrix[2,2], data_matrix[0,2],
						 data_matrix[1,2])
	
	if data_type == 'EDGE_SE2':

		if edge[0] < 0:

			print 'EDGE_SE2 data is not in correct format'
			return

		else:
			to_write = '{} {} {} {} {} {} {} {} {} {} {} {}'.format(data_type, edge[0], edge[1],
						 data_array[0,0], data_array[1,0], data_array[2,0], 
						 data_matrix[0,0], data_matrix[0,1], data_matrix[1,1],
						 data_matrix[1,1], data_matrix[1,2],
						 data_matrix[2,2])

	if data_type == 'VERTEX2' or data_type == 'VERTEX_SE2':

		if edge[1] < 0:

			print 'VERTEX data is not in correct format'
			return

		else:
			to_write = '{} {} {} {} {}'.format(data_type, edge[1],
						 data_array[0,0], data_array[1,0], data_array[2,0])


	to_write = to_write + '\n'

	buff = buff + to_write

	return buff

if __name__ == '__main__':

	if OUTPUT_FORMAT == 'toro':

		print 'Generating a TORO style graph file \n'

	else:

		print 'Generating a G2O style graph file \n'

	# initialize state of robot
	x = np.array([[0.0], [0.0], [0.0]])

	x_odo = np.array([[0.0], [0.0], [0.0]])

	# initialize control
	u = np.array([[1.0], [0.1]])

	steps = NUM_SIM_STEPS

	# initialize containers
	x_dat = np.zeros(steps+1)

	y_dat = np.zeros(steps+1)

	theta_dat = np.zeros(steps+1)

	x_odo_dat = np.zeros(steps+1)

	y_odo_dat = np.zeros(steps+1)

	pose_id = 0

	# open files to write slam data
	if OUTPUT_FORMAT == 'toro':		
		
		fh = open('slam_test_data.graph','w')

		fh_gt = open('slam_test_data_ground_truth.graph','w')

	else:
		fh = open('slam_test_data.g2o','w')

		fh_gt = open('slam_test_data_ground_truth.g2o','w')

	# write data which is starting point
	unary_edge = (-1,0)

	buff_vertex = ''

	buff_vertex_gt = ''

	buff_edge2 = ''

	buff_edge2_lc = ''

	if OUTPUT_FORMAT == 'toro':

		buff_vertex = write_slam_data_formatted(buff_vertex, 'VERTEX2', x_odo, -1, unary_edge) # write state

		buff_vertex_gt = write_slam_data_formatted(buff_vertex_gt, 'VERTEX2', x, -1, unary_edge) # write state

	else:
		
		buff_vertex = write_slam_data_formatted(buff_vertex, 'VERTEX_SE2', x_odo, -1, unary_edge) # write state

		buff_vertex_gt = write_slam_data_formatted(buff_vertex_gt, 'VERTEX_SE2', x, -1, unary_edge) # write state

	num_loop_closures = 0

	distance_travelled = 0

	for i in range(0, steps):

		u[1, 0] = 0.1 *abs(m.sin(pose_id/110))

		# generate odometry
		odo_val, x_odo = get_odometry(x_odo, u)

		edge = (pose_id,pose_id+1)

		unary_edge = (-1,pose_id+1)

		# update true state
		x = evolve(x, u)
		
		if OUTPUT_FORMAT == 'toro':
			# write odometry edge
			buff_edge2 = write_slam_data_formatted(buff_edge2, 'EDGE2', odo_val, np.linalg.inv(P_Wg), edge)

			# write odometry vertex
			buff_vertex = write_slam_data_formatted(buff_vertex, 'VERTEX2', x_odo, -1, unary_edge)

			buff_vertex_gt = write_slam_data_formatted(buff_vertex_gt, 'VERTEX2', x, -1, unary_edge) # write state

		else:
			buff_edge2 = write_slam_data_formatted(buff_edge2, 'EDGE_SE2', odo_val, np.linalg.inv(P_Wg), edge)

			buff_vertex = write_slam_data_formatted(buff_vertex, 'VERTEX_SE2', x_odo, -1, unary_edge)

			buff_vertex_gt = write_slam_data_formatted(buff_vertex_gt, 'VERTEX_SE2', x, -1, unary_edge) # write state

		# Check for Loop Closure
		if pose_id > MIN_POSE_WINDOW_LC:
			 
			lc_check = check_loop_closure(x, x_dat, y_dat, pose_id)

			check_pose = 0

			for check in lc_check:

				if check == True:

					num_loop_closures = num_loop_closures + 1

					edge_lc = (check_pose, pose_id+1)

					#print 'have a loop closure', edge_lc

					lc_constraint = generate_loop_closure_constraint(np.array([ [ x_dat[ check_pose ] ], [ y_dat[ check_pose ] ], [theta_dat[ check_pose ]] ]), x )

					if OUTPUT_FORMAT == 'toro':
					
						buff_edge2_lc = write_slam_data_formatted(buff_edge2_lc, 'EDGE2', lc_constraint, P_Wg_inv, edge_lc)
					
					else:
					
						buff_edge2_lc = write_slam_data_formatted(buff_edge2_lc, 'EDGE_SE2', lc_constraint, P_Wg_inv, edge_lc)

				check_pose = check_pose + 1


		# add data to containers
		x_dat[pose_id+1] = x[0]

		y_dat[pose_id+1] = x[1]

		theta_dat[pose_id+1] = x[2]

		x_odo_dat[pose_id+1] = x_odo[0]

		y_odo_dat[pose_id+1] = x_odo[1]

		distance_travelled = distance_travelled + m.sqrt( (x[0] - x_dat[pose_id])**2 + (x[1] - y_dat[pose_id])**2 ) 

		pose_id += 1

	print 'Number of loop closures = ', num_loop_closures

	print 'Distance travelled = ', distance_travelled 

	# write data and close file
	fh.write(buff_vertex)
	fh.write(buff_edge2)
	fh.write(buff_edge2_lc)
	fh.close()

	# write ground truth and close file
	fh_gt.write(buff_vertex_gt)
	fh_gt.close()

	plt.plot(x_dat, y_dat, label="Ground Truth")

	plt.plot(x_odo_dat, y_odo_dat, label="Odometry")

	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
	
	plt.show()