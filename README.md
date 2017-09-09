# py_slam_dataset
Python programs to generate a pose-graph SLAM data set and plot the output

# generate_slam_dataset

A python program that generates a 2D pose graph output file in g2o or toro format (use can choose). It uses a bicycle motion model to drive the robot on some user defined control signal. 
It will generate the loop closure constraints as well.

# plot_graph_file

A python program to plot a SLAM output graph file, the file can be in g2o or toro format.
