# Segmenting Collision Avoidance
_Linden Adamson_
## Overview
Segmenting Collision Avoidance is a work-in-progress algorithm for self driving vehicles. It is designed to process RGB-D image data from a stero camera and return safe trajectories for the ego vehicle to follow by predicting the movement of surrounding objects. It is meant to be paired with another autonomous vehicle model handling higher-level tasks such as following road laws and the chosen route, while SCA provides low-level driving adjustments to ensure safety.

The algorithm uses Ultralitic's FastSAM model as a first step. The dataset used in testing was captured in CARLA Simulator.

Demoing the algorithm's current functionality can be done in [Colab](https://colab.research.google.com/drive/1RuYl2oYogi_rdPMVeIv6bfQhAwWUqGbg?usp=sharing).
## Functionality
### Complete
- Tracking of objects between frames
- Isolation of tracked objects' placement in 3D space from background, other noise
### To Come
- Utilization of GPU to achieve real-time usecase feasibility
- Placement of tracked objects in space where collisions can be detected
- Predict tracked objects' future movements
- Determine safe trajectories for the ego vehicle
### In Consideration
- Support for depth information in point cloud format, eg from a LiDAR device 
## About Bike for the Blind
I am developing Segmenting Collision Avoidance for my "Bike for the Blind" senior capstone project. 
