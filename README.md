# Segmenting Collision Avoidance
_Linden Adamson_

![Alt Text](https://github.com/LindenAdamson/SegmentingCollisionAvoidance/blob/main/scripts/gifs/cd_rt.gif)

## Overview
Segmenting Collision Avoidance is a work-in-progress script for self driving vehicles. It is designed to process RGB-D image data from a stereo camera and return safe trajectories for the ego vehicle to follow by predicting the movement of surrounding objects. It is meant to be paired with another autonomous vehicle model handling higher-level tasks such as following road laws and choosing when to turn, while SCA provides low-level driving adjustments and emergency braking to ensure safety.

This script uses depth estimations from the Luxonis OAK-D's built in stereo depth model, as well as Ultralitic's FastSAM model for image segmentation and object tracking.

Demoing the scripts's current functionality can be done in [Colab](https://colab.research.google.com/drive/1RuYl2oYogi_rdPMVeIv6bfQhAwWUqGbg?usp=sharing).
## Functionality
### Complete
- Isolation of potentially hazardous objects and determination of their placement in space
- Prediction of tracked objects' future movements
- Parallelization of viable algorithms for GPU utilization
### In Progress
- Testing on data captured with recently motorized velomobile
- Conversion of "permitted velocity" into steering corrections based on this data
### To Come
- Porting of some or all Python code to C++ for real-time use case feasibility on NVIDIA Jetson Orin Nano
### In Consideration
- Support for depth information in point cloud format, eg from a LiDAR device 
## About Bike for the Blind
I am developing Segmenting Collision Avoidance for my "Bike for the Blind" senior capstone project. Dan Parker, a blind race car driver and our client, has asked my team for a self steering bicycle he can use for exercise and getting around his neighborhood. My team is outfitting a velomobile with motorized steering and braking, as well as building computer vision models to transform the bike into an autonomous vehicle. 
