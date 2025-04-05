# Segmenting Collision Avoidance
_Linden Adamson_

![Alt Text](https://github.com/LindenAdamson/SegmentingCollisionAvoidance/blob/main/scripts/gifs/cd_rt.gif)

## Overview
Segmenting Collision Avoidance is a work-in-progress algorithm for self driving vehicles. It is designed to process RGB-D image data from a stereo camera and return safe trajectories for the ego vehicle to follow by predicting the movement of surrounding objects. It is meant to be paired with another autonomous vehicle model handling higher-level tasks such as following road laws and choosing when to turn, while SCA provides low-level driving adjustments and emergency braking to ensure safety.

The algorithm uses Ultralitic's FastSAM model as a first step. The dataset used in the demo was captured in CARLA Simulator.

Demoing the algorithm's current functionality can be done in [Colab](https://colab.research.google.com/drive/1RuYl2oYogi_rdPMVeIv6bfQhAwWUqGbg?usp=sharing).
## Functionality
### Complete
- Isolation of tracked objects' placement in 3D space from background, other noise
- Predict tracked objects' future movements
### In Progress
- Prediction improvements
- Determine safe trajectories for the ego vehicle
### To Come
- Utilization of GPU for real-time use case feasibility
### In Consideration
- Support for depth information in point cloud format, eg from a LiDAR device 
## About Bike for the Blind
I am developing Segmenting Collision Avoidance for my "Bike for the Blind" senior capstone project. Dan Parker, a blind race car driver and our client, has asked my team for a self steering bicycle he can use for exercise and getting around his neighborhood. My team is outfitting a velomobile with motorized steering and braking, as well as building computer vision models to transform the bike into an autonomous vehicle. 
