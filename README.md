# Segmenting Collision Avoidance
_Handheld OAK-D camera footage, Segmenting Collision Avoidance simulation output:_

![Alt Text](https://github.com/LindenAdamson/SegmentingCollisionAvoidance/blob/main/scripts/gifs/run1.gif)

![Alt Text](https://github.com/LindenAdamson/SegmentingCollisionAvoidance/blob/main/scripts/gifs/run2.gif)

## Overview
Segmenting Collision Avoidance is a work-in-progress script for self driving vehicles. It is designed to process RGB-D image data from a stereo camera and return the safest trajectory for the ego vehicle to follow by calculating the locations of surrounding objects. It is meant to be paired with another autonomous vehicle model handling higher-level tasks such as following road laws and traveling to a destination, while SCA provides low-level driving adjustments to ensure safety.

This script uses depth estimations from the Luxonis OAK-D's built in stereo depth model, as well as Ultralitic's FastSAM model for image segmentation and object tracking.

Demoing the scripts's current functionality can be done in [Colab](https://colab.research.google.com/drive/1RuYl2oYogi_rdPMVeIv6bfQhAwWUqGbg?usp=sharing).
## Functionality
### Complete
- Core functionality
### In Progress
- Integration with OAK-D for real-time use on NVIDIA Jetson Orin Nano
- General optimizations for higher performance
### To Come
- Porting from Python to C++ for significantly increased performance
### In Consideration
- Support for depth information in point cloud format, eg from a LiDAR device 
## About Bike for the Blind
I am developing Segmenting Collision Avoidance for my "Bike for the Blind" senior capstone project. Dan Parker, a blind race car driver and our client, has asked my team for a self steering bicycle he can use for exercise and getting around his neighborhood. My team is outfitting a velomobile with motorized steering and braking, as well as building computer vision models to transform the bike into an autonomous vehicle. 
