import cv2
import numpy as np
import argparse
import os
from time import sleep
import datetime
from PIL import Image
from SCA_classes import Segmentation_Collision_Avoidance, Debug_Timer, Config
import matplotlib.pyplot as plt
import multiprocessing
import threading
import socket
import struct
import select
from emergency_brake import Emergency_Brake

HOST = "127.0.0.1"
PORT = 65432

last_rectif_right = None
right_intrinsic = [[860.0, 0.0, 640.0],
                   [0.0, 860.0, 360.0],
                   [0.0, 0.0, 1.0]]
lrcheck   = True      # Better handling for occlusions
extended  = False     # Closer-in minimum depth, disparity range is doubled
subpixel  = False     # Better accuracy for longer distance, fractional disparity
pcl_converter = None  # (Set this if you have a point cloud visualizer)

sca = Segmentation_Collision_Avoidance("config")

noise_thresh = Config.get("noise_thresh")
kblur = Config.get("kblur")
safe_cm = Config.get("safe_cm")
boxes = Config.get("boxes")
box_ring = Config.get("box_ring")
ebrake = Emergency_Brake(noise_thresh, kblur, safe_cm, boxes, box_ring)

def convert_to_cv2_frame(name, image):
    data = image.getData()
    w = image.getWidth()
    h = image.getHeight()
    rows, cols = Config.get("dimensions")

    if name == "rgb":
        yuv = np.array(data).reshape((h * 3 // 2, w)).astype(np.uint8)
        frame = np.array(cv2.resize(cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12), (cols, rows)))
        return frame

    global last_rectif_right
    baseline = 75  # mm (example value)
    focal = right_intrinsic[0][0]
    max_disp = 96
    disp_type = np.uint8
    disp_levels = 1
    if extended:
        max_disp *= 2
    if subpixel:
        max_disp *= 32
        disp_type = np.uint16
        disp_levels = 32

    if name in ["depth", "bad_depth"]:
        disp = np.array(data).astype(np.uint8).view(disp_type).reshape((h, w))
        with np.errstate(divide='ignore'):
            frame = (disp_levels * baseline * focal / disp).astype(np.uint16)
        # frame = frame[:,40:] # for whatever reason, there is extra fov on the left??
        frame = np.asarray(Image.fromarray(frame).resize((cols, rows)))
        return frame
    
    # if name == "bad_depth":
    #     disp = np.array(data).astype(np.uint8).view(disp_type).reshape((h, w))
    #     frame = np.asarray(Image.fromarray(disp).resize((cols, rows)))
    #     return frame
    
def timeDeltaToMilliS(delta) -> float:
    return delta.total_seconds() * 1000

def oakd(rgbd_queue, bad_depth_queue, turn_off_queue):
    import depthai as dai

    while True:

        # Cant be in a separate func bc of weird depthai and multiprocessing interaction
        median    = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7
        pipeline = dai.Pipeline()

        color_cam = pipeline.create(dai.node.ColorCamera)
        color_cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        color_cam.setInterleaved(False)
        color_cam.setBoardSocket(dai.CameraBoardSocket.RGB)

        xout_rgb_video = pipeline.create(dai.node.XLinkOut)
        xout_rgb_video.setStreamName("rgb")

        color_cam.video.link(xout_rgb_video.input)
        
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)
        mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setConfidenceThreshold(200)
        stereo.setRectifyEdgeFillColor(0)
        stereo.initialConfig.setMedianFilter(median)
        stereo.setLeftRightCheck(lrcheck)
        stereo.setExtendedDisparity(extended)
        stereo.setSubpixel(subpixel)

        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)
        
        xout_left         = pipeline.create(dai.node.XLinkOut)
        xout_right        = pipeline.create(dai.node.XLinkOut)
        xout_disparity    = pipeline.create(dai.node.XLinkOut)
        xout_rectif_left  = pipeline.create(dai.node.XLinkOut)
        xout_rectif_right = pipeline.create(dai.node.XLinkOut)
        xout_left.setStreamName("left")
        xout_right.setStreamName("right")
        xout_disparity.setStreamName("depth")
        xout_rectif_left.setStreamName("rectified_left")
        xout_rectif_right.setStreamName("rectified_right")
        
        bad_stereo = pipeline.create(dai.node.StereoDepth)
        mono_left.out.link(bad_stereo.left)
        mono_right.out.link(bad_stereo.right)
        xout_bad_disparity    = pipeline.create(dai.node.XLinkOut)
        xout_bad_disparity.setStreamName("bad_depth")
        bad_stereo.disparity.link(xout_bad_disparity.input)

        stereo.syncedLeft.link(xout_left.input)
        stereo.syncedRight.link(xout_right.input)
        stereo.disparity.link(xout_disparity.input)
        stereo.rectifiedLeft.link(xout_rectif_left.input)
        stereo.rectifiedRight.link(xout_rectif_right.input)

        streams = ["rgb", "depth", "bad_depth", "left", "right", "rectified_left", "rectified_right"]

        with dai.Device(pipeline) as device:
            output_queues = {s: device.getOutputQueue(name=s, maxSize=4, blocking=False) for s in streams}

            on = False
            rgb = None
            depth = None
            while True:
                try:
                    if not turn_off_queue.empty(): 
                        msg = int(turn_off_queue.get())
                        print(msg)
                        print("seen")
                        on = bool(msg)
                        print(on)
                        rgb = None
                        depth = None
                    rgbframe = output_queues["rgb"].tryGet()
                    depthframe = output_queues["depth"].tryGet()
                    bad_depthframe = output_queues["bad_depth"].tryGet()
                    if on:
                        if rgbframe is not None:
                            rgb = convert_to_cv2_frame('rgb', rgbframe)
                        if depthframe is not None:
                            depth = convert_to_cv2_frame('depth', depthframe)
                        if rgb is not None and depth is not None:
                            rgbd_queue.put([rgb, depth]) 
                            rgb = None
                            depth = None
                        if bad_depthframe is not None:
                            depth = convert_to_cv2_frame('bad_depth', bad_depthframe)
                            bad_depth_queue.put(depth)
                except RuntimeError:
                    print("runtime error")
                    break

def run_sca(rgbd_queue, steering_queue):
    while True:
        try:
            while True:
                if not rgbd_queue.empty(): 
                    rgb, depth = rgbd_queue.get()
                    steer = sca.add_np_array(np.copy(rgb), np.copy(depth), 0)
                    if steer is not None:
                        print(steer)
                        push_steer = "s" + str(steer)
                        steering_queue.put(push_steer)
        except:
            pass

def run_emergency_brake(bad_depth_queue, braking_queue):
    while True:
        try:
            while True:
                if not bad_depth_queue.empty(): 
                    depth = bad_depth_queue.get()
                    brake_val = ebrake.get_brake(depth)
                    print(brake_val)
                    push_brake = "b" + str(brake_val)
                    braking_queue.put(push_brake)
        except:
            pass

def send_socket(steering_queue, braking_queue, turn_off_queue):

    def send(s, steering_queue, braking_queue):
        while True:
            if not steering_queue.empty(): 
                msg = steering_queue.get()
                s.sendall(msg.encode())
            if not braking_queue.empty(): 
                msg = braking_queue.get()
                s.sendall(msg.encode())

    def receive(s, turn_off_queue):
        while True:
            data = s.recv(1024).decode()
            print("from server: " + data)
            turn_off_queue.put(data)

    while True:
        print("waiting for socket connect")
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((HOST, PORT))
                t1 = threading.Thread(target=send, args=(s, steering_queue, braking_queue,))
                t2 = threading.Thread(target=receive, args=(s, turn_off_queue,))
                t1.start()
                t2.start()
                t1.join()
                t2.join()
        except (ConnectionRefusedError, BrokenPipeError): # get rid
            pass

def main():
    rgbd_queue = multiprocessing.Queue()
    bad_depth_queue = multiprocessing.Queue()
    steering_queue = multiprocessing.Queue()
    braking_queue = multiprocessing.Queue()
    turn_off_queue = multiprocessing.Queue()

    p1 = multiprocessing.Process(target=oakd, args=(rgbd_queue, bad_depth_queue, turn_off_queue))
    p2 = multiprocessing.Process(target=run_sca, args=(rgbd_queue, steering_queue,))
    p3 = multiprocessing.Process(target=run_emergency_brake, args=(bad_depth_queue, braking_queue,))
    p4 = multiprocessing.Process(target=send_socket, args=(steering_queue, braking_queue, turn_off_queue))

    p1.start()
    p2.start()
    p3.start()
    p4.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()

if __name__ == '__main__':
    main()
