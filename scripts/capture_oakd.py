import cv2
import depthai as dai
import numpy as np
import argparse
import os
from time import sleep
import datetime
from PIL import Image

run_name = "run4"
capture = True
path = 'output\\' + run_name

# ----------------- Global variables for Depth configuration -----------------
last_rectif_right = None
right_intrinsic = [[860.0, 0.0, 640.0],
                   [0.0, 860.0, 360.0],
                   [0.0, 0.0, 1.0]]
lrcheck   = True      # Better handling for occlusions
extended  = True     # Closer-in minimum depth, disparity range is doubled
subpixel  = True     # Better accuracy for longer distance, fractional disparity
median    = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7
pcl_converter = None  # (Set this if you have a point cloud visualizer)
# -----------------------------------------------------------------------------

def create_combined_pipeline():
    pipeline = dai.Pipeline()

    # ---------------------- Color (RGB) Section ----------------------
    color_cam = pipeline.create(dai.node.ColorCamera)
    color_cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    color_cam.setInterleaved(False)
    color_cam.setBoardSocket(dai.CameraBoardSocket.RGB)

    # Create XLinkOut nodes for the RGB outputs.
    xout_rgb_video = pipeline.create(dai.node.XLinkOut)
    xout_rgb_video.setStreamName("rgb_video")

    # Link ColorCamera outputs.
    color_cam.video.link(xout_rgb_video.input)
    # ------------------------------------------------------------------

    # ---------------------- Stereo / Depth Section --------------------
    # Create MonoCamera nodes for left and right.
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

    # Create the StereoDepth node.
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setConfidenceThreshold(200)
    stereo.setRectifyEdgeFillColor(0)  # Black edges.
    stereo.initialConfig.setMedianFilter(median)
    stereo.setLeftRightCheck(lrcheck)
    stereo.setExtendedDisparity(extended)
    stereo.setSubpixel(subpixel)

    # Link the mono camera outputs to the StereoDepth node.
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    # # Create XLinkOut nodes for the stereo outputs.
    # xout_disparity    = pipeline.create(dai.node.XLinkOut)
    # xout_disparity.setStreamName("disparity")

    # # Link StereoDepth outputs.
    # stereo.disparity.link(xout_disparity.input)

    # Create XLinkOut nodes for the stereo outputs.
    xout_left         = pipeline.create(dai.node.XLinkOut)
    xout_right        = pipeline.create(dai.node.XLinkOut)
    xout_disparity    = pipeline.create(dai.node.XLinkOut)
    xout_rectif_left  = pipeline.create(dai.node.XLinkOut)
    xout_rectif_right = pipeline.create(dai.node.XLinkOut)
    xout_left.setStreamName("left")
    xout_right.setStreamName("right")
    xout_disparity.setStreamName("disparity")
    xout_rectif_left.setStreamName("rectified_left")
    xout_rectif_right.setStreamName("rectified_right")

    # Link StereoDepth outputs.
    stereo.syncedLeft.link(xout_left.input)
    stereo.syncedRight.link(xout_right.input)
    stereo.disparity.link(xout_disparity.input)
    stereo.rectifiedLeft.link(xout_rectif_left.input)
    stereo.rectifiedRight.link(xout_rectif_right.input)

    # IMU
    imu = pipeline.create(dai.node.IMU)
    xlinkOut = pipeline.create(dai.node.XLinkOut)

    xlinkOut.setStreamName("imu")
    imu.enableIMUSensor(dai.IMUSensor.LINEAR_ACCELERATION, 400)
    imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, 400)
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(10)

    imu.out.link(xlinkOut.input)
    # ------------------------------------------------------------------

    # Return all stream names for later retrieval.
    streams = ["rgb_video", "left", "right", "disparity", "rectified_left", "rectified_right", "imu"]
    return pipeline, streams

def convert_to_cv2_frame(name, image):
    data = image.getData()
    w = image.getWidth()
    h = image.getHeight()
    r = (1280 // 2, 720 // 2)

    if name == 'rgb_video':
        # Full-resolution video comes as YUV NV12.
        yuv = np.array(data).reshape((h * 3 // 2, w)).astype(np.uint8)
        frame = np.array(cv2.resize(cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12), r))
        return frame

    # --- For Stereo / Depth streams ---
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

    if name == 'disparity':
        disp = np.array(data).astype(np.uint8).view(disp_type).reshape((h, w))
        with np.errstate(divide='ignore'):
            frame = (disp_levels * baseline * focal / disp).astype(np.uint16)
        # frame = frame[:,40:] # for whatever reason, there is extra fov on the left??
        frame = np.asarray(Image.fromarray(frame).resize(r))
        # frame = np.where(frame == 0, 1000, frame)
        # frame = np.where(frame > 1000, 1000, frame)
        # print(frame[r[0] // 2, r[1] // 2])
        return frame
    
    # if name == 'disparity':
    #     disp = np.array(data).astype(np.uint8).view(disp_type).reshape((h, w))
    #     # Optionally compute depth from disparity:
    #     with np.errstate(divide='ignore'):
    #         depth = (disp_levels * baseline * focal / disp).astype(np.uint16)
    #         frame = (disp * 255. / max_disp).astype(np.uint8)
    #     frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
    #     return frame
    
def timeDeltaToMilliS(delta) -> float:
    return delta.total_seconds() * 1000

def main():

    pipeline, streams = create_combined_pipeline()
    
    # Set the specific output directory
    # base_output_folder = "" #file path of output data
    # os.makedirs(base_output_folder, exist_ok=True)
    if capture:
        imu = open(path + '\\imu.npy', 'wb')
    tsF  = "{:07d}"

    with dai.Device(pipeline) as device:
        output_queues = {s: device.getOutputQueue(name=s, maxSize=4, blocking=False) for s in streams}
        print("Running combined pipeline. Press 'q' to exit.")

        while True:
            for s in streams:
                if s in ["left", "right", "rectified_left", "rectified_right"]:
                    continue
                in_frame = output_queues[s].tryGet()
                if in_frame is not None:
                    ms = timeDeltaToMilliS(in_frame.getTimestampDevice())
                    if s == "imu":
                        packet = in_frame.packets[0]
                        accel = packet.acceleroMeter
                        gyro = packet.gyroscope
                        array = np.array([accel.x, accel.y, accel.z, gyro.x, gyro.y, gyro.z, ms])
                        if capture:
                            np.save(imu, array)
                        # print(array)
                    else:
                        frame = convert_to_cv2_frame(s, in_frame)
                        if frame is not None:
                            cv2.imshow(s, frame)
                            
                            if capture:
                                if s == "disparity":
                                    with open(path + '\\depth_' + tsF.format(int(ms)) + '.npy', 'wb') as depth:
                                        np.save(depth, frame)
                                else:
                                    with open(path + '\\rgb_' + tsF.format(int(ms)) + '.npy', 'wb') as rgb:
                                        np.save(rgb, frame)
                            # print(timeDeltaToMilliS(in_frame.getTimestampDevice()))
                            # shape = frame.shape
                            # print(shape)
                            # print(shape[0] / shape[1])

                            # Initialize counter for this stream if not exists
                            # if s not in frame_counters:
                            #     frame_counters[s] = 0
                            # frame_counters[s] += 1

                            # Save frame
                            # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                            # filename = f"{timestamp}_{s}_{frame_counters[s]:04d}.png"
                            # stream_folder = os.path.join(base_output_folder, s)
                            # os.makedirs(stream_folder, exist_ok=True)
                            
                            # filepath = os.path.join(stream_folder, filename)
                            # cv2.imwrite(filepath, frame)
                # else:
                    # print("a")

            if cv2.waitKey(1) == ord('q'):
                break

        cv2.destroyAllWindows()



if __name__ == '__main__':
    main()