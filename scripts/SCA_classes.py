import numba_functs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
import cv2
import time
import yaml
from ultralytics import FastSAM
from collections import defaultdict
from shapely.geometry import Point, Polygon
from numba import njit, prange
from timeit import default_timer as timer
from PIL import Image

class Segmentation_Collision_Avoidance:
    def __init__(self, config, dt):
        self.config = config
        self.dt = dt
        self.predictor = Predictor(config, self.dt)
    
    def add_CARLA_image_file(self, imgName):
        self.dt.start("open_img")
        rgbImg = Image.open("../data/rgb" + imgName + ".jpeg")
        depthImg = Image.fromarray(np.loadtxt("../data/depth" + imgName + ".out", delimiter=","))
        self.dt.stop("open_img")
        self.resize_images(rgbImg, depthImg)
        
    def add_OAKD_image_file(self, imgName): # TODO: better match rgb and stereo w/ FOV and image sizes in account
        self.dt.start("open_img")
        rgbImg = Image.open("../oakd_data/test/rgb" + imgName + ".png")
        depthImg = np.asarray(Image.open("../oakd_data/test/depth" + imgName + ".png"))[:,:,0].astype(float)
        depthImg = -5.417 * depthImg / 100 + 9.125 # estimation
        depthImg = Image.fromarray(depthImg)
        self.dt.stop("open_img")
        self.resize_images(rgbImg, depthImg)

    def resize_images(self, rgbImg, depthImg):
        # to same FOV
        rgbImg = np.asarray(rgbImg)
        depthImg = np.asarray(depthImg)
        rgb_shape_h, rgb_shape_v = rgbImg.shape[0:2]
        depth_shape_h, depth_shape_v = depthImg.shape
        c_fov_h, c_fov_v = self.config["color_fov"]
        d_fov_h, d_fov_v = self.config["depth_fov"]
        c_h_l = int(rgb_shape_h * (1 - c_fov_h / max(c_fov_h, d_fov_h)) / 2)
        c_v_l = int(rgb_shape_v * (1 - c_fov_v / max(c_fov_v, d_fov_v)) / 2)
        d_h_l = int(depth_shape_h * (1 - d_fov_h / max(c_fov_h, d_fov_h)) / 2)
        d_v_l = int(depth_shape_v * (1 - d_fov_v / max(c_fov_v, d_fov_v)) / 2)
        if(c_h_l > 0):
            rgbImg = rgbImg[c_h_l:-c_h_l,:,:]
        if(c_v_l > 0):
            rgbImg = rgbImg[:,c_v_l:-c_v_l,:]
        if(d_h_l > 0):
            depthImg = depthImg[d_h_l:-d_h_l,:]
        if(d_v_l > 0):
            depthImg = depthImg[:,d_v_l:-d_v_l]
        # to same size
        h, v = self.config["dimensions"]
        rgbImg = np.asarray(Image.fromarray(rgbImg).resize((v, h)))
        depthImg = np.asarray(Image.fromarray(depthImg).resize((v, h)))
        self.predictor.receive_img(rgbImg, depthImg)

class Predictor:
    def __init__(self, config, dt):
        self.config = config
        self.dt = dt
        self.window = Window(config, self.dt)

    def receive_img(self, rgbImg, depthImg):
        self.window.receive_img(rgbImg, depthImg)

class Window:
    def __init__(self, config, dt):
        self.config = config
        self.dt = dt
        self.tracking_imgs = self.config["tracking_imgs"]
        self.model = FastSAM("FastSAM-s.pt")
        self.frames = []

    def receive_img(self, rgbImg, depthImg): # TODO: reshape imgs to same size
        self.dt.start("fastSAM")
        results = self.model.track(rgbImg, persist=True)[0]
        self.dt.stop("fastSAM")
        frame = Frame(self.config, rgbImg, depthImg, results, self.dt)
        if len(self.frames) == self.tracking_imgs:
            self.frames.pop(0)
        self.frames.append(frame)
        if len(self.frames) == self.tracking_imgs:
            self.objects_in_scope = self.get_objects_in_scope()

    def get_objects_in_scope(self):
        objects_in_scope = []
        d = {}
        for frame in self.frames:
            for object in frame.objects:
                if object.in_scope:
                    if object.id in d:
                        d[object.id] = d[object.id] + 1
                    else:
                        d[object.id] = 1
        for i_frames, frame in enumerate(self.frames):
            objects_in_scope.append([])
            for i_objects, object in enumerate(frame.objects):
                if object.id in d:
                    if d[object.id] == self.tracking_imgs:
                        objects_in_scope[i_frames].append(object)
        return objects_in_scope

class Frame:
    def __init__(self, config, rgbImg, depthImg, results, dt):
        self.config = config
        self.dt = dt
        self.rgbImg = rgbImg
        self.depthImg = depthImg
        self.clean_images()
        self.results = results
        self.rows, self.cols = self.config["dimensions"]
        self.objects = self.populate_objects()
        self.clean_objects()
        self.dt.start("make_cartesian")
        self.cartImg = self.make_cartesian()
        self.dt.stop("make_cartesian")
        # TODO filter objects

    def clean_images(self):
        # self.depthImg = np.where(np.logical_or(self.depthImg == 9.125, self.depthImg == 8.85415), np.nan, self.depthImg)
        self.depthImg = np.where(self.depthImg == 9.125, np.nan, self.depthImg)

    def populate_objects(self):
        objects = [] # slow?
        ids = np.array(self.results.boxes.id.int().cpu().tolist())
        boxes = self.results.boxes.xywh.cpu()
        outlines = self.results.masks.xy
        for idx, (id, box, outline) in enumerate(zip(ids, boxes, outlines)):
            obj = Object(self.config, id, box, outline, self.dt)
            objects.append(obj)
        return objects
    
    def clean_objects(self):
        img_size = self.config["dimensions"][0] * self.config["dimensions"][1]
        min_area = self.config["min_object_area"]
        for obj in self.objects:
            if float(obj.area) / float(img_size) < min_area:
                obj.set_out_of_scope()

    def make_cartesian(self):
        depth_sphe = np.empty([self.rows, self.cols, 3]) # theta = left/right, phi = up/down, rho = distance        
        depth_cart = np.empty([self.rows, self.cols, 3]) # x = left/right, y = up/down, z = in/out
        horiz_fov, vert_fov = self.config["fov"]
        horiz_fov = horiz_fov * math.pi / 180
        vert_fov = vert_fov * math.pi / 180
        depth_sphe[:,:,0] = np.linspace(-horiz_fov / 2, horiz_fov / 2, num = self.cols)
        depth_sphe[:,:,1] = np.linspace(-vert_fov / 2, vert_fov / 2, num = self.rows)[np.newaxis].T
        depth_sphe[:,:,2] = self.depthImg

        def sphe_to_cart(sphe):
            theta = sphe[0]
            phi = sphe[1]
            rho = sphe[2]
            x = rho * math.sin(theta)
            y = rho * math.sin(phi)
            z = rho
            return [x, -y, z]

        for i in range(self.rows):
            for j in range(self.cols):
                depth_cart[i,j,:] = sphe_to_cart(depth_sphe[i,j,:])
        return depth_cart

class Object:
    def __init__(self, config, id, box, outline, dt):
        self.config = config
        self.dt = dt
        self.id = id
        self.box = box
        self.in_scope = True
        self.segMask = self.get_seg_mask(outline)
        self.area = np.sum(self.segMask)

    def get_seg_mask(self, outline):
        if outline.shape[0] < 3:
            self.set_out_of_scope()
            return
        segMask = np.zeros(self.config["dimensions"])
        polygon = Polygon(outline)
        points = np.array([[[x, y] for x, y in zip(*polygon.boundary.coords.xy)]]).astype(int)
        segMask = cv2.fillPoly(segMask, points, color=1).astype(bool)
        return segMask

    def set_out_of_scope(self):
        self.in_scope = False

class Debug_Timer:
    def __init__(self, debug: bool):
        self.debug = debug
        self.start_time = {}
        self.samples = {}
        self.total = {}
        self.max = {}
        self.min = {}

    def start(self, topic: str):
        if not self.debug:
            return
        self.start_time[topic] = timer()

    def stop(self, topic: str):
        if not self.debug:
            return
        time = timer() - self.start_time[topic]
        if topic in self.samples:
            self.samples[topic] = self.samples[topic] + 1
            self.total[topic] = self.total[topic] + time
            self.max[topic] = max(self.max[topic], time)
            self.min[topic] = min(self.max[topic], time)
        else:
            self.samples[topic] = 1
            self.total[topic] = time
            self.max[topic] = time
            self.min[topic] = time

    def print_all(self):
        if not self.debug:
            return
        for key in self.samples:
            print(key + ":")
            print("\tavg = " + str(self.total[key] / self.samples[key])\
                + ", min = " + str(self.min[key])\
                + ", max = " + str(self.max[key]))
            print("\ttotal = " + str(self.total[key])\
                + ", " + str(self.samples[key]) + " samples")