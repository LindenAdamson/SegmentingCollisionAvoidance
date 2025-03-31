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

def timeit(func):
    def wrapper(*args, **kwargs):
        Debug_Timer.start(func.__name__)
        result = func(*args, **kwargs)
        Debug_Timer.stop(func.__name__)
        return result
    return wrapper

class Segmentation_Collision_Avoidance:
    def __init__(self, config):
        Config.load(config)
        self.predictor = Predictor()
    
    @timeit
    def add_CARLA_image_file(self, imgName):
        Debug_Timer.start("open_img")
        rgbImg = Image.open("../data/rgb" + imgName + ".jpeg")
        depthImg = Image.fromarray(np.loadtxt("../data/depth" + imgName + ".out", delimiter=","))
        Debug_Timer.stop("open_img")
        self.resize_images(rgbImg, depthImg)
        
    @timeit
    def add_OAKD_image_file(self, imgName): # TODO: better match rgb and stereo w/ FOV and image sizes in account
        Debug_Timer.start("open_img")
        rgbImg = Image.open("../oakd_data/test/rgb" + imgName + ".png")
        depthImg = np.asarray(Image.open("../oakd_data/test/depth" + imgName + ".png"))[:,:,0].astype(float)
        depthImg = -5.417 * depthImg / 100 + 9.125 # estimation
        depthImg = Image.fromarray(depthImg)
        Debug_Timer.stop("open_img")
        self.resize_images(rgbImg, depthImg)

    def resize_images(self, rgbImg, depthImg):
        # to same FOV
        rgbImg = np.asarray(rgbImg)
        depthImg = np.asarray(depthImg)
        rgb_shape_h, rgb_shape_v = rgbImg.shape[0:2]
        depth_shape_h, depth_shape_v = depthImg.shape
        rgb_fov_h, rgb_fov_v = Config.get("color_fov")
        depth_fov_h, depth_fov_v = Config.get("depth_fov")
        rgb_horiz_diff = int(rgb_shape_h * (1 - rgb_fov_h / max(rgb_fov_h, depth_fov_h)) / 2)
        rgb_vert_diff = int(rgb_shape_v * (1 - rgb_fov_v / max(rgb_fov_v, depth_fov_v)) / 2)
        depth_horiz_diff = int(depth_shape_h * (1 - depth_fov_h / max(rgb_fov_h, depth_fov_h)) / 2)
        depth_vert_diff = int(depth_shape_v * (1 - depth_fov_v / max(rgb_fov_v, depth_fov_v)) / 2)
        if(rgb_horiz_diff > 0):
            rgbImg = rgbImg[rgb_horiz_diff:-rgb_horiz_diff,:,:]
        if(rgb_vert_diff > 0):
            rgbImg = rgbImg[:,rgb_vert_diff:-rgb_vert_diff,:]
        if(depth_horiz_diff > 0):
            depthImg = depthImg[depth_horiz_diff:-depth_horiz_diff,:]
        if(depth_vert_diff > 0):
            depthImg = depthImg[:,depth_vert_diff:-depth_vert_diff]
        # to same size
        h, v = Config.get("dimensions")
        rgbImg = np.asarray(Image.fromarray(rgbImg).resize((v, h)))
        depthImg = np.asarray(Image.fromarray(depthImg).resize((v, h)))
        self.predictor.receive_img(rgbImg, depthImg)

class Predictor:
    def __init__(self):
        self.window = Window()

    def receive_img(self, rgbImg, depthImg):
        self.window.receive_img(rgbImg, depthImg)

class Window:
    def __init__(self):
        self.tracking_imgs = Config.get("tracking_imgs")
        self.model = FastSAM("FastSAM-s.pt")
        # self.model.to('cuda') 2080 has drivers that are too old lol
        self.frames = []

    def receive_img(self, rgbImg, depthImg): # TODO: reshape imgs to same size
        Debug_Timer.start("fastSAM")
        results = self.model.track(rgbImg, persist=True)[0]
        Debug_Timer.stop("fastSAM")
        frame = Frame(rgbImg, depthImg, results)
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
                        d[object.id] += 1
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
    def __init__(self, rgbImg, depthImg, results):
        self.rgbImg = rgbImg
        self.depthImg = depthImg
        # self.clean_images()
        self.results = results
        self.rows, self.cols = Config.get("dimensions")
        self.objects = self.populate_objects()
        self.clean_objects()
        self.cartImg = self.make_cartesian()
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
            obj = Object(id, box, outline)
            objects.append(obj)
        return objects
    
    @timeit
    def clean_objects(self):
        img_size = Config.get("dimensions")[0] * Config.get("dimensions")[1]
        min_area = Config.get("min_object_area")
        sky_bright_percent = Config.get("sky_brightness_percent")
        max_percent_sky_overlap = Config.get("max_percent_sky_overlap")
        rgb_sum = np.sum(self.rgbImg, axis=2)
        max_bright = np.max(rgb_sum)
        bright_cutoff = max_bright * (1 - sky_bright_percent)
        sky = np.where(rgb_sum > bright_cutoff, 1, 0)
        for obj in self.objects:
            if float(obj.area) / float(img_size) < min_area: # object area too small to care
                obj.set_out_of_scope()
            if np.sum(np.logical_and(obj.segMask, sky)) / obj.area > max_percent_sky_overlap: # object is the sky
                obj.set_out_of_scope()

    @timeit
    def make_cartesian(self):
        depth_sphe = np.empty([self.rows, self.cols, 3]) # theta = left/right, phi = up/down, rho = distance        
        depth_cart = np.empty([self.rows, self.cols, 3]) # x = left/right, y = up/down, z = in/out
        horiz_fov =  min(Config.get("color_fov")[0], Config.get("depth_fov")[0])
        vert_fov =  min(Config.get("color_fov")[1], Config.get("depth_fov")[1])
        horiz_fov *= math.pi / 180
        vert_fov *= math.pi / 180
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
    def __init__(self, id, box, outline):
        self.id = id
        self.box = box
        self.in_scope = True
        self.segMask = self.get_seg_mask(outline)
        self.area = np.sum(self.segMask)

    def get_seg_mask(self, outline):
        if outline.shape[0] < 3:
            self.set_out_of_scope()
            return
        segMask = np.zeros(Config.get("dimensions"))
        polygon = Polygon(outline)
        points = np.array([[[x, y] for x, y in zip(*polygon.boundary.coords.xy)]]).astype(int)
        segMask = cv2.fillPoly(segMask, points, color=1).astype(bool)
        return segMask

    def set_out_of_scope(self):
        self.in_scope = False

class Debug_Timer:
    start_time = {}
    samples = {}
    total = {}
    maxi = {}
    mini = {}

    @classmethod
    def start(cls, topic: str):
        cls.start_time[topic] = timer()

    @classmethod
    def stop(cls, topic: str):
        time = timer() - cls.start_time[topic]
        if topic in cls.samples:
            cls.samples[topic] += 1
            cls.total[topic] += time
            cls.maxi[topic] = max(cls.maxi[topic], time)
            cls.mini[topic] = min(cls.mini[topic], time)
        else:
            cls.samples[topic] = 1
            cls.total[topic] = time
            cls.maxi[topic] = time
            cls.mini[topic] = time

    @classmethod
    def print_all(cls):
        for key in cls.samples:
            print(key + ":")
            print("\tavg = " + str(cls.total[key] / cls.samples[key])\
                + ", min = " + str(cls.mini[key])\
                + ", max = " + str(cls.maxi[key]))
            print("\ttotal = " + str(cls.total[key])\
                + ", " + str(cls.samples[key]) + " samples")

class Config:
    c = None

    @classmethod
    def load(cls, name):
        with open(name + ".yaml") as f:
            cls.c = yaml.load(f, Loader=yaml.FullLoader)

    @classmethod
    def get(cls, name):
        return cls.c[name]