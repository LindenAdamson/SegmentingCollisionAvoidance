import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
import cv2
import yaml
from ultralytics import FastSAM
from timeit import default_timer as timer
from PIL import Image

def timeit(func):
    def wrapper(*args, **kwargs):
        Debug_Timer.start(func.__name__)
        result = func(*args, **kwargs)
        Debug_Timer.stop(func.__name__)
        return result
    return wrapper

class SCA:
    def __init__(self, config):
        Config.load(config)
        self.frame = Frame()
        self.model = FastSAM("FastSAM-s.pt")

    @timeit
    def add_npy_file(self, rgb_file, depth_file, preferred_steering_angle):
        rgbImg = np.load(rgb_file)
        depthImg = np.load(depth_file)
        h, v = Config.get("dimensions")
        rgbImg = np.copy(np.asarray(Image.fromarray(rgbImg[:,:,::-1]).resize((v, h))))
        depthImg = np.copy(np.asarray(Image.fromarray(depthImg).resize((v, h)), dtype=np.float32))
        depthImg = depthImg / 1000. # cm to m
        # return self.window.receive_img(rgbImg, depthImg, preferred_steering_angle)
        results = self.model.track(rgbImg, persist=True, verbose=False)[0]
        return self.frame.new_data(rgbImg, depthImg, preferred_steering_angle, results)

class Frame:
    def __init__(self):
        self.entities = Entities()
        self.rows, self.cols = Config.get("dimensions")
        self.threed = np.zeros((self.rows, self.cols, 3), dtype=np.float32)
        self.consolidation_grid = np.empty((self.rows, self.cols), dtype=int)
        self.least_sqares_idxs = np.empty((self.rows, self.cols))
        lin = np.linspace(0, self.cols - 1, self.cols)
        for i in range(self.rows):
            self.least_sqares_idxs[i,:] = lin
        self.least_sqares_idxs = self.least_sqares_idxs
        horiz_fov = min(Config.get("color_fov")[0], Config.get("depth_fov")[0]) * math.pi / 180
        vert_fov = min(Config.get("color_fov")[1], Config.get("depth_fov")[1]) * math.pi / 180
        self.lr = np.linspace(-horiz_fov / 2, horiz_fov / 2, self.cols)
        self.ud = np.linspace(vert_fov / 2, -vert_fov / 2, self.rows)
        max_angle = Config.get("max_steering_angle")
        angle_samples = Config.get("angle_samples")
        self.angle_steps = np.linspace(-max_angle, max_angle, angle_samples)

    @timeit
    def new_data(self, rgb, depth, prefd_angle, results):
        self.rgb = rgb
        self.depth = depth
        self.make_threed()
        self.populate_objects(results)
        self.consolidate_objects()
        self.filter_objects()
        self.fit_least_squares()
        arcs = self.get_arcs_fit_lines()
        return self.find_best_steering_angle(arcs, prefd_angle)

    @timeit
    def make_threed(self):
        divisor = 1 / (45 * math.pi / 180)
        camera_height = Config.get("camera_vertical_height")
        for i in range(self.rows):
            self.threed[i,:,0] = self.depth[i,:] * self.lr * divisor
        for i in range(self.cols):
            self.threed[:,i,1] = self.depth[:,i] * self.ud * divisor + camera_height
        self.threed[:,:,2] = self.depth

    @timeit
    def populate_objects(self, results):
        self.entities.reset_scopes()
        outlines = results.masks.xy
        for i, outline in enumerate(outlines):
            mask = self.entities.get_mask(i)
            tempmask = np.zeros((self.rows, self.cols))
            if outline.shape[0] >= 3:
                tempmask = cv2.fillPoly(tempmask, [outline.astype(int)], color=1)
                mask[:] = tempmask.astype(bool)
            else:
                self.entities.set_out_scope(i)
        for i in range(outlines.__len__(), Config.get("max_possible_entities")):
            self.entities.set_out_scope(i)
        self.entities.codify_scopes(True)

    @timeit
    def consolidate_objects(self):
        areas = self.entities.get_areas_vector()
        argsort = areas.argsort()[::-1] # largest to smallest
        self.consolidation_grid[:] = -1
        for i in argsort:
            mask = self.entities.get_mask(i)
            d = {}
            section = self.consolidation_grid[mask]
            for j in range(section.size):
                other_idx = section[j]
                if other_idx == -1:
                    continue
                if other_idx not in d:
                    d[other_idx] = 1
                else:
                    d[other_idx] += 1
            for idx in d.keys():
                other_mask = self.entities.get_mask(idx)
                if d[idx] / self.entities.get_area(i) > Config.get("object_overlap_percent"):
                    other_mask[:] = np.logical_or(mask, other_mask)
                    self.entities.set_out_scope(i)
            self.consolidation_grid[np.logical_and(self.consolidation_grid == -1, mask)] = i
        self.entities.codify_scopes(True)
    
    @timeit
    def filter_objects(self):
        img_size = self.rows * self.cols
        max_in_scope_entity_height = Config.get("max_in_scope_entity_height")
        min_object_in_scope_percent = Config.get("min_object_in_scope_percent")
        min_object_area = Config.get("min_object_area")
        too_far_to_care = Config.get("too_far_to_care")
        bottom = Config.get("bottom_idxs_for_ground")
        too_far = np.where(np.logical_or(self.depth > too_far_to_care, self.depth == 0), 0, 1)
        too_high = self.threed[:,:,1] < max_in_scope_entity_height
        for i in range(self.entities.number):
            mask = self.entities.get_mask(i)
            mask[:] = mask & too_far & too_high
            area = np.sum(mask)
            if area / img_size < min_object_area:
                self.entities.set_out_scope(i)
            if area / self.entities.get_area(i) < min_object_in_scope_percent:
                self.entities.set_out_scope(i)
        maxi = 0
        id = -1
        for i in range(self.entities.number):
            sumi = np.sum(self.entities.get_mask(i)[-bottom:-1,:])
            if sumi > maxi:
                maxi = sumi
                id = i
        self.entities.set_out_scope(id)
        self.entities.codify_scopes(True)

    @timeit
    def fit_least_squares(self):
        stds = Config.get("least_squares_outlier_stds")
        sin_lr = np.sin(self.lr)
        for i in range(self.entities.number):
            mask = self.entities.get_mask(i)
            x = self.threed[mask,0].flatten()
            y = self.threed[mask,2].flatten()
            y = np.where(mask, self.threed[:,:,2], np.nan).flatten()
            x = np.where(mask, self.least_sqares_idxs, np.nan).flatten()
            for a in [x,y]:
                mean = np.nanmean(a)
                std = np.nanstd(a)
                lower_bound = mean - std * stds
                upper_bound = mean + std * stds
                filter = np.logical_or(a < lower_bound, a > upper_bound)
                x[filter] = np.nan
                y[filter] = np.nan
            n = x.size - np.sum(np.isnan(x))
            sx = np.nansum(x)
            sy = np.nansum(y)
            sxy = np.nansum(x * y)
            sxd = np.nansum(np.power(x, 2))
            m = (n * sxy - sx * sy) / (n * sxd - np.power(sx, 2))
            b = (sy - m * sx) / n
            mini = np.nanmin(x)
            maxi = np.nanmax(x)
            depth_line = np.array([[mini, m * mini + b], [maxi, m * maxi + b]])
            if np.sum(np.isnan(depth_line)):
                self.entities.set_out_scope(i)
            for j in range(2):
                depth_line[j,0] = depth_line[j,1] * sin_lr[depth_line[j,0].astype(int)]
            box = self.entities.get_box(i)
            box[:] = self.make_safety_box(np.copy(depth_line))
        self.entities.codify_scopes(False)

    @timeit
    def make_safety_box(self, segment):
        ax, ay, bx, by = segment.flatten()
        m = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
        nx = Config.get("safe_distance") * (ax - bx) / m
        ny = Config.get("safe_distance") * (ay - by) / m
        box = np.empty((5,2))
        box[0,0] = ax + nx + ny
        box[1,0] = ax + nx - ny
        box[2,0] = bx - nx - ny
        box[3,0] = bx - nx + ny
        box[4,0] = ax + nx + ny
        box[0,1] = ay + ny - nx
        box[1,1] = ay + ny + nx
        box[2,1] = by - ny + nx
        box[3,1] = by - ny - nx
        box[4,1] = ay + ny - nx
        return box
    
    @timeit
    def get_arcs_fit_lines(self):
        angle_samples = Config.get("angle_samples")
        t = Config.get("seconds_into_future")
        vmax = Config.get("max_considered_velocity")
        arcs = np.full(angle_samples, vmax * t)
        for i in range(self.entities.number):
            box = self.entities.get_box(i)
            for i in range(4):
                segment = box[i:i+2,:]
                try:
                    for i, a in enumerate(self.angle_steps):
                        if a == 0:
                            y = self.y_intercept(segment)
                            if y.__len__() == 1:
                                arcs[i] = min(y[0], arcs[i])
                        else:
                            track = self.make_angle_path_circle(a)
                            options = self.intersect(track, segment)
                            if options.__len__() == 0:
                                continue
                            elif options.__len__() == 1:
                                arc = self.arc_length(track, options[0])
                            else:
                                arc1 = self.arc_length(track, options[0])
                                arc2 = self.arc_length(track, options[1])
                                if arc1 < arc2:
                                    arc = arc1
                                else:
                                    arc = arc2
                            arcs[i] = min(arc, arcs[i])
                except:
                    pass
        return arcs
    
    @timeit
    def make_angle_path_circle(self, x):
        l = Config.get("length_front_to_back_wheels")
        a = (90 - x) * math.pi / 180
        val = math.tan(a) * (math.sin(a) + l) + math.cos(a)
        return np.array([val,0,abs(val)])

    @timeit
    def intersect(self, circle, segment):
        sys = segment[:,1]
        options = []
        l = np.copy(segment)
        l -= circle[0:2]
        r = circle[2]
        x1, y1, x2, y2 = l.flatten()
        dx = x2 - x1
        dy = y2 - y1
        dr = dx**2 + dy**2
        det = x1 * y2 - x2 * y1
        disc = r**2 * dr - det**2
        if disc < 0:
            pass
        elif disc == 0:
            option = np.array([det * dy, -det * dx])
            if sys[0] >= option[1] >= sys[1] or sys[1] >= option[1] >= sys[0]:
                options.append(option)
        else:
            xpm = np.sign(dy) * dx * math.sqrt(disc)
            ypm = abs(dy) * math.sqrt(disc)
            option1 = np.array([det * dy + xpm, -det * dx + ypm]) / dr + circle[0:2]
            option2 = np.array([det * dy - xpm, -det * dx - ypm]) / dr + circle[0:2]
            if sys[0] >= option1[1] >= sys[1] or sys[1] >= option1[1] >= sys[0]:
                options.append(option1)
            if sys[0] >= option2[1] >= sys[1] or sys[1] >= option2[1] >= sys[0]:
                options.append(option2)
        return options

    @timeit
    def arc_length(self, circle, point):
        rad = -np.atan((point[1] - circle[1]) / (point[0] - circle[0]))
        xdel = point[0] - circle[0] >= 0
        ydel = point[1] - circle[1] >= 0
        if circle[0] > 0:
            if xdel and ydel: # quad 2
                rad += math.pi
            elif xdel and not ydel: # quad 3
                rad += math.pi
            elif not xdel and ydel: # quad 1
                pass
            else: # quad 4 # not xdel and not ydel
                rad += math.pi * 2
        else:
            if xdel and ydel: # quad 2
                rad = -rad
            elif xdel and not ydel: # quad 3
                rad = math.pi * 2 - rad
            elif not xdel and ydel: # quad 1
                rad = math.pi - rad
            else: # quad 4 # not xdel and not ydel
                rad = math.pi -rad
        return rad * circle[2]
    
    @timeit
    def y_intercept(self, segment):
        option = []
        sys = segment[:,1]
        m = (segment[0,1] - segment[1,1]) / (segment[0,0] - segment[1,0])
        y = segment[0,1] - m * segment[0,0]
        if sys[0] >= y >= sys[1] or sys[1] >= y >= sys[0]:
            option.append(y)
        return option

    @timeit
    def find_best_steering_angle(self, arcs, prefd_angle):
        unimpeded_val = Config.get("max_considered_velocity") * Config.get("seconds_into_future")
        max_angle = Config.get("max_steering_angle")
        if np.sum(arcs == unimpeded_val):
            possible_angles = self.angle_steps[arcs == unimpeded_val]
        else:
            possible_angles = self.angle_steps[arcs == np.max(arcs)]
        best_difference = max_angle * 2
        best_angle = max_angle * 2
        for angle in possible_angles:
            dif = abs(prefd_angle - angle)
            if dif < best_difference:
                best_difference = dif
                best_angle = angle
        return best_angle

class Entities:
    def __init__(self):
        rows, cols = Config.get("dimensions")
        self.max_entities = Config.get("max_possible_entities")
        self.masks = np.zeros((rows, cols, self.max_entities), dtype=bool)
        self.areas = np.zeros((self.max_entities), dtype=int)
        self.boxes = np.zeros((5, 2, self.max_entities), dtype=np.float32)

    @timeit
    def reset_scopes(self):
        self.scopes = np.ones((self.max_entities), dtype=bool)
        self.idxs = np.linspace(0, self.max_entities - 1, self.max_entities, dtype=int)
        self.number = self.max_entities
    
    @timeit
    def set_out_scope(self, i):
        i = self.idxs[i]
        self.scopes[i] = 0

    @timeit
    def codify_scopes(self, recalc_area):
        count = 0
        for i in range(self.max_entities):
            if self.scopes[i]:
                self.idxs[count] = i
                count += 1
        self.number = count
        if recalc_area:
            for i in range(count):
                idx = self.idxs[i]
                self.areas[idx] = np.sum(self.masks[:,:,idx])

    @timeit
    def get_mask(self, i):
        idx = self.idxs[i]
        return self.masks[:,:,idx]
    
    @timeit
    def get_areas_vector(self):
        return self.areas[self.scopes]
    
    @timeit
    def get_area(self, i):
        idx = self.idxs[i]
        return self.areas[idx]
    
    @timeit
    def get_box(self, i):
        idx = self.idxs[i]
        return self.boxes[:,:,idx]

class Debug_Timer:
    start_time = {}
    samples = {}
    total = {}
    maxi = {}
    mini = {}
    on = True

    @classmethod
    def start(cls, topic: str):
        if cls.on:
            cls.start_time[topic] = timer()

    @classmethod
    def stop(cls, topic: str):
        if cls.on:
            if topic in cls.start_time:
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
        if cls.on:
            totals = np.array(list(cls.total.values()))
            argsort = totals.argsort()[::-1]
            topics = list(cls.total.keys())
            for i in range(cls.total.__len__()):
                topic = topics[argsort[i]]
                print(topic + ":")
                print("\ttotal = " + str(cls.total[topic])\
                    + ", " + str(cls.samples[topic]) + " samples")
                print("\tavg = " + str(cls.total[topic] / cls.samples[topic])\
                    + ", min = " + str(cls.mini[topic])\
                    + ", max = " + str(cls.maxi[topic]))

    @classmethod
    def print(cls, topic):
        if cls.on and topic in cls.start_time:
            print(topic + ":")
            print("\ttotal = " + str(cls.total[topic])\
                + ", " + str(cls.samples[topic]) + " samples")
            print("\tavg = " + str(cls.total[topic] / cls.samples[topic])\
                + ", min = " + str(cls.mini[topic])\
                + ", max = " + str(cls.maxi[topic]))
            
    @classmethod
    def reset(cls):
        if cls.on:        
            cls.start_time = {}
            cls.samples = {}
            cls.total = {}
            cls.maxi = {}
            cls.mini = {}

    @classmethod
    def turn_off(cls):
        cls.on = False

class Config:
    c = None
    d = {}

    @classmethod
    def load(cls, name):
        with open(name + ".yaml") as f:
            cls.c = yaml.load(f, Loader=yaml.FullLoader)

    @classmethod
    def get(cls, name):
        if name in cls.d:
            return cls.d[name]
        return cls.c[name]
    
    @classmethod
    def define(cls, name, value):
        cls.d[name] = value