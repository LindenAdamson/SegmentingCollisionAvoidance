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

class Segmentation_Collision_Avoidance:
    def __init__(self, config):
        Config.load(config)
        self.window = Window()
    
    def plot(self):
        fig = plt.figure()
        window = self.window
        frame = window.frame
        plt_colors = list(mcolors.TABLEAU_COLORS.values())
        rows, cols = Config.get("dimensions")
        # fig, axs = plt.subplots(1,2,figsize=(16,9), gridspec_kw={'width_ratios': [2, 1]})
        plt.subplot(2,2,1)
        plt.imshow(frame.rgbImg)
        obj_list = list(window.objects_in_scope)
        for color_idx, obj in enumerate(obj_list):
            outline = np.concatenate((obj.outline, [obj.outline[0,:]]))
            plt.plot(outline[:,0],outline[:,1],color=plt_colors[color_idx % plt_colors.__len__()])#, linewidth=3)
        plt.xlim(0,cols)
        plt.ylim(rows+1,-1)

        plt.subplot(2,2,3)
        plt.imshow(frame.depthImg)

        plt.subplot(1,2,2)
        ax = plt.gcf().gca()

        max_angle = Config.get("max_steering_angle")
        angle_samples = Config.get("angle_samples")
        angle_steps = np.linspace(-max_angle, max_angle, angle_samples)
        # arcs = window.get_arcs_fit_lines()
        arcs = window.arcs
        lab = 'Output Steering Angle: ' + f'{window.best_steering_angle:.2f}' + '°'
        def plot_arc(radius, arc, ax, dotted=False):
            linsty = ':'
            linwid = 3
            color = 'k'
            samples = 100
            unimpeded_length = 20
            for_legend = None
            if radius == 0:
                if arc == 60:
                    if dotted:
                        for_legend, = ax.plot([0, 0], [0, unimpeded_length], c=color,\
                            linestyle=linsty, label=lab, linewidth=linwid)
                    else:
                        ax.plot([0, 0], [0, unimpeded_length], c='g')
                else:
                    if dotted:
                        for_legend, = ax.plot([0, 0], [0, arc], c=color,\
                            linestyle=linsty, label=lab, linewidth=linwid)
                    else:
                        ax.plot([0, 0], [0, arc], c='y')
            elif arc == Config.get("max_considered_velocity") * Config.get("seconds_into_future"):
                x = radius - radius * math.cos(unimpeded_length / radius)
                lin = np.linspace(0,x,samples)
                if dotted:
                    for_legend, = ax.plot(lin, np.sqrt(radius ** 2 - (lin - radius) ** 2), c=color,\
                        linestyle=linsty, label=lab, linewidth=linwid)
                else:
                    ax.plot(lin, np.sqrt(radius ** 2 - (lin - radius) ** 2), c='g')
            else:
                x = radius - radius * math.cos(arc / radius)
                lin = np.linspace(0,x,samples)
                if dotted:
                    for_legend, = ax.plot(lin, np.sqrt(radius ** 2 - (lin - radius) ** 2), c=color,\
                        linestyle=linsty, label=lab, linewidth=linwid)
                else:
                    ax.plot(lin, np.sqrt(radius ** 2 - (lin - radius) ** 2), c='y')
            return for_legend

        for color_idx, obj in enumerate(window.objects_in_scope):
            least_squares_line = obj.least_squares_line
            if least_squares_line is None:
                continue
            box = window.make_safety_box(least_squares_line)
            plt.plot(box[:,0], box[:,1], color=plt_colors[color_idx % plt_colors.__len__()])
        for i, a in enumerate(angle_steps):
            if a == 0:
                plot_arc(0, arcs[i], ax)
                if a == window.best_steering_angle:
                    for_legend = plot_arc(0, arcs[i], ax, True)
            else:
                plot_arc(window.make_angle_path_circle(a)[0], arcs[i], ax)
                if a == window.best_steering_angle:
                    for_legend = plot_arc(window.make_angle_path_circle(a)[0], arcs[i], ax, True)
        width, height = ax.get_figure().get_size_inches()
        xdim = 6
        plt.xlim(-xdim,xdim)
        ydim = xdim / height * width
        plt.ylim(0,2 * ydim)
        plt.legend(handles=[for_legend], loc='upper center')#, prop={'size': 15})
        plt.tight_layout()
        return fig

    # @timeit
    def add_npy_file(self, rgb_file, depth_file, preferred_steering_angle):
        rgbImg = np.load(rgb_file)
        depthImg = np.load(depth_file)
        h, v = Config.get("dimensions")
        rgbImg = np.copy(np.asarray(Image.fromarray(rgbImg[:,:,::-1]).resize((v, h))))
        depthImg = np.copy(np.asarray(Image.fromarray(depthImg).resize((v, h)), dtype=np.float32))
        depthImg = depthImg / 1000. # cm to m
        return self.window.receive_img(rgbImg, depthImg, preferred_steering_angle)
    
    def add_np_array(self, rgb, depth, preferred_steering_angle):
        # h, v = Config.get("dimensions")
        # rgbImg = np.copy(np.asarray(Image.fromarray(rgbImg[:,:,::-1]).resize((v, h))))
        # depthImg = np.copy(np.asarray(Image.fromarray(depthImg).resize((v, h)), dtype=np.float32))
        rgbImg = np.copy(rgb[:,:,::-1])
        depthImg = np.copy(depth.astype(np.float32)) / 1000
        # depthImg = depthImg / 1000. # cm to m
        return self.window.receive_img(rgbImg, depthImg, preferred_steering_angle)

class Window:
    def __init__(self):
        self.model = FastSAM("FastSAM-s.pt")

    @timeit
    def receive_img(self, rgbImg, depthImg, preferred_steering_angle):
        try:
            self.preferred_steering_angle = preferred_steering_angle
            Debug_Timer.start("fastSAM")
            results = self.model.track(rgbImg, persist=True, verbose=False)[0]
            Debug_Timer.stop("fastSAM")
            Debug_Timer.start("frame init")
            self.frame = Frame(rgbImg, depthImg, results)
            Debug_Timer.stop("frame init")
            # Debug_Timer.start("objs and preds")
            self.objects_in_scope = []
            for obj in self.frame.objects:
                if obj.in_scope:
                    self.objects_in_scope.append(obj)
                    self.frame.fit_least_squares(obj)
            self.arcs = self.get_arcs_fit_lines()
            self.best_steering_angle = self.find_best_steering_angle()
            return self.best_steering_angle
        except AttributeError:
            pass

    # @timeit
    def find_best_steering_angle(self):
        unimpeded_val = Config.get("max_considered_velocity") * Config.get("seconds_into_future")
        max_angle = Config.get("max_steering_angle")
        angle_samples = Config.get("angle_samples")
        angle_steps = np.linspace(-max_angle, max_angle, angle_samples)
        if np.sum(self.arcs == unimpeded_val):
            possible_angles = angle_steps[self.arcs == unimpeded_val]
        else:
            possible_angles = angle_steps[self.arcs == np.max(self.arcs)]
        best_difference = max_angle * 2
        best_angle = max_angle * 2
        for angle in possible_angles:
            dif = abs(self.preferred_steering_angle - angle)
            if dif < best_difference:
                best_difference = dif
                best_angle = angle
        return best_angle

    @timeit
    def get_arcs_fit_lines(self):
        max_angle = Config.get("max_steering_angle")
        angle_samples = Config.get("angle_samples")
        l = Config.get("length_front_to_back_wheels")
        t = Config.get("seconds_into_future")
        vmax = Config.get("max_considered_velocity")
        angle_steps = np.linspace(-max_angle, max_angle, angle_samples)
        arcs = np.full(angle_samples, vmax * t)
        for color_idx, obj in enumerate(self.objects_in_scope):
            least_squares_line = obj.least_squares_line
            if least_squares_line is None:
                continue
            box = self.make_safety_box(least_squares_line)
            for i in range(4):
                line = box[i:i+2,:]
                try:
                    for i, a in enumerate(angle_steps):
                        if a == 0:
                            y = self.y_intercept(line)
                            if y.__len__() == 1:
                                arcs[i] = min(y[0], arcs[i])
                        else:
                            track = self.make_angle_path_circle(a)
                            options = self.intersect(track, line)
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
    
    # @timeit
    def make_angle_path_circle(self, x):
        l = Config.get("length_front_to_back_wheels")
        a = (90 - x) * math.pi / 180
        val = math.tan(a) * (math.sin(a) + l) + math.cos(a)
        return np.array([val,0,abs(val)])
    
    # @timeit
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
        
    # @timeit
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
    
    # @timeit
    def y_intercept(self, segment):
        option = []
        sys = segment[:,1]
        m = (segment[0,1] - segment[1,1]) / (segment[0,0] - segment[1,0])
        y = segment[0,1] - m * segment[0,0]
        if sys[0] >= y >= sys[1] or sys[1] >= y >= sys[0]:
            option.append(y)
        return option

class Frame:
    def __init__(self, rgbImg, depthImg, results):
        self.rgbImg = rgbImg
        self.depthImg = depthImg
        self.results = results
        self.rows, self.cols = Config.get("dimensions")
        self.objects = self.populate_objects()
        self.cartImg = self.make_cartesian()
        self.consolidate_objects()
        self.filter_objects()

    @timeit
    def populate_objects(self):
        objects = []
        ids = np.array(self.results.boxes.id.int().cpu().tolist())
        Debug_Timer.start("get_outlines")
        outlines = self.results.masks.xy
        Debug_Timer.stop("get_outlines")
        dimensions = Config.get("dimensions")
        Debug_Timer.start("zip_outlines")
        for id, outline in zip(ids, outlines):
            segMask = np.zeros(dimensions)
            if outline.shape[0] >= 3:
                segMask = cv2.fillPoly(segMask, [outline.astype(int)], color=1)
            obj = Object(id, outline, segMask.astype(bool))
            if outline.shape[0] < 3:
                obj.set_out_of_scope()
            objects.append(obj)
        Debug_Timer.stop("zip_outlines")
        return objects

    @timeit
    def filter_objects(self):
        img_size = Config.get("dimensions")[0] * Config.get("dimensions")[1]
        max_in_scope_entity_height = Config.get("max_in_scope_entity_height")
        min_object_in_scope_percent = Config.get("min_object_in_scope_percent")
        min_object_area = Config.get("min_object_area")
        self.low_confidence = np.where(np.logical_or(self.depthImg > Config.get("too_far_to_care"),\
            self.depthImg == 0), 1, 0)
        for obj in self.objects:
            if not obj.in_scope:
                continue
            obj.segMask = obj.segMask & np.logical_not(self.low_confidence)\
                & (self.cartImg[:,:,1] < max_in_scope_entity_height)
            if float(np.sum(obj.segMask)) / float(img_size) < min_object_area:
                obj.set_out_of_scope()
            if np.sum(obj.segMask) / obj.original_area < min_object_in_scope_percent:
                obj.set_out_of_scope()
        bottom = Config.get("bottom_idxs_for_ground")
        maxi = 0
        id = -1
        for obj in self.objects:
            if obj.in_scope:
                sumi = np.sum(obj.segMask[-bottom:-1,:])
                if sumi > maxi:
                    maxi = sumi
                    id = obj.id
        for obj in self.objects:
            if obj.id == id:
                obj.set_out_of_scope()

    @timeit
    def consolidate_objects(self):
        obj_list = []
        for obj in self.objects:
            if obj.in_scope:
                obj_list.append(obj)
        num_obj = obj_list.__len__()
        areas = np.empty(num_obj)
        for i in range(num_obj):
            obj = obj_list[i]
            Debug_Timer.start("find_area")
            areas[i] = np.sum(obj.segMask)
            Debug_Timer.stop("find_area")
        argsort = areas.argsort()[::-1] # largest to smallest
        grid = np.full(self.rgbImg.shape[0:2], -1, dtype=int)
        Debug_Timer.start("loop")
        for i in range(num_obj):
            Debug_Timer.start("sectioning")
            obj = obj_list[argsort[i]]
            d = []
            section = grid.flatten()[obj.segMask.flatten()]
            for j in np.unique(section):
                if j != -1:
                    d.append(j)
            Debug_Timer.stop("sectioning")
            Debug_Timer.start("combining")
            for idx in d:
                other_obj = obj_list[argsort[idx]]
                if np.sum(np.logical_and(obj.segMask, other_obj.segMask)) / np.sum(obj.segMask)\
                    > Config.get("object_overlap_percent"):
                    other_obj.segMask = np.logical_or(obj.segMask, other_obj.segMask)
                    obj.set_out_of_scope()
            grid[np.logical_and(grid == -1, obj.segMask)] = i
            Debug_Timer.stop("combining")
        Debug_Timer.stop("loop")

    @timeit
    def make_cartesian(self):
        rows, cols = Config.get("dimensions")
        cartImg = np.empty([rows, cols, 3]) # x = left/right, y = up/down, z = in/out
        horiz_fov = min(Config.get("color_fov")[0], Config.get("depth_fov")[0]) * math.pi / 180
        vert_fov = min(Config.get("color_fov")[1], Config.get("depth_fov")[1]) * math.pi / 180
        lr = np.linspace(-horiz_fov / 2, horiz_fov / 2, cols)
        ud = np.linspace(vert_fov / 2, -vert_fov / 2, rows)
        divisor = 1 / (45 * math.pi / 180)
        camera_height = Config.get("camera_vertical_height")
        heightImg = np.copy(self.depthImg)
        for i in range(cols):
            heightImg[:,i] = heightImg[:,i] * ud * divisor + camera_height
        widthImg = np.copy(self.depthImg)
        for i in range(rows):
            widthImg[i,:] = widthImg[i,:] * lr * divisor
        cartImg[:,:,0] = widthImg
        cartImg[:,:,1] = heightImg
        cartImg[:,:,2] = self.depthImg
        return cartImg

    @timeit
    def fit_least_squares(self, obj):
        Debug_Timer.start("pre")
        mask = obj.segMask
        top_down = np.array([self.cartImg[mask,0].flatten(), self.cartImg[mask,2].flatten()]).T
        top_down = top_down[np.logical_not(np.isnan(top_down[:,0])),:]
        x = top_down[:,0]
        y = top_down[:,1]
        lin = np.linspace(0, self.cartImg[:,:,2].shape[1] - 1, self.cartImg[:,:,2].shape[1])
        xs = np.empty(self.cartImg[:,:,2].shape)
        for i in range(self.cartImg[:,:,2].shape[0]):
            xs[i,:] = lin
        y = np.where(mask, self.cartImg[:,:,2], np.nan).flatten()
        x = np.where(mask, xs, np.nan).flatten()

        stds = Config.get("least_squares_outlier_stds")
        for a in [x,y]:
            mean = np.nanmean(a)
            std = np.nanstd(a)
            lower_bound = mean - std * stds
            upper_bound = mean + std * stds
            filter = np.logical_or(a < lower_bound, a > upper_bound)
            x[filter] = np.nan
            y[filter] = np.nan

        def fit_line(x, y, degree):
            n = x.size - np.sum(np.isnan(x))
            sx = np.nansum(x)
            sy = np.nansum(y)
            sxy = np.nansum(x * y)
            sxd = np.nansum(np.power(x, degree))
            m = (n * sxy - sx * sy) / (n * sxd - np.power(sx, degree))
            b = (sy - m * sx) / n
            return m, b

        Debug_Timer.stop("pre")
        Debug_Timer.start("least_squares")
        m, b = fit_line(x, y, 2)
        Debug_Timer.stop("least_squares")
        Debug_Timer.start("post")
        depth_line = np.array([[np.nanmin(x), m * np.nanmin(x) + b], [np.nanmax(x), m * np.nanmax(x) + b]])
        if np.sum(np.isnan(depth_line)):
            return
        rows, cols = Config.get("dimensions")
        horiz_fov = min(Config.get("color_fov")[0], Config.get("depth_fov")[0]) * math.pi / 180
        lr = np.sin(np.linspace(-horiz_fov / 2, horiz_fov / 2, cols))
        for i in range(2):
            depth_line[i,0] = depth_line[i,1] * lr[depth_line[i,0].astype(int)]
        obj.least_squares_line = depth_line
        Debug_Timer.stop("post")

class Object:
    def __init__(self, id, outline, segMask):
        self.id = id
        self.outline = outline
        self.segMask = segMask
        self.original_area = np.sum(segMask)
        self.in_scope = bool(True)
        self.least_squares_line = np.array([[0,0], [0,0]], dtype=np.float64)

    def set_out_of_scope(self):
        self.in_scope = bool(False)

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