import numba_functs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
import cv2
import yaml
from ultralytics import FastSAM, SAM
from shapely.geometry import Polygon
from numba import njit, prange
from timeit import default_timer as timer
from PIL import Image
from scipy import stats

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
        plt.subplot(2, 2, 1)
        plt.title('Tracked Objects')
        plt.imshow(self.window.current_frame.rgbImg)
        colors = {}
        obj_list = list(self.window.objects_in_scope)
        for obj in obj_list:
            outline = np.concatenate((obj.outline, [obj.outline[0,:]]))
            colors[obj.id] = plt.plot(outline[:,0],outline[:,1])[-1].get_color()
        plt.subplot(2, 2, 3)
        plt.title('Top Down 2D Prediction')
        plt.scatter(0,0, c='k')
        ax = plt.gcf().gca()
        areas = np.empty(len(obj_list))
        for i in range(len(obj_list)):
            areas[i] = obj_list[i].circle[2]
        argsort = areas.argsort()
        for i in range(len(obj_list)):
            obj = obj_list[argsort[len(obj_list) - i - 1]]
            circle = obj.circle
            show_fit_circle = plt.Circle((circle[0], circle[1]), circle[2], color=colors[obj.id])
            ax.add_patch(show_fit_circle)
        for pred in self.window.predictions.values():
            if pred.future is not None:
                if pred.id in colors:
                    color = colors[pred.id]
                else:
                    color = 'grey'
                num = 5
                for i in range(num):
                    by = int(pred.future.shape[0] / num)
                    c = pred.future[i * by,:]
                    show_fit_circle = plt.Circle((c[0], c[1]), c[2], color=color, fill=False)
                    ax.add_patch(show_fit_circle)
        plt.axis('equal')
        plt.subplot(1, 2, 2)
        ax = plt.gcf().gca()
        plt.xlabel("Steering Angle")
        plt.ylabel("Permissable Velocity")
        ax.yaxis.set_label_position("right")
        plt.scatter(0,0, c='k')
        grid = self.window.trajectory_grid
        plt.plot(grid[-1,:,0], grid[-1,:,1], c='k', label='20 km/h')
        plt.legend()
        plt.plot(grid[:,0,0], grid[:,0,1], c='k')
        plt.plot(grid[:,-1,0], grid[:,-1,1], c='k')
        traj_bool = self.window.get_trajectories()
        angle_samples = Config.get("angle_samples")
        argsort = traj_bool.argsort()
        for i_traj in range(angle_samples):
            i_traj = argsort[-i_traj]
            traj_val = traj_bool[i_traj]
            if grid[traj_val,i_traj,1] > 10:
                color = 'g'
            elif grid[traj_val,i_traj,1] < 5:
                color = 'r'
            else:
                color = 'orange'
            plt.plot(grid[:traj_val,i_traj,0], grid[:traj_val,i_traj,1], c=color)
        plt.axis('equal')
        plt.tight_layout()
        return fig
    
    def add_image_file(self, rgb, depth):
        rgbImg = Image.open(rgb)
        depthImg = np.asarray(Image.open(depth))[:,:,0].astype(float)
        depthImg = -5.417 * depthImg / 100 + 9.125 # estimation
        depthImg = Image.fromarray(depthImg)
        self.resize_images(rgbImg, depthImg)
    
    @timeit
    def add_CARLA_image_file(self, imgName):
        Debug_Timer.start("open_img")
        rgbImg = Image.open("../data/rgb" + imgName + ".jpeg")
        depthImg = Image.fromarray(np.loadtxt("../data/depth" + imgName + ".out", delimiter=","))
        Debug_Timer.stop("open_img")
        self.resize_images(rgbImg, depthImg)
        
    @timeit
    def add_OAKD_image_file(self, imgName):
        Debug_Timer.start("open_img")
        rgbImg = Image.open("../oakd_data/test/rgb" + imgName + ".png")
        depthImg = np.asarray(Image.open("../oakd_data/test/depth" + imgName + ".png"))[:,:,0].astype(float)
        depthImg = -5.417 * depthImg / 100 + 9.125 # estimation
        depthImg = Image.fromarray(depthImg)
        Debug_Timer.stop("open_img")
        self.resize_images(rgbImg, depthImg)

    def resize_images(self, rgbImg, depthImg): # to common FOV
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
        h, v = Config.get("dimensions")
        rgbImg = np.asarray(Image.fromarray(rgbImg).resize((v, h)))
        depthImg = np.asarray(Image.fromarray(depthImg).resize((v, h)))
        self.window.receive_img(rgbImg, depthImg)

    def get_trajectories(self):
        return self.window.get_trajectories()

class Window:
    def __init__(self):
        self.tracking_imgs = Config.get("tracking_imgs")
        self.model = FastSAM("FastSAM-s.pt")
        # self.model = SAM("sam_b.pt")
        # self.model.to('cuda') 2080 has drivers that are too old lol
        self.frames = []
        self.predictions = {}
        self.trajectory_grid = self.make_trajectory_grid()

    def make_trajectory_grid(self):
        l = Config.get("length_front_to_back_wheels")
        max_angle = Config.get("max_steering_angle")
        distance = Config.get("distance_steering_prediction")
        angle_samples = Config.get("angle_samples")
        prediction_samples = Config.get("seconds_into_future") * Config.get("samples_per_second")
        if prediction_samples % 2 == 0:
            prediction_samples += 1
        def A(x):
            return (90 - x) * math.pi / 180
        def U(x):
            a = A(x)
            return math.tan(a) * (math.sin(a) + l) + math.cos(a)
        def w(a, t):
            u = U(a)
            return -u * math.cos(t / u) + u
        def z(a, t):
            u = U(a)
            return u * math.sin(t / u)
        trajectory_grid = np.zeros([prediction_samples + 1, angle_samples, 2])
        angle_steps = np.linspace(-max_angle / 2, max_angle / 2, angle_samples)
        time_steps = np.linspace(0, distance, prediction_samples + 1)
        for a in range(angle_steps.size):
            for t in range(time_steps.size):
                a_s = angle_steps[a]
                t_s = time_steps[t]
                trajectory_grid[t,a,0] = w(a_s,t_s)
                trajectory_grid[t,a,1] = z(a_s,t_s)
        return trajectory_grid

    def receive_img(self, rgbImg, depthImg):
        Debug_Timer.start("fastSAM")
        results = self.model.track(rgbImg, persist=True, verbose=False)[0]
        Debug_Timer.stop("fastSAM")
        self.current_frame = Frame(rgbImg, depthImg, results)
        if len(self.frames) == self.tracking_imgs:
            self.frames.pop(0)
        self.frames.append(self.current_frame)
        if len(self.frames) == self.tracking_imgs:
            self.objects_in_scope = self.get_objects_in_scope()
            for obj in self.objects_in_scope:
                self.current_frame.fit_circle(obj)
        else:
            self.objects_in_scope = []
        for obj in self.objects_in_scope:
            if obj.id in self.predictions:
                self.predictions[obj.id].add_circle(obj)
            else:
                pred = Prediction(obj)
                self.predictions[obj.id] = pred
        for pred in list(self.predictions.values()):
            if pred.past_lifetime() > Config.get("out_of_sight_lifetime"):
                del self.predictions[pred.id]

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
        for object in self.current_frame.objects:
            if object.id in d:
                if d[object.id] == self.tracking_imgs:
                    objects_in_scope.append(object)
        return objects_in_scope

    @timeit
    def get_trajectories(self):
        angle_samples = Config.get("angle_samples")
        future_samples = Config.get("seconds_into_future") * Config.get("samples_per_second")
        traj_bool = np.zeros(angle_samples, dtype=int) + future_samples + 1
        for i_traj in range(angle_samples):
            traj = self.trajectory_grid[:,i_traj,:]
            for pred in self.predictions.values():
                if pred.future is not None:
                    for i_time in range(future_samples):
                        circle = pred.future[i_time,:]
                        point = traj[i_time,:]
                        distance = np.sum(np.square(circle[0:2] - point))
                        if distance < (circle[2] + Config.get("safe_distance")) ** 2:
                            traj_bool[i_traj] = min(i_time, traj_bool[i_traj])
                            break
        return traj_bool

class Frame:
    def __init__(self, rgbImg, depthImg, results):
        self.rgbImg = rgbImg
        self.depthImg = depthImg
        self.results = results
        self.rows, self.cols = Config.get("dimensions")
        self.objects = self.populate_objects()
        self.cartImg = self.make_cartesian()
        self.filter_objects()

    def populate_objects(self):
        objects = []
        ids = np.array(self.results.boxes.id.int().cpu().tolist())
        boxes = self.results.boxes.xywh.cpu()
        outlines = self.results.masks.xy
        for idx, (id, box, outline) in enumerate(zip(ids, boxes, outlines)):
            obj = Object(id, box, outline)
            objects.append(obj)
        return objects
    
    @timeit
    def filter_objects(self):
        img_size = Config.get("dimensions")[0] * Config.get("dimensions")[1]
        min_area = Config.get("min_object_area")
        sky_bright_percent = Config.get("sky_brightness_percent")
        max_percent_sky_overlap = Config.get("max_percent_sky_overlap")
        max_percent_ground_overlap = Config.get("max_percent_ground_overlap")
        rgb_sum = np.sum(self.rgbImg, axis=2)
        max_bright = np.max(rgb_sum)
        bright_cutoff = max_bright * (1 - sky_bright_percent)
        sky = np.where(rgb_sum > bright_cutoff, 1, 0)
        ground = self.find_ground()
        for obj in self.objects:
            if obj.segMask is None:
                obj.set_out_of_scope()
                continue
            if np.sum(obj.segMask & sky) / obj.area > max_percent_sky_overlap:
                obj.set_out_of_scope()
                continue
            if np.sum(obj.segMask & ground) / obj.area > max_percent_ground_overlap:
                obj.set_out_of_scope()
                continue
            obj.segMask = obj.segMask & np.logical_not(sky) & np.logical_not(ground)
            if float(obj.area) / float(img_size) < min_area:
                obj.set_out_of_scope()

    @timeit
    def make_cartesian(self):
        rows = self.depthImg.shape[0]
        cols = self.depthImg.shape[1]
        cartImg = np.empty([rows, cols, 3]) # x = left/right, y = up/down, z = in/out
        horiz_fov = min(Config.get("color_fov")[0], Config.get("depth_fov")[0]) * math.pi / 180
        vert_fov = min(Config.get("color_fov")[1], Config.get("depth_fov")[1]) * math.pi / 180
        lr = np.sin(np.linspace(-horiz_fov / 2, horiz_fov / 2, cols))
        ud = np.sin(np.linspace(-vert_fov / 2, vert_fov / 2, rows))
        # self.make_cartesian_njit_helper(lr, ud, self.depthImg, cartImg)
        for i in range(rows):
            for j in range(cols):
                rho = self.depthImg[i,j]
                x = rho * lr[j]
                y = rho * ud[i]
                z = rho
                cartImg[i,j,:] = [x, -y, z]
        return cartImg
    
    @timeit
    def find_ground(self):
        ground = np.zeros(self.depthImg.shape, dtype=bool)
        for column in range(self.depthImg.shape[1]):
            length = self.depthImg[:,0].shape[0]
            outline = np.asarray([self.cartImg[:,column,2], self.cartImg[:,column,1], np.zeros(length) - 1]).T[::-1,:]
            start = np.where(np.min(outline[:,0]) == outline[:,0])[0][0]
            # Debug_Timer.start("put in buckets")
            lda = .25
            max_dist = .4
            max_angle = .5
            curr_bucket = np.max(outline[:,0]) ** 2
            prev_bucket = np.max(outline[:,0]) ** 2
            bucket_i = -1
            buckets = np.zeros((length, 3)) # [x, y, total]
            for p_i in range(start, length):
                point = outline[p_i,0:2]
                if abs(point[0] - curr_bucket) < lda:
                    outline[p_i,2] = bucket_i
                    buckets[bucket_i, 0:2] += point
                    buckets[bucket_i, 2] += 1
                    curr_bucket = buckets[bucket_i, 0] / buckets[bucket_i, 2]
                    continue
                if abs(point[0] - prev_bucket) < lda:
                    outline[p_i,2] = bucket_i - 1
                    buckets[bucket_i - 1, 0:2] += point
                    buckets[bucket_i - 1, 2] += 1
                    prev_bucket = buckets[bucket_i - 1, 0] / buckets[bucket_i - 1, 2]
                    continue
                bucket_i += 1
                prev_bucket = curr_bucket
                curr_bucket = point[0]
                buckets[bucket_i, 2] += 1
                buckets[bucket_i, 0:2] += point
            buckets = buckets[:bucket_i + 1,0:2] / buckets[:bucket_i + 1,2].reshape(bucket_i + 1,1)
            # Debug_Timer.stop("put in buckets")
            # Debug_Timer.start("find idx")
            dist = np.sum(np.square(buckets[:-1,:] - buckets[1:,:]), axis=1)
            angle = np.acos(abs(buckets[:-1,0] - buckets[1:,0]) ** 2 / dist)
            done = 0
            for i in range(buckets.shape[0] - 2):
                if dist[i] > max_dist or abs(angle[i] - angle[i + 1]) > max_angle:
                    done = i + 1
                    break
            if done >= buckets.shape[0]:
                done = buckets.shape[0] - 1
            c = buckets[done]
            mini = np.max(outline[:,0]) - np.min(outline[:,0])
            for i in range(start, length):
                dist = np.sum(np.square(c - outline[i,0:2]))
                if dist < mini:
                    mini = dist
                    idx = i
            # Debug_Timer.stop("find idx")
            ground[length - idx:,column] = True
        return ground
    
    @timeit
    def fit_circle(self, obj):
        mask = obj.segMask
        top_down = np.array([self.cartImg[mask,0].flatten(), self.cartImg[mask,2].flatten()]).T
        top_down = top_down[np.logical_not(np.isnan(top_down[:,0])),:]
        right_most = top_down[np.where(top_down[:,0] == np.nanmax(top_down[:,0]))[0][0], :]
        left_most = top_down[np.where(top_down[:,0] == np.nanmin(top_down[:,0]))[0][0], :]
        right_most_angle = math.atan(right_most[0] / right_most[1])
        left_most_angle = math.atan(left_most[0] / left_most[1])
        angle_from_straight = (right_most_angle + left_most_angle) / 2
        sin = math.sin(angle_from_straight)
        cos = math.cos(angle_from_straight)
        rm = np.array([[cos,sin],[-sin,cos]])
        rotated = np.matmul(top_down, rm)
        # find edge paralel to x axis
        h = np.max(rotated[:,1])
        d = rotated[:,0] * h / rotated[:,1]
        n = np.sqrt((d - rotated[:,0]) ** 2 + (h - rotated[:,1]) ** 2)
        iso = np.asarray([d, h - n]).T
        num_p = Config.get("number_edge_points")
        argsort = iso[:, 0].argsort()
        sorted_iso = iso[argsort]
        buckets = np.linspace(np.min(sorted_iso[:,0]), np.max(sorted_iso[:,0]), num_p)
        wheres = np.zeros(buckets.size - 1, dtype=int)
        buckets_i = 0
        last_bucket = 0
        for points_i in range(sorted_iso.shape[0]):
            if sorted_iso[points_i,0] > buckets[buckets_i + 1]:
                wheres[buckets_i] = np.where(sorted_iso[last_bucket:points_i,1]\
                    == np.min(sorted_iso[last_bucket:points_i,1]))[0][0] + last_bucket
                last_bucket = points_i
                buckets_i += 1
        try:
            wheres[-1] = np.where(sorted_iso[last_bucket:points_i,1]\
                == np.min(sorted_iso[last_bucket:points_i,1]))[0][0] + last_bucket
        except ValueError:
            wheres = wheres[:-1]
        to_fit = rotated[argsort[wheres],:] # TODO: fix bug where sometimes a value in wheres isn't set
        # fit a circle to the linear regression of the edge
        res = stats.linregress(to_fit[:,0], to_fit[:,1])
        ends = np.empty((2,2))
        ends[0,0] = to_fit[0,0]
        ends[1,0] = to_fit[-1,0]
        ends[0,1] = res.intercept + res.slope * to_fit[0,0]
        ends[1,1] = res.intercept + res.slope * to_fit[-1,0]
        midpoint = np.sum(ends, axis=0) / 2
        radius = math.sqrt((ends[0,0] - ends[1,0]) ** 2 + (ends[0,1] - ends[1,1]) ** 2) / 2
        s = radius * math.pi / 4
        theta = math.atan(-1 / res.slope)
        x = s * math.cos(theta)
        y = s * math.sin(theta)
        center = np.asarray([x * np.sign(y) + midpoint[0], y * np.sign(y) + midpoint[1]])
        fit_circle = np.array([center[0], center[1], radius]) # [x, y, radius]
        sin = math.sin(-angle_from_straight)
        cos = math.cos(-angle_from_straight)
        rm = np.array([[cos,sin],[-sin,cos]])
        fit_circle[0:2] = np.matmul(fit_circle[0:2], rm)
        obj.circle = fit_circle

class Object:
    def __init__(self, id, box, outline):
        self.id = id
        self.box = box
        self.in_scope = True
        self.outline = outline
        self.segMask = self.get_seg_mask(outline)
        self.area = np.sum(self.segMask)
        self.circle = None

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

class Prediction:
    def __init__(self, object, files_mode=False):
        if files_mode:
            self.lifetime = 0
        else:
            self.lifetime = timer()
        self.files_mode = files_mode
        self.id = object.id
        self.circles = np.empty((2,3))
        self.circles[0,:] = object.circle
        self.samples = 1
        self.total_radius = object.circle[2]
        self.future = None

    def add_circle(self, object):
        if self.files_mode:
            time_delta = Config.get("file_mode_image_speed")
            self.lifetime = 0
        else:
            time_delta = timer() - self.lifetime
            self.lifetime = timer()
        self.circles[1,:] = self.circles[0,:]
        self.circles[0,:] = object.circle
        self.samples += 1
        self.total_radius += object.circle[2]
        future_samples = Config.get("seconds_into_future") * Config.get("samples_per_second")
        self.future = np.empty((future_samples + 1, 3))
        self.future[0,:] = object.circle
        self.future[:,2] = self.total_radius / self.samples
        velocity = (self.circles[0,0:2] - self.circles[1,0:2]) / time_delta
        for i in range(future_samples):
            self.future[i + 1,0:2] = self.future[i,0:2] + velocity

    def past_lifetime(self):
        if self.files_mode:
            self.lifetime += Config.get("file_mode_image_speed")
            return self.lifetime
        else:
            return timer() - self.lifetime

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
            keys = list(cls.total.keys())
            for i in range(len(cls.total)):
                key = keys[argsort[i]]
                print(key + ":")
                print("\ttotal = " + str(cls.total[key])\
                    + ", " + str(cls.samples[key]) + " samples")
                print("\tavg = " + str(cls.total[key] / cls.samples[key])\
                    + ", min = " + str(cls.mini[key])\
                    + ", max = " + str(cls.maxi[key]))
            
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

    @classmethod
    def load(cls, name):
        with open(name + ".yaml") as f:
            cls.c = yaml.load(f, Loader=yaml.FullLoader)

    @classmethod
    def get(cls, name):
        return cls.c[name]