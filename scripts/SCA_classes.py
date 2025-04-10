import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
import cv2
import yaml
import numba_functs
import numba as nb
import numba.types as nbt
from ultralytics import FastSAM
from timeit import default_timer as timer
from PIL import Image
from scipy import stats
from shapely import Polygon, Point

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
        numba_functs.compile_numba_functs()
        self.window = Window()
    
    def plot(self):
        frame = self.window.frame
        plt_colors = list(mcolors.TABLEAU_COLORS.values())
        fig = plt.figure()
        plt.subplot(3, 3, 1)
        plt.title('Tracked Objects')
        plt.imshow(self.window.frame.rgbImg)
        obj_list = list(self.window.objects_in_scope)
        for obj in obj_list:
            outline = np.concatenate((obj.outline, [obj.outline[0,:]]))
            plt.plot(outline[:,0],outline[:,1],color=plt_colors[obj.id % plt_colors.__len__()])
        plt.subplot(3, 3, 4)
        plt.title('Top-Down Prediction')
        t = Config.get("seconds_into_future")
        for pred in list(self.window.predictions.values()):
            if pred.samples == 1:
                continue
            for two in range(2):
                start = pred.starts[two]
                line = pred.time_in[two](t)
                hyp = math.sqrt((start[0] - line[0])**2 + (start[1] - line[1])**2)
                longer_t = 2 * (pred.radius + Config.get("safe_distance")) * t/ hyp + t
                line = pred.time_in[two](longer_t)
                pred_line = np.array([start,line])
                plt.scatter(pred.current_circle[0],pred.current_circle[1], color=plt_colors[pred.id % plt_colors.__len__()])
                plt.plot(pred_line[:,0],pred_line[:,1], color=plt_colors[pred.id % plt_colors.__len__()])
        plt.scatter(0,0, c='k')
        ax = plt.gcf().gca()
        width, height = ax.get_figure().get_size_inches()
        xdim = 10
        plt.xlim(-xdim,xdim)
        ydim = xdim / height * width
        plt.ylim(-5,2 * ydim - 5)
        plt.subplot(3, 3, 3)
        plt.title('Depth Masks of Objects')
        nan_depthImg = np.zeros(frame.depthImg.shape)
        nan_depthImg = np.where(frame.low_confidence, np.nan, frame.depthImg)
        plt.imshow(nan_depthImg)
        for obj in self.window.objects_in_scope:
            mask = np.where(obj.segMask,0,np.nan)
            plt.imshow(mask)
        plt.subplot(3, 3, 5)
        plt.title('Object Histories')
        plt.scatter(0,0,c='k')
        ax = plt.gcf().gca()
        for pred in self.window.predictions.values():
            plt.scatter(pred.circles[:,0], pred.circles[:,1], c=plt_colors[pred.id % plt_colors.__len__()])
            plt.plot(pred.circles[:,0], pred.circles[:,1], c=plt_colors[pred.id % plt_colors.__len__()])
        plt.axis('equal')
        plt.subplot(3, 3, 2)
        plt.title('Sky and Ground')
        plt.imshow(self.window.frame.rgbImg)
        nan_depthImg = np.zeros(frame.depthImg.shape)
        nan_depthImg = np.where(frame.ground, frame.depthImg, np.nan)
        plt.imshow(nan_depthImg)
        plt.xlim(0,self.window.frame.rgbImg.shape[1])
        plt.subplot(3, 3, 6)
        plt.title('Circles of Best Fit')
        plt.scatter(0,0,c='k')
        ax = plt.gcf().gca()
        plt.axis('equal')
        for obj in self.window.objects_in_scope:
            mask = obj.segMask
            top_down = np.array([frame.cartImg[mask,0].flatten(), frame.cartImg[mask,2].flatten()]).T
            plt.scatter(top_down[:,0],top_down[:,1],s=1,c=plt_colors[obj.id % plt_colors.__len__()])
            circle = obj.circle
            show_fit_circle = plt.Circle((circle[0], circle[1]), circle[2], fill=False)
            ax.add_patch(show_fit_circle)
        plt.tight_layout()
        plt.subplot(3, 1, 3)
        plt.title('Permitted Velocity')
        max_angle = Config.get("max_steering_angle")
        angle_samples = Config.get("angle_samples")
        angle_steps = np.linspace(-max_angle, max_angle, angle_samples)
        velocities, avoidance, collision = self.window.get_velocities()
        if collision:
            plt.plot(angle_steps, avoidance, c='r')
        else:
            plt.plot(angle_steps, velocities, c='b')
        return fig
    
    def add_image_file(self, rgb, depth):
        rgbImg = Image.open(rgb)
        depthImg = np.asarray(Image.open(depth))[:,:,0].astype(float)
        depthImg = -5.417 * depthImg / 100 + 9.125 # estimation
        depthImg = Image.fromarray(depthImg)
        self.resize_images(rgbImg, depthImg)
    
    def add_demo_image_file(self, imgName):
        # Debug_Timer.start("open_img")
        rgbImg = Image.open("/content/SegmentingCollisionAvoidance/oakd_data/test/rgb" + imgName + ".png")
        depthImg = np.asarray(Image.open("/content/SegmentingCollisionAvoidance/oakd_data/test/depth"\
            + imgName + ".png"))[:,:,0].astype(float)
        depthImg = -5.417 * depthImg / 100 + 9.125 # estimation
        depthImg = Image.fromarray(depthImg)
        # Debug_Timer.stop("open_img")
        self.resize_images(rgbImg, depthImg)
    
    def add_CARLA_image_file(self, imgName):
        # Debug_Timer.start("open_img")
        rgbImg = Image.open("../data/rgb" + imgName + ".jpeg")
        depthImg = Image.fromarray(np.loadtxt("../data/depth" + imgName + ".out", delimiter=","))
        # Debug_Timer.stop("open_img")
        self.resize_images(rgbImg, depthImg)
        
    def add_OAKD_image_file(self, imgName):
        # Debug_Timer.start("open_img")
        rgbImg = Image.open("../oakd_data/test/rgb" + imgName + ".png")
        depthImg = np.asarray(Image.open("../oakd_data/test/depth" + imgName + ".png"))[:,:,0].astype(float)
        depthImg = -5.417 * depthImg / 100 + 9.125 # estimation
        depthImg = Image.fromarray(depthImg)
        # Debug_Timer.stop("open_img")
        self.resize_images(rgbImg, depthImg)

    # @timeit
    def resize_images(self, rgbImg, depthImg): # to common FOV
        Debug_Timer.start("image resizing")
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
        Debug_Timer.stop("image resizing")
        self.window.receive_img(rgbImg, depthImg)

    # def get_velocities(self):
    #     return self.window.get_velocities()

    def get_steering_angle(self, preferred_angle, bang_bang):
        velocities, avoidance, collision = self.window.get_velocities()
        if bang_bang: # angles in and out in [-1, 0, 1]: [left, center, right]
            steering = Config.get("bang_bang_steering")
            if collision:
                idx = avoidance.argsort()[-1]
                if idx < avoidance.__len__() - 1:
                    return steering[0]
                return steering[2]
            else: # for now, do nothing
                return preferred_angle
        else: # angles in and out in [0...1]: [left...right]
            if collision:
                return float(avoidance.argsort()[-1]) / (avoidance.__len__() - 1)
            else: # for now, do nothing
                return preferred_angle

class Window:
    def __init__(self):
        self.tracking_imgs = Config.get("tracking_imgs")
        self.model = FastSAM("FastSAM-s.pt")
        # self.model.to('cuda') 2080 has drivers that are too old lol
        self.predictions = {}

    def receive_img(self, rgbImg, depthImg):
        Debug_Timer.start("fastSAM")
        results = self.model.track(rgbImg, persist=True, verbose=False)[0]
        Debug_Timer.stop("fastSAM")
        Debug_Timer.start("frame init")
        self.frame = Frame(rgbImg, depthImg, results)
        Debug_Timer.stop("frame init")
        # Debug_Timer.start("objs and preds")
        self.objects_in_scope = []
        ids_list = []
        for obj in self.frame.objects:
            if obj.in_scope:
                self.objects_in_scope.append(obj)
                self.frame.fit_circle(obj)
                ids_list.append(obj.id)
        for obj in self.objects_in_scope:
            if obj.id in self.predictions:
                self.predictions[obj.id].add_circle(obj, Config.get("file_mode_image_speed")) # change for ros!
            else:
                pred = Prediction(obj)
                self.predictions[obj.id] = pred
        for pred in list(self.predictions.values()):
            if pred.id not in ids_list:
                del self.predictions[pred.id]
        # Debug_Timer.stop("objs and preds")

    # @timeit
    def get_velocities(self):
        brake = False
        max_angle = Config.get("max_steering_angle")
        angle_samples = Config.get("angle_samples")
        l = Config.get("length_front_to_back_wheels")
        t = Config.get("seconds_into_future")
        safe_distance = Config.get("safe_distance")
        def A(x):
            return (90 - x) * math.pi / 180
        def U(x):
            a = A(x)
            return math.tan(a) * (math.sin(a) + l) + math.cos(a)
        angle_steps = np.linspace(-max_angle, max_angle, angle_samples)
        collision = False
        velocities = np.full(angle_samples, Config.get("max_considered_velocity"))
        avoidance = np.zeros(angle_samples)
        for pred in list(self.predictions.values()):
            try:
                obj_collision = False
                if pred.samples == 1:
                    continue
                starts = np.array([pred.starts[0], pred.starts[1]])
                ends = np.array([pred.time_in[0](t), pred.time_in[1](t)])
                hyp = math.sqrt((starts[0,0] - ends[0,0])**2 + (starts[0,1] - ends[0,1])**2)
                longer_t = 2 * (pred.radius + safe_distance) * t/ hyp + t
                ends = np.array([pred.time_in[0](longer_t), pred.time_in[1](longer_t)])
                coords = np.append(starts,ends[::-1,:],axis=0)
                polygon = Polygon(coords)
                point = Point(0, 0)
                if polygon.contains(point):
                    obj_collision = True
                    collision = True
                for two in range(2):
                    pred_line = np.array([starts[two,:], ends[two,:]])
                    for i, a in enumerate(angle_steps):
                        if a == 0:
                            y = self.y_intercept(pred_line)
                            time = pred.time_out[two]([0,y])
                            if time > 0 and y > 0:
                                velocities[i] = min(y / time, velocities[i])
                        else:
                            val = U(a)
                            track = np.array([val,0,abs(val)])
                            num_p, inter = self.intersect(track, pred_line)
                            if num_p == 0:
                                continue
                            elif num_p == 1:
                                arc = self.arc_length(track,inter)
                                pt = inter[0,:]
                            else:
                                arc1 = self.arc_length(track,inter[0,:])
                                arc2 = self.arc_length(track,inter[1,:])
                                if arc1 < arc2:
                                    arc = arc1
                                    pt = inter[0,:]
                                else:
                                    arc = arc2
                                    pt = inter[1,:]
                            time = pred.time_out[two](pt)
                            # time = min(pred.time_out[two](pt), t)
                            if time > 0:
                                velocities[i] = min(arc / time, velocities[i])
                        if obj_collision:
                            if velocities[i] == 0:
                                avoidance[i] = 1000000 # this should be big enough
                            else:
                                avoidance[i] = max(1.0 / velocities[i], avoidance[i])
            except:
                pass
        return [velocities, avoidance, collision]
    
    def intersect(self, circle, segment):
        out = np.empty((2,2))
        dist1 = (circle[0] - segment[0,0])**2 + (circle[1] - segment[0,1])**2
        dist2 = (circle[0] - segment[1,0])**2 + (circle[1] - segment[1,1])**2
        if min(dist1, dist2) > circle[2]**2:
            return [0, out]
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
            return [0, out]
        elif disc == 0:
            out[0,:] = [det * dy, -det * dx]
            return [1, out]
        else:
            xpm = np.sign(dy) * dx * math.sqrt(disc)
            ypm = abs(dy) * math.sqrt(disc)
            out[0,:] = [det * dy + xpm, -det * dx + ypm]
            out[1,:] = [det * dy - xpm, -det * dx - ypm]
            out /= dr
            return [2, out + circle[0:2]]
        
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

    def y_intercept(self, line):
        m = (line[0,1] - line[1,1]) / (line[0,0] - line[1,0])
        return line[0,1] - m * line[0,0]

    def x_intercept(self, line):
        m = (line[0,1] - line[1,1]) / (line[0,0] - line[1,0])
        b = line[0,1] - m * line[0,0]
        return -b / m

class Frame:
    def __init__(self, rgbImg, depthImg, results):
        self.rgbImg = rgbImg
        self.depthImg = depthImg
        self.results = results
        self.rows, self.cols = Config.get("dimensions")
        self.objects = self.populate_objects()
        self.cartImg = self.make_cartesian()
        self.filter_objects()
        self.consolidate_objects()

    @timeit
    def populate_objects(self):
        objects = []
        ids = np.array(self.results.boxes.id.int().cpu().tolist())
        outlines = self.results.masks.xy
        dimensions = Config.get("dimensions")
        for id, outline in zip(ids, outlines):
            segMask = np.zeros(dimensions)
            if outline.shape[0] >= 3:
                segMask = cv2.fillPoly(segMask, [outline.astype(int)], color=1)
            obj = Object(id, outline, segMask.astype(bool))
            if outline.shape[0] < 3:
                obj.set_out_of_scope()
            objects.append(obj)
        return objects
    
    # @timeit
    # def filter_objects(self):
    #     img_size = Config.get("dimensions")[0] * Config.get("dimensions")[1]
    #     sky_bright_percent = Config.get("sky_brightness_percent")
    #     max_percent_sky_overlap = Config.get("max_percent_sky_overlap")
    #     max_percent_ground_overlap = Config.get("max_percent_ground_overlap")
    #     min_object_area = Config.get("min_object_area")
    #     rgb_sum = np.sum(self.rgbImg, axis=2)
    #     max_bright = np.max(rgb_sum)
    #     bright_cutoff = max_bright * (1 - sky_bright_percent)
    #     self.sky = np.where(rgb_sum > bright_cutoff, 1, 0)
    #     self.ground = self.find_ground()
    #     self.low_confidence = np.where(self.depthImg > Config.get("magic_trust_number"), 1, 0)
    #     # Debug_Timer.start("actually filter")
    #     numba_functs.filter_objects_njit_helper(self.objects, self.objects.__len__(), self.sky, self.ground, self.low_confidence,\
    #         img_size, max_percent_sky_overlap, max_percent_ground_overlap, min_object_area)
    #     # Debug_Timer.stop("actually filter")

    def filter_objects(self):
        img_size = Config.get("dimensions")[0] * Config.get("dimensions")[1]
        sky_bright_percent = Config.get("sky_brightness_percent")
        max_percent_sky_overlap = Config.get("max_percent_sky_overlap")
        max_percent_ground_overlap = Config.get("max_percent_ground_overlap")
        min_object_area = Config.get("min_object_area")
        rgb_sum = np.sum(self.rgbImg, axis=2)
        max_bright = np.max(rgb_sum)
        bright_cutoff = max_bright * (1 - sky_bright_percent)
        self.sky = np.where(rgb_sum > bright_cutoff, 1, 0)
        self.ground = self.find_ground()
        self.low_confidence = np.where(self.depthImg > Config.get("magic_trust_number"), 1, 0)
        # Debug_Timer.start("actually filter")
        for obj in self.objects:
            if not obj.in_scope:
                continue
            if np.sum(obj.segMask & self.sky) / obj.area() > max_percent_sky_overlap:
                obj.set_out_of_scope()
                continue
            if np.sum(obj.segMask & self.ground) / obj.area() > max_percent_ground_overlap:
                obj.set_out_of_scope()
                continue
            obj.segMask = obj.segMask & np.logical_not(self.sky) &\
                np.logical_not(self.ground) & np.logical_not(self.low_confidence)
            if float(obj.area()) / float(img_size) < min_object_area:
                obj.set_out_of_scope()
        # Debug_Timer.stop("actually filter")

    # @timeit
    def consolidate_objects(self):
        obj_list = []
        for obj in self.objects:
            if obj.in_scope:
                obj_list.append(obj)
        num_obj = obj_list.__len__()
        areas = np.empty(num_obj)
        for i in range(num_obj):
            obj = obj_list[i]
            areas[i] = obj.area()
        argsort = areas.argsort()[::-1] # largest to smallest
        grid = np.full(self.rgbImg.shape[0:2], -1, dtype=int)
        for i in range(num_obj):
            obj = obj_list[argsort[i]]
            d = []
            section = grid.flatten()[obj.segMask.flatten()]
            for j in range(section.size):
                if section[j] not in d:
                    d.append(section[j])
            for idx in d:
                if idx == -1:
                    continue
                other_obj = obj_list[argsort[idx]]
                if np.sum(np.logical_and(obj.segMask, other_obj.segMask)) / obj.area()\
                    > Config.get("object_overlap_percent"):
                    other_obj.segMask = np.logical_or(obj.segMask, other_obj.segMask)
                    obj.set_out_of_scope()
            grid[np.logical_and(grid == -1, obj.segMask)] = i

    # @timeit
    def make_cartesian(self):
        rows = self.depthImg.shape[0]
        cols = self.depthImg.shape[1]
        cartImg = np.empty([rows, cols, 3]) # x = left/right, y = up/down, z = in/out
        horiz_fov = min(Config.get("color_fov")[0], Config.get("depth_fov")[0]) * math.pi / 180
        vert_fov = min(Config.get("color_fov")[1], Config.get("depth_fov")[1]) * math.pi / 180
        lr = np.sin(np.linspace(-horiz_fov / 2, horiz_fov / 2, cols))
        ud = np.sin(np.linspace(-vert_fov / 2, vert_fov / 2, rows))
        numba_functs.make_cartesian_njit_helper(rows, cols, lr, ud, self.depthImg, cartImg)
        return cartImg
    
    # @timeit
    def make_cartesian_no_gpu(self):
        rows, cols = Config.get("dimensions")
        cartImg = np.empty([rows, cols, 3]) # x = left/right, y = up/down, z = in/out
        horiz_fov = min(Config.get("color_fov")[0], Config.get("depth_fov")[0]) * math.pi / 180
        vert_fov = min(Config.get("color_fov")[1], Config.get("depth_fov")[1]) * math.pi / 180
        lr = np.sin(np.linspace(-horiz_fov / 2, horiz_fov / 2, cols))
        ud = np.sin(np.linspace(-vert_fov / 2, vert_fov / 2, rows))
        for i in range(rows):
            for j in range(cols):
                rho = self.depthImg[i,j]
                x = rho * lr[j]
                y = rho * ud[i]
                z = rho
                cartImg[i,j,:] = [x, -y, z]
        return cartImg
    
    # @timeit
    def find_ground(self):
        rows, cols = Config.get("dimensions")
        ground = np.zeros((rows, cols), dtype=bool)
        outline_matrix = np.asarray([self.cartImg[:,:,2], self.cartImg[:,:,1], np.full((rows, cols), -1)]).T[:,::-1,:]
        buckets_matrix = np.zeros((cols, rows, 3)) # [x, y, total]
        numba_functs.find_ground_njit_helper(rows, cols, ground, outline_matrix, buckets_matrix)
        return ground
    
    @timeit
    def find_ground_no_gpu(self):
        ground = np.zeros(self.depthImg.shape, dtype=bool)
        for column in range(self.depthImg.shape[1]):
            length = self.depthImg[:,0].shape[0]
            outline = np.asarray([self.cartImg[:,column,2], self.cartImg[:,column,1], np.full(length, -1)]).T[::-1,:]
            start = np.where(np.min(outline[:,0]) == outline[:,0])[0][0]
            # Debug_Timer.start("put in buckets")
            lda = .25
            max_dist = 1
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
            done = buckets.shape[0] - 1
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
    
    # @timeit
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
        min_rotated = np.min(rotated[:,1])
        max_radius = abs(math.sin(left_most_angle - angle_from_straight)\
            - math.sin(right_most_angle - angle_from_straight)) * min_rotated
        radius = min(radius, max_radius)
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

# @nb.experimental.jitclass([('id', nbt.i8),('outline', nbt.f4[:, :]),\
#     ('segMask', nbt.b1[:, :]), ('in_scope', nbt.b1), ('circle', nbt.f8[:])])
class Object:
    def __init__(self, id, outline, segMask):
        self.id = id
        self.outline = outline
        self.segMask = segMask
        self.in_scope = bool(True)
        self.circle = np.array([0,0,0], dtype=np.float64)

    def set_out_of_scope(self):
        self.in_scope = bool(False)

    def area(self):
        return np.sum(self.segMask)

class Prediction:
    def __init__(self, object):
        self.id = object.id
        self.circles = np.copy(object.circle)[0:2].reshape(1,2)
        self.current_circle = self.circles[-1,:]
        self.samples = 1
        self.total_radius = object.circle[2]
        self.time_in = None
        self.time_out = None
        self.radius = None
        self.starts = None

    def add_circle(self, object, time_delta):
        measurement_trust = Config.get("measurement_trust")
        actual = object.circle[0:2].reshape(1,2)
        self.total_radius += object.circle[2]
        if self.samples == 1:
            self.circles = np.append(self.circles, actual, axis=0)
        else:
            predicted = self.for_compromise(time_delta).reshape(1,2)
            compromise = actual * measurement_trust + predicted * (1 - measurement_trust)
            self.circles = np.append(self.circles, compromise, axis=0)
        self.current_circle = self.circles[-1,:]
        self.samples += 1
        self.radius = self.total_radius / self.samples
        velocity = (self.current_circle - self.circles[-2,:]) / time_delta
        hyp = math.sqrt(velocity[0]**2 + velocity[1]**2)
        addition = velocity * (self.radius + Config.get("safe_distance")) / hyp
        start = self.current_circle + [-addition[0], -addition[1]]
        start0 = np.copy(start + [addition[1], -addition[0]])
        start1 = np.copy(start + [-addition[1], addition[0]])
        self.starts = [start0, start1]
        self.for_compromise = self.lambda_time_in(velocity, self.current_circle)
        self.time_in = [self.lambda_time_in(velocity, start0), self.lambda_time_in(velocity, start1)]
        self.time_out = [self.lambda_time_out(velocity, start0), self.lambda_time_out(velocity, start1)]
    
    def lambda_time_in(self, velocity, start):
        return lambda t : t * velocity + start
    
    def lambda_time_out(self, velocity, start):
        return lambda t : ((t - start) / velocity)[0]

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
            for i in range(cls.total.__len__()):
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