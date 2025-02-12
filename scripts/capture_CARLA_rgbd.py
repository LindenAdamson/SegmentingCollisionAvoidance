import sys
import numpy as np
import argparse
import logging
import random
from PIL import Image

try:
    sys.path.append('../CARLA/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg')
except IndexError:
    pass
import carla # type: ignore

saved_lone = []
saved_depth = []

# OAK-D values
camera_hz = 10
camera_freq = 1. / float(camera_hz)
camera_pixels_x = 1920 #160 #320 * 2 #1280
camera_pixels_y = 1080 #120 #240 * 2 #800
camera_horiz_fov = 72

collided = False

imgname = "1080p"

class imu:
    def __init__(self):
        self.x = None
        self.y = None
        self.z = None
    def set(self, xyz):
        self.x = xyz.x
        self.y = xyz.y
        self.z = xyz.z
    def get_x(self):
        return self.x
    def get_y(self):
        return self.y
    def get_z(self):
        return self.z

class CarlaControlNode:
    def __init__(self):
        # Get argvs
        argparser = argparse.ArgumentParser(
            description=__doc__)
        argparser.add_argument(
            "--run",
            "-r",
            default=0,
            type=int
        )
        argparser.add_argument(
            '--host',
            metavar='H',
            default='127.0.0.1',
            help='IP of the host server (default: 127.0.0.1)')
        argparser.add_argument(
            '-p', '--port',
            metavar='P',
            default=2000,
            type=int,
            help='TCP port to listen to (default: 2000)')
        args = argparser.parse_args()
        #client.load_world('Town01')

        # Connect to CARLA
        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(2.0)
        # self.world = self.client.load_world('Town01')
        # self.world.set_weather(carla.WeatherParameters.ClearNoon)
        self.world = self.client.get_world()

        # Spawn ego vehicle
        ego_bp = self.world.get_blueprint_library().find('vehicle.tesla.model3')
        ego_bp.set_attribute('role_name','ego')
        print('\nEgo role_name is set')
        ego_color = random.choice(ego_bp.get_attribute('color').recommended_values)
        ego_bp.set_attribute('color',ego_color)
        print('\nEgo color is set')

        spawn_points = self.world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if 0 < number_of_spawn_points:
            random.shuffle(spawn_points)
            ego_transform = spawn_points[0]
            self.vehicle = self.world.spawn_actor(ego_bp,ego_transform)
            print('\nEgo is spawned')
        else: 
            logging.warning('Could not found any spawn points')
            
        self.vehicle.set_autopilot(True)

        # Spawn camera
        cam_bp = None
        cam_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute("image_size_x",str(camera_pixels_x))
        cam_bp.set_attribute("image_size_y",str(camera_pixels_y))
        cam_bp.set_attribute("fov",str(camera_horiz_fov))
        cam_bp.set_attribute("sensor_tick",str(camera_freq))
        cam_location = carla.Location(2,0,1)
        cam_rotation = carla.Rotation(0,0,0)
        cam_transform = carla.Transform(cam_location,cam_rotation)
        ego_lone = self.world.spawn_actor(cam_bp,cam_transform,attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)

        cam_bp = None
        cam_bp = self.world.get_blueprint_library().find('sensor.camera.depth')
        cam_bp.set_attribute("image_size_x",str(camera_pixels_x))
        cam_bp.set_attribute("image_size_y",str(camera_pixels_y))
        cam_bp.set_attribute("fov",str(camera_horiz_fov))
        cam_bp.set_attribute("sensor_tick",str(camera_freq))
        cam_location = carla.Location(2,0,1)
        cam_rotation = carla.Rotation(0,0,0)
        cam_transform = carla.Transform(cam_location,cam_rotation)
        ego_depth = self.world.spawn_actor(cam_bp,cam_transform,attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)
        
        def save_lone(image):
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            global saved_lone
            saved_lone.append(array)

        def save_depth(image):
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            depth = np.empty([array.shape[0], array.shape[1]])
            depth[:,:] = ((array[:,:,0] + array[:,:,1] * 256.0 + array[:,:,2] * 256.0 * 256.0)/((256.0 * 256.0 * 256.0) - 1))
            # depth[:,:,1] = depth[:,:,0]
            # depth[:,:,2] = depth[:,:,0]
            
            depth = depth * 1000 # farthest depth that can be seen is 1km
            #depth = np.clip(depth, 0, 12) # farthest distance the oakd can infer is 12m
            #depth = depth * 255 / 12
            global saved_depth
            saved_depth.append(depth)

        ego_lone.listen(lambda image: save_lone(image))
        ego_depth.listen(lambda image: save_depth(image))

        global saved_lone
        global saved_depth

        frame = 0
        while True:
            world_snapshot = self.world.wait_for_tick() # make sure the camera has called back

            while not (len(saved_lone) > 0 and len(saved_depth) > 0):
                pass
            
            img_lone = saved_lone[0]
            img_depth = saved_depth[0]
            saved_lone = []
            saved_depth = []

            im = Image.fromarray(img_lone)
            im.save("data/rgb" + imgname + str(frame) + ".jpeg")
            np.savetxt('data/depth' + imgname + str(frame) + '.out', img_depth, delimiter=',')
            print("frame: " + str(frame))
            transform = np.empty([6])
            transform[0] = self.vehicle.get_transform().location.x
            transform[1] = self.vehicle.get_transform().location.y
            transform[2] = self.vehicle.get_transform().location.z
            transform[3] = self.vehicle.get_transform().rotation.pitch
            transform[4] = self.vehicle.get_transform().rotation.yaw
            transform[5] = self.vehicle.get_transform().rotation.roll
            np.savetxt('data/transform' + imgname + str(frame) + '.out', transform, delimiter=',')

            frame = frame + 1

if __name__ == '__main__':
    try:
        node = CarlaControlNode()
    except KeyboardInterrupt:
        pass