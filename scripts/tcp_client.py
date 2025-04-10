import socket
import glob
import numpy as np
from SCA_classes import Segmentation_Collision_Avoidance, Debug_Timer

HOST = "192.168.1.113"  # The server's hostname or IP address
PORT = 65432  # The port used by the server

sca = Segmentation_Collision_Avoidance("config")
rgbImgs = sorted(glob.glob("../oakd_data/streetViewData_view1/rgb_video/*.png"))
depthImgs = sorted(glob.glob("../oakd_data/streetViewData_view1/disparity/*.png"))
sca.add_image_file(rgbImgs[0], depthImgs[0])
rgbImg = sca.window.frame.rgbImg
depthImg = sca.window.frame.depthImg
flat = np.append(rgbImg, depthImg).tobytes()

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    Debug_Timer.reset()
    Debug_Timer.start("data transfer")
    # s.sendall(flat)
    s.sendall(b"Hello, world")
    data = s.recv(1024)
    Debug_Timer.stop("data transfer")

Debug_Timer.print_all()
print(f"Received {data!r}")