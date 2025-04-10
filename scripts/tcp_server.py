import socket
import numpy as np

HOST = "192.168.1.113"  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)
# RGB_BYTES_SIZE = 388800
# RGB_SHAPE = (270, 480, 3)
# RGB_DATA_TYPE = np.uint8
# DEPTH_BYTES_SIZE = 518400
# DEPTH_SHAPE = (270, 480)
# DEPTH_DATA_TYPE = np.float32

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print(f"Connected by {addr}")
        while True:
            data = conn.recv(1024)
            if not data:
                break
            print(data.__len__())
            conn.sendall(b"Hello, world")