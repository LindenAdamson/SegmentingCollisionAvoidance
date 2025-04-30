import socket
import time

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)

count = 0

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        conn.sendall("1".encode())
        print(f"Connected by {addr}")
        while True:
            data = conn.recv(1024).decode()
            if "s" in data:
                steer = float(data.split("s")[1])
                print(steer)
                count += 1
            elif "b" in data:
                brake = float(data.split("b")[1])
                print(brake)
                count += 1
            if count == 20:
                print("20 reached")
                conn.sendall("0".encode())
                time.sleep(10)
                print("start again")
                conn.sendall("1".encode())
                count = 0