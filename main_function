import socket
import math
import random
import json
import numpy as np
#from datapoint import datapoint
import time
#import reset_current_datapoints as rcd
#import trilat as tri
from Main_Class import sensor_fusion

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address = ('127.0.0.1', 5678)
message = str(random.randint(1, 101)) + ' 51'
sent = sock.sendto(bytes(message.encode()), server_address)


class Anchor:
    def __init__(self, id, x, y, z):
        self.id = id
        self.x = x
        self.y = y
        self.z = z

class Tag:
    def __init__(self, ip, user):
        self.ip = ip
        self.user = user
dplist = {}
malist = {}
input_data = open("Config.json", "r")
json_data = json.load(input_data)
anchors = [json_data["anchors1"]]
tags = [json_data["tags1"]]
values_all = []
id, x, y, z = [], [], [], []
ip, user = [], []
thresh = 2
socket_for_sending = socket.socket()
host = '192.168.50.145'
port = 12345
socket_for_sending.bind((host,port))
socket_for_sending.listen(1)
angle1 = 0.0
angle2 = 0.0
angle3 = 0.0

for i in anchors:
    for j in range(len(i)):
        id.append(i[j]["id"])
        x.append(i[j]["x"])
        y.append(i[j]["y"])
        z.append(i[j]["z"])
        malist[i[j]["id"]] = [[thresh]]

for i in tags:
    for j in range(len(i)):
        ip.append(i[j]["ip"])
        user.append(i[j]["user"])


def first_init():
    # Receive response
    data, server = sock.recvfrom(4096)
    data = data.decode("utf-8")
    data = data.strip('{ }')
    data = data.split()
    if((not (int(data[1]) == 255)) and len(data) == 14):
        angle1 = float(data[6])
        angle2 = float(data[7])
        angle3 = float(data[8])
    else:
        first_init()
def main():
    sensr = sensor_fusion(angle1,angle2,angle3,time.time())

    while 1:
        try:
            # Send data
            sent = sock.sendto(bytes(message.encode()), server_address)
            # Receive response
            data, server = sock.recvfrom(4096)
            data = data.decode("utf-8")
            data = data.strip('{ }')
            data = data.split()
            if((not (int(data[1]) == 255)) and len(data) == 14):
                sensr.set_angles(alpha=float(data[6]),phi=float(data[7]),theta=float(data[8]),time_T=time.time())
                gravity_vec = np.array([float(data[3]),float(data[4]),float(data[5])])
                print(np.linalg.inv(sensr.Orientation)*gravity_vec.transpose())
                time.sleep(0.1)
        finally:
            pass


