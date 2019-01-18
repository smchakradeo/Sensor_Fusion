import socket
import math
import random
import json
import numpy as np
import time
from Main_Class import sensor_fusion
class Main_Class(object):
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_address = ('127.0.0.1', 5678)
        self.message = str(random.randint(1, 101)) + ' 51'
        self.first_init()
        self.angle1 = 0.0
        self.angle2 = 0.0
        self.angle3 = 0.0
        self.x_states = np.array([0,0,0,0,0,0]).transpose()
        self.U = np.array([0,0,0]).transpose()

    def first_init(self):
        # Receive response
        sent = self.sock.sendto(bytes(self.message.encode()), self.server_address)
        data, server = self.sock.recvfrom(4096)
        data = data.decode("utf-8")
        data = data.strip('{ }')
        data = data.split()
        if((not (int(data[1]) == 255)) and len(data) == 14):
            self.angle1 = float(data[6])
            self.angle2 = float(data[7])
            self.angle3 = float(data[8])

            #Update the state vector and control vector here
        else:
            self.first_init()

    def motion_model(self,X_states,U,sensr):
        T = sensr.time_T
        A = np.array([[1,0,0,T,0,0],
                      [0,1,0,0,T,0],
                      [0,0,1,0,0,T],
                      [0,0,0,1,0,0],
                      [0,0,0,0,1,0],
                      [0,0,0,0,0,1]])

        B = np.array([[(T**2)/2,0,0],
                      [0,(T**2)/2,0],
                      [0,0,((T**2)/2)],
                      [T,0,0],
                      [0,T,0],
                      [0,0,T]])

        rotation = np.array([[1,0,0,0,0,0],
                             [0,1,0,0,0,0],
                             [0,0,1,0,0,0],
                             [0,0,0,sensr.Rotation[0,:]],
                             [0,0,0,sensr.Rotation[1,:]],
                             [0,0,0,sensr.Rotation[2,:]]])

        U_vec = 0.5*((self.U + U))
        self.U = U
        self.x_states = np.matmul(A,np.matmul(rotation,self.x_states))+np.matmul(B,np.matmul(sensr.Orientation,U_vec))


    def main(self):
        sensr = sensor_fusion(self.angle1,self.angle2,self.angle3,time.time())
        while 1:
            try:
                # Send data
                sent = self.sock.sendto(bytes(self.message.encode()), self.server_address)
                # Receive response
                data, server = self.sock.recvfrom(4096)
                data = data.decode("utf-8")
                data = data.strip('{ }')
                data = data.split()
                if((not (int(data[1]) == 255)) and len(data) == 14):
                    sensr.set_angles(alpha=float(data[6]),phi=float(data[7]),theta=float(data[8]),time_T=time.time())
                    U_vec = np.array([float(data[3]),float(data[4]),float(data[5])]).transpose()-sensr.gravity
                    self.motion_model(self.x_states,U_vec,sensr)
                    time.sleep(0.1)
            finally:
                pass


