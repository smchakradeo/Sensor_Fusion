import math
import matplotlib.pyplot as plt
import numpy as np

class sensor_fusion(object):
    def __init__(self,alpha,phi,theta,time_T):
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.alpha_dot = alpha
        self.phi_dot = phi
        self.theta_dot = theta
        self.time_T = time_T
        self.R_W = np.identity(3)
        self.R_V = np.identity(3)
        self.R_U = np.identity(3)
        self.Rotation = np.identity(3)
        self.Orientation = np.identity(3)
        self.gravity = np.array([0,0,0]).transpose()


    def set_angles(self,alpha,phi,theta,time_T):
        time = time_T - self.time_T
        self.time_T = time/1000
        self.roll = self.time_T*((alpha+self.alpha_dot)/2)
        self.pitch = self.time_T * ((phi + self.phi_dot) / 2)
        self.yaw = self.time_T * ((theta + self.theta_dot) / 2)
        self.alpha_dot = alpha
        self.phi_dot = phi
        self.theta_dot = theta
        # ----------------------------------
        self.R_U[1][1] = math.cos(self.roll)
        self.R_U[1][2] = -math.sin(self.roll)
        self.R_U[2][1] = math.sin(self.roll)
        self.R_U[2][2] = math.cos(self.roll)
        #----------------------------------
        self.R_V[0][0] = math.cos(self.pitch)
        self.R_V[0][2] = math.sin(self.pitch)
        self.R_V[2][0] = -math.sin(self.pitch)
        self.R_V[2][2] = math.cos(self.pitch)
        # ----------------------------------
        self.R_W[0][0] = math.cos(self.yaw)
        self.R_W[0][1] = -math.sin(self.yaw)
        self.R_W[1][0] = math.sin(self.yaw)
        self.R_W[1][1] = math.cos(self.yaw)
        # ----------------------------------
        self.Rotation = np.matmul(np.matmul(self.R_W,self.R_V),self.R_U)
        Orientation = np.matmul(self.Rotation,self.Orientation)
        self.Orientation = Orientation
        self.gravity =  np.matmul(np.linalg.inv(self.Orientation),np.array([0,0,9.87]).transpose())

