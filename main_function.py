import socket
import math
import random
import json
import numpy as np
import time
from scipy import signal
from pyquaternion import Quaternion

from Main_Class import sensor_fusion
class main_Class(object):
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_address = ('127.0.0.1', 5678)
        self.message = str(random.randint(1, 101)) + ' 51'
        self.Phi= np.identity(15,float)
        self.B = np.array([[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],float)
        self.H = np.zeros((7,9),float)
        self.Pk = np.diag([0, 0, 0, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0.1, 0.1, 0.1])
        self.Rk = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        arr = np.concatenate((np.array([[0.0001, 0.0001, 0.0001]]),np.zeros((1,6),float),np.array([[0.0001, 0.0001, 0.0001]]),np.zeros((1,3),float)),axis=1)
        self.Qk = np.diag(arr)
        self.calib_result = np.array([0,0,0],float)
        self.ini_ori = np.identity(3)
        self.orientation = self.ini_ori
        self.x_states = np.array([0, 0, 0, 0, 0, 0], float)
        self.error_states= np.zeros((15,1),float)
        self.U = np.array([[0, 0, 0,0,0,0]], float).T
        self.time_t = 0.0
        self.DT = 0.0
        self.calibrate_count = 0
        self.quat_gy = Quaternion(array=[0,0,0,0])
        self.acc_bias = np.array([[0.21,0.06,0.1]]).T
        self.gyro_bias = np.array([[math.radians(-1.5),math.radians(-1.5),math.radians(-5.0)]]).T
        self.z = np.zeros([1,7],float)
        self.first_init()

    def first_init(self):
        # Receive response
        sent = self.sock.sendto(bytes(self.message.encode()), self.server_address)
        data, server = self.sock.recvfrom(4096)
        data = data.decode("utf-8")
        data = data.strip('{ }')
        data = data.split()
        if ((not (int(data[1]) == 255)) and len(data) == 14):
            self.angle1 = float(data[6])
            self.angle2 = float(data[7])
            self.angle3 = float(data[8])
            mag = np.array([(float(data[9])),(float(data[10])),(float(data[11]))]).transpose()
            grav = np.array([(float(data[3])), (float(data[4])), (float(data[5]))]).transpose()
            val = ((float(data[9]))**2+(float(data[10]))**2+(float(data[11]))**2)**0.5
            self.ini_ori[:,2] = grav
            self.ini_ori[:, 1] = np.cross(grav,mag)
            self.ini_ori[:, 0] = np.cross(self.ini_ori[:,1],grav)
            self.ini_ori[:,2] = self.ini_ori[:,2]/np.linalg.norm(self.ini_ori[:,2])
            self.ini_ori[:,1] = self.ini_ori[:,1]/np.linalg.norm(self.ini_ori[:,1])
            self.ini_ori[:,0] = self.ini_ori[:,0]/np.linalg.norm(self.ini_ori[:,0])
            self.time_t = float(data[2])
            self.quat_gy = Quaternion(matrix=self.ini_ori)
            
            # Update the state vector and control vector here
            #print('Initial: ', self.ini_ori)
        else:
            self.first_init()

    

    def calibration(self):
        # Send data
        sent = self.sock.sendto(bytes(self.message.encode()), self.server_address)
        # Receive response
        data, server = self.sock.recvfrom(4096)
        data = data.decode("utf-8")
        data = data.strip('{ }')
        data = data.split()
        if ((not (int(data[1]) == 255)) and len(data) == 14):
            magr = np.array([float(data[9]), float(data[10]), float(data[11])]).transpose()
            self.calib_result += magr
            self.calibrate_count = self.calibrate_count+1
            #time.sleep(0.1)


    def time_update(self,T):
      time = T - self.time_t
      self.DT = time / 1000
      self.time_t =  T

    def filesave(self,data_save):
        #data_save = data_save.strip()
        #data_save = str(data_save).strip('[ ]')
        #data_save = data_save.strip("' '")
        inp1 = open("Data1.txt", "a+")
        inp1.write(data_save)
        inp1.write("\n")
        inp1.close()


    def motion_model(self, U):
        T = self.DT
        print("Time: ", T)
        dOmega = np.array([
            [0, -U[2], U[1]],
            [U[2], 0, -U[0]],
            [-U[1], U[0], 0]
        ])
        #print(dOmega)
        temp = (2*np.identity(3) - dOmega*T)
        inv_var = np.linalg.inv(temp)
        
        self.orientation = (self.orientation.dot((2*np.identity(3) + dOmega*T).dot(inv_var)))
        self.x_states[0:3] = self.x_states[0:3]  + (T * (self.x_states[3:6]) + (0.5*T**2)*(self.orientation.dot(U[3:6])) - np.array([0.0,0.0,9.8]))
        self.x_states[3:6] = self.x_states[3:6] + (T*(self.orientation.dot(U[3:6])) - np.array([0.0,0.0,9.8]))
        #print('x_states: ',self.x_states_predicted)
        #S = Quaternion(scalar=0.0, vector=[U[3],U[4],U[5]])
        #qdot = (0.5 * (self.quat_gy * S))
        #quat = self.quat_gy +  (qdot * T)
        #self.quat_gy = quat.normalised
        #error_states = self.error_motion_model(self.orientation,U)
        #print(self.x_states)
        

    def error_motion_model(self,U):
        accn = (self.orientation.dot(U[3:6]))
        S = np.array([[0,-accn[2],accn[1]],
                      [accn[2],0,-accn[0]],
                      [-accn[1],accn[0],0]])      #The acceleration is bias corrected and transformed to the navigation frame accn = Rot_b_to_n * accn_body

        phi1_1 = np.concatenate((np.identity(3),self.DT*self.orientation),axis=1)
        phi1 = np.concatenate((phi1_1,np.zeros([3,9],float)),axis=1)
        
        phi2_1 = np.concatenate((np.zeros([3,3],float),np.identity(3)),axis=1)
        phi2 = np.concatenate((phi2_1,np.zeros([3,9],float)),axis=1)
        
        phi3_1 = np.concatenate((np.zeros([3,3],float),np.zeros([3,3],float)),axis=1)
        phi3_2 = np.concatenate((phi3_1,np.identity(3)),axis=1)
        phi3_3 = np.concatenate((self.DT*np.identity(3),np.zeros([3,3],float)),axis=1)
        phi3 = np.concatenate((phi3_2,phi3_3),axis=1)
        
        phi4_1 = np.concatenate((-self.DT*S, np.zeros([3,6],float)),axis=1)
        phi4_2 = np.concatenate((phi4_1, np.identity(3)),axis=1)
        phi4 = np.concatenate((phi4_2,self.DT*self.orientation),axis=1)
        
        phi5 = np.concatenate((np.zeros([3,12],float),np.identity(3)),axis=1)
        
        
        self.Phi = np.concatenate((phi1,phi2,phi3,phi4,phi5),axis=0)
        
        self.error_states = self.Phi.dot(self.error_states)
        return self.error_states

    def error_observation_model(self):
        h1 = np.concatenate((np.array([[0,0,1]],float),np.zeros([1,12],float)),axis=1)
        h2 = np.concatenate((np.zeros((3,3),float),np.identity(3,float),np.zeros((3,9),float)),axis=1)
        h3 = np.concatenate((np.zeros((3,9),float), np.identity(3,float),np.zeros((3,3),float)),axis=1)
        """self.H = np.array([[0, 0, 1, np.zeros((1,12),float)],
                      [np.zeros((3,3),float),np.identity(3,float),np.zeros((3,9),float)],
                      [np.zeros((3,9),float), np.identity(3,float),np.zeros((3,3),float)]])"""
        self.H = np.concatenate((h1,h2,h3),axis=0)
        Z = self.H.dot(self.error_states)
        return Z
    def get_heading(self):
        quat = Quaternion(matrix = self.orientation)
        q = quat.elements
        yaw = (math.atan2(2.0 * (q[1] *q[2] - q[0] * q[3]),
                                                       -1+2*(q[0] * q[0] + q[1] * q[1])))
        return yaw

    def kalman_filter(self, U_vec, zk=None, location=None, flag=False):
        """Performs Kalman Filtering on pandas timeseries data.
        :param: zk (pandas timeseries): input data
        :param: xk (np.array): a priori state estimate vector
        :param: A (np.matrix): state transition coefficient matrix
        :param: B (np.matrix): control coefficient matrix
        :param: Pk (np.matrix): prediction covariance matrix
        :param: uk (np.array): control signal vector
        :param: wk (float): process noise (has covariance Q)
        :param: Q (float): process covariance
        :param: R (float): measurement covariance
        :param: H (np.matrix):  transformation matrix
        :return: output (np.array): kalman filtered data
        """
        try:
            print(flag)
            if (not flag):          #Not stationary
                #Prediction
                self.motion_model(U_vec)
                prediction = self.error_observation_model()
                self.Pk = np.matmul(self.Phi,np.matmul(self.Pk,self.Phi.T)) + self.Qk
                self.error_states = np.zeros([15,1],float)
            else:                   #Stationary
                # Prediction
                self.error_states = self.error_motion_model(U_vec)
                prediction = self.error_observation_model()
                self.Pk = np.matmul(self.Phi, np.matmul(self.Pk, self.Phi.T)) + self.Qk
                #Correction
                vk = zk.reshape(-1,1) - prediction
                
                S = (np.matmul(self.H, np.matmul(self.Pk, self.H.T))) + self.Rk
                Kk = np.matmul(np.matmul(self.Pk, self.H.T), np.linalg.inv(S))
                self.error_states = self.error_states + np.matmul(Kk, vk)  # Update estimate with gain * residual
                self.Pk = np.matmul((np.identity(15) - np.matmul(Kk, self.H)), self.Pk)  # Update error covariance
                output = self.error_states
                #print(self.x_states)
                self.x_states[0:3] = self.x_states[0:3] - output[6:9].T.flatten()
                self.x_states[3:6] = self.x_states[3:6] - output[9:12].T.flatten()

                dPhi = self.error_states[0:3]
                dTheta = np.array([
                    [0, dPhi[2], -dPhi[1]],
                    [-dPhi[2], 0, dPhi[0]],
                    [dPhi[1], -dPhi[0], 0]
                ], float)
                self.orientation = np.matmul(
                    np.matmul((2 * np.identity(3) + dTheta), np.linalg.inv(2*np.identity(3) - dTheta)),
                    self.orientation)
        except np.linalg.linalg.LinAlgError:
            pass

    def main(self,T):
        self.time_t = T*1000
        sensr = sensor_fusion(self.ini_ori, self.time_t)
        #self.quat = sensr.quat
        while 1:
            try:
                # Send data
                sent = self.sock.sendto(bytes(self.message.encode()), self.server_address)
                # Receive response
                data, server = self.sock.recvfrom(4096)
                data = data.decode("utf-8")
                data = data.strip('{ }')
                data = data.split()
                if ((not (int(data[1]) == 255)) and len(data) == 14):
					
                    self.time_update(float(data[2]))
                    magr = (np.array([float(data[9]), float(data[10]), float(data[11])]).reshape(-1,1)-(self.calib_result/self.calibrate_count).reshape(-1,1))
                    accn = (np.array([float(data[3]), float(data[4]), float(data[5])]).reshape(-1,1)-self.acc_bias-np.array([self.error_states[12][0],self.error_states[13][0],self.error_states[14][0]]).reshape(-1,1))
                    gyro = (np.array([math.radians(float(data[6])), math.radians(float(data[7])), math.radians(float(data[8]))]).reshape(-1,1)-self.gyro_bias-np.array([self.error_states[3][0],self.error_states[4][0],self.error_states[5][0]]).reshape(-1,1))
                    U_vec = np.concatenate((gyro.T,accn.T),axis=1).flatten()
                    sensr.set_angles(accn,magr,self.DT)
                    yaw = self.get_heading()
                    print("States: ", self.orientation)
                    if(abs(11-abs(np.linalg.norm(accn)))<=2):
                        self.motion_model(U_vec)
                        self.z[0,2] = sensr.yaw_a - yaw
                        self.z[0,3:6] = gyro.flatten()
                        self.kalman_filter(U_vec=U_vec,zk=self.z,location=None,flag=True)
                        
                    else:
                        self.kalman_filter(U_vec=U_vec,zk=None,location=None,flag=False)
  
            finally:
                pass


obj = main_Class()
ini_time = time.time()
while(time.time()-ini_time<=6):
	
    obj.calibration()
print('Callibration Done')
obj.main(time.time())
