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
        self.angle1 = 0.0
        self.angle2 = 0.0
        self.angle3 = 0.0
        self.Phi= np.identity(15,float)
        self.B = np.array([[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],float)
        #self.Pk = np.identity(6)
        #self.Q = np.diag([1.0,1.0,1.0,1.0,1.0,1.0])
        #self.R = np.diag([0.1,0.1,0.1])
        self.H = np.zeros([7,9],float)
        self.Pk = np.diag([0, 0, 0, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0.1, 0.1, 0.1])
        self.Rk = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        self.Qk = np.diag([0.0001, 0.0001, 0.0001, np.zeros([1, 6], float), 0.0001, 0.0001, 0.0001, np.zeros([1, 3], float)])
        self.calib_result = np.array([0,0,0],float).transpose()
        self.ini_ori = np.identity(3)
        self.orientation_predicted = self.ini_ori
        self.orientation_corrected = self.ini_ori
        self.x_states = np.array([0, 0, 0, 0, 0, 0], float).transpose()
        self.x_states_predicted = self.x_states
        self.error_states_predicted = np.zeros([15,1],float)
        self.error_states = np.zeros([15, 1], float)
        self.U = np.array([0, 0, 0,0,0,0], float).transpose()
        self.time_t = 0.0
        self.DT = 0.0
        self.calibrate_count = 0
        self.quat_gy = Quaternion(array=[0,0,0,0])
        self.acc_bias = np.array([1.5,1.5,1.5])
        self.gyro_bias = np.array([1.5,1.5,1.5])
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
        """self.A = np.array([[1, 0, 0, T, 0, 0],
                      [0, 1, 0, 0, T, 0],
                      [0, 0, 1, 0, 0, T],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1]], float)

        self.B = np.array([[((T ** 2) / 2), 0, 0],
                      [0, ((T ** 2) / 2), 0],
                      [0, 0, ((T ** 2) / 2)],
                      [T, 0, 0],
                      [0, T, 0],
                      [0, 0, T]], float)

        rotation = np.array([[1, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0],
                             [0, 0, 0, sensr.Rotation[0][0], sensr.Rotation[0][1], sensr.Rotation[0][2]],
                             [0, 0, 0, sensr.Rotation[1][0], sensr.Rotation[1][1], sensr.Rotation[1][2]],
                             [0, 0, 0, sensr.Rotation[2][0], sensr.Rotation[2][1], sensr.Rotation[2][2]]], float)
        U_vec = (1-0.2)*self.U+0.2*U
        #U_vec = 0.5 * (self.U + U)
        self.U = U
        orientation = sensr.Orientation
        orientation1 = sensr.Orientation_1
        self.x_states = np.matmul(self.A, np.matmul(rotation, self.x_states)) + np.matmul(self.B, np.matmul(orientation, U_vec))
        self.x_states_1 = np.matmul(self.A, np.matmul(rotation, self.x_states_1)) + np.matmul(self.B, np.matmul(orientation1, U_vec))
        #print('states: ', self.x_states)
        #print('states2: ', self.x_states_1)"""


        dOmega = np.array([
            [0, -U[2], U[1]],
            [U[2], 0, -U[0]],
            [-U[1], U[0], 0]
        ], float)
        self.orientation_predicted = np.matmul(self.orientation_corrected, np.matmul((2 * np.identity(3, float) + dOmega*T), np.linalg.inv((2 * np.identity(3, float) - dOmega*T))))
        self.x_states_predicted[0:2,0] = self.x_states[0:2,0]  + T * (self.x_states[3:5,0]) + 0.5*T**2*(self.orientation_predicted*(U[3:5]) - np.array([0.0,0.0,9.8]).transpose())
        self.x_states_predicted[3:5,0] = self.x_states[3:5,0] + T*(self.orientation_predicted*(U[3:5]) - np.array([0.0,0.0,9.8]).transpose())
        print(self.x_states_predicted)
        #S = Quaternion(scalar=0.0, vector=[U[3],U[4],U[5]])
        #qdot = (0.5 * (self.quat_gy * S))
        #quat = self.quat_gy +  (qdot * T)
        #self.quat_gy = quat.normalised
        error_states = self.error_motion_model(self.orientation_predicted,U)
        return error_states

    def error_motion_model(self,orientation,U):
        accn = np.matmul(orientation,U[3:5])
        S = np.array([[0,-accn[2],accn[1]],
                      [accn[2],0,-accn[0]],
                      [-accn[1],accn[0],0]],float)      #The acceleration is bias corrected and transformed to the navigation frame accn = Rot_b_to_n * accn_body

        self.Phi = np.array([[np.identity(3,float), self.DT*orientation,np.zeros([3,9],float)],
                       [np.zeros([3,3],float),np.identity(3,float),np.zeros([3,9],float)],
                       [np.zeros([3,3],float),np.zeros([3,3],float),np.identity(3,float),self.DT*np.identity(3,float),np.zeros([3,3],float)],
                       [-self.DT*S, np.zeros([3,6],float),np.identity(3), self.DT*orientation],
                        [np.zeros([3,12],float),np.identity(3,float)]  ])
        error_states = np.matmul(self.Phi,self.error_states)
        """dPhi = self.error_states[0:2]
        dTheta = np.array([
            [0, dPhi[2], -dPhi[1]],
            [-dPhi[2], 0, dPhi[0]],
            [dPhi[1], -dPhi[0], 0]
        ], float)
        self.orientation = np.matmul(np.matmul((2 * np.identity(3, float) + dTheta), (2 * np.identity(3, float) - dTheta)), self.orientation)"""
        return error_states

    def error_observation_model(self,states):
        self.H = np.array([[0, 0, 1, np.zeros([1,12],float)],
                      [np.zeros([3,3],float),np.identity(3,float),np.zeros([3,9],float)],
                      [np.zeros([3,9],float), np.identity(3,float),np.zeros([3,3],float)]])
        Z = np.matmul(self.H,states)
        return Z

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
            """
            output = np.zeros(len(zk))
            H = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0]])
            # time update (prediction)
            self.x_states = self.motion_model(U_vec,sensr)
            #self.x_states = np.matmul(self.A,self.x_states) + np.matmul(self.B,self.U) #Predict state
            zk_pred = np.matmul(H,self.x_states) # Predict measurement
            self.Pk = np.matmul(self.A,np.matmul(self.Pk,self.A.T)) + self.Q # Predict error covariance
            # measurement update (correction)
            vk = zk - zk_pred # Measurement residual
            S = (np.matmul(H,np.matmul(self.Pk,H.T))) + self.R # Prediction covariance
            Kk = np.matmul(np.matmul(self.Pk,H.transpose()), np.linalg.inv(S)) # Compute Kalman Gain
            self.x_states = self.x_states + np.matmul(Kk , vk) # Update estimate with gain * residual
            self.Pk = np.matmul((1 - np.matmul(Kk,H)),self.Pk) # Update error covariance
            output = self.x_states
            #print('Output: ', output)
            return output"""
            if (flag):          #Not stationary
                #Prediction
                self.error_states = self.motion_model(U_vec)
                prediction = self.error_observation_model(self.error_states)
                self.Pk = np.matmul(self.Phi,np.matmul(self.Pk,self.Phi.T)) + self.Qk
            else:           #Stationary
                # Prediction
                self.error_states = self.motion_model(U_vec)
                prediction = self.error_observation_model(self.error_states)
                self.Pk = np.matmul(self.Phi, np.matmul(self.Pk, self.Phi.T)) + self.Qk
                self.orientation_corrected = self.orientation_predicted
                #Correction
                vk = zk - prediction
                S = (np.matmul(self.H, np.matmul(self.Pk, self.H.T))) + self.Rk
                Kk = np.matmul(np.matmul(self.Pk, self.H.T), np.linalg.inv(S))
                self.error_states = self.error_states + np.matmul(Kk, vk)  # Update estimate with gain * residual
                self.Pk = np.matmul((np.identity(15) - np.matmul(Kk, self.H)), self.Pk)  # Update error covariance
                output = self.error_states
                self.x_states[0:2] = self.x_states_predicted[0:2] - output[6:8]
                self.x_states[3:5] = self.x_states_predicted[3:5] - output[9:11]

                dPhi = self.error_states[0:2]
                dTheta = np.array([
                    [0, dPhi[2], -dPhi[1]],
                    [-dPhi[2], 0, dPhi[0]],
                    [dPhi[1], -dPhi[0], 0]
                ], float)
                self.orientation_corrected = np.matmul(
                    np.matmul((2 * np.identity(3, float) + dTheta), (2 * np.identity(3, float) - dTheta)),
                    self.orientation_predicted)
        except np.linalg.linalg.LinAlgError:
            pass

    def main(self):
        sensr = sensor_fusion(self.ini_ori, self.time_t)
        self.quat = sensr.quat
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
                    magr = np.subtract(np.array([float(data[9]), float(data[10]), float(data[11])]).transpose(),self.calib_result/self.calibrate_count)
                    accn = np.subtract(np.array([float(data[3]), float(data[4]), float(data[5])]),self.error_states[12:14])
                    gyro = np.subtract(np.array([float(data[6]), float(data[7]), float(data[8])]),self.error_states[3:5])
                    U_vec = np.array([gyro,accn])
                    sensr.set_angles(alpha=float(data[6]), phi=float(data[7]), theta=float(data[8]),acc= accn,mag=magr,time_T=float(data[2]))
                    #if abs((np.linalg.norm(accn))-9.8) <=1.5:
                     #   self.kalman_filter()
                    #U_vec = np.subtract(np.array([float(data[3]), float(data[4]), float(data[5])], float).transpose(),
                    #                    sensr.gravity)
                    #xk = self.motion_model(U_vec, sensr)
                    #zk = np.array([0,0,0]).T
                    #final_xyz = self.kalman_filter(zk,U_vec,sensr)
                    #self.filesave(str(final_xyz))


            finally:
                pass


obj = main_Class()
ini_time = time.time()
while(time.time()-ini_time<=6):
    obj.calibration()
print('Callibration Done')
obj.main()
