import socket
import math
import random
import json
import numpy as np
import time
from Main_Class import sensor_fusion


class main_Class(object):
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_address = ('127.0.0.1', 5678)
        self.message = str(random.randint(1, 101)) + ' 51'
        self.angle1 = 0.0
        self.angle2 = 0.0
        self.angle3 = 0.0
        self.A = np.identity(6)
        self.B = np.array([[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],float)
        self.Pk = np.identity(6)
        self.Q = np.diag([1.0,1.0,1.0,1.0,1.0,1.0])
        self.R = np.diag([0.1,0.1,0.1])
        self.calib_result = np.array([0,0,0],float).transpose()
        self.ini_ori = np.identity(3)
        self.x_states = np.array([0, 0, 0, 0, 0, 0], float).transpose()
        self.x_states_1 = self.x_states
        self.U = np.array([0, 0, 0], float).transpose()
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
            #time.sleep(0.1)


    def filesave(self,data_save):
        #data_save = data_save.strip()
        #data_save = str(data_save).strip('[ ]')
        #data_save = data_save.strip("' '")
        inp1 = open("Data1.txt", "a+")
        inp1.write(data_save)
        inp1.write("\n")
        inp1.close()


    def motion_model(self, U, sensr):
        T = sensr.DT
        self.A = np.array([[1, 0, 0, T, 0, 0],
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
        #print('states2: ', self.x_states_1)
        return self.x_states


    
    def kalman_filter(self,zk,U_vec,sensr):
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
            return output
        except np.linalg.linalg.LinAlgError:
            pass

    def main(self):
        sensr = sensor_fusion(self.ini_ori, time.time())
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
                    magr = np.subtract(np.array([float(data[9]), float(data[10]), float(data[11])]).transpose(),self.calib_result)
                    accn = np.array([float(data[3]), float(data[4]), float(data[5])]).transpose()
                    sensr.set_angles(alpha=float(data[6]), phi=float(data[7]), theta=float(data[8]),acc= accn,mag=magr,time_T=float(data[0]))
                    U_vec = np.subtract(np.array([float(data[3]), float(data[4]), float(data[5])], float).transpose(),
                                        sensr.gravity)
                    #xk = self.motion_model(U_vec, sensr)
                    zk = np.array([0,0,0]).T
                    final_xyz = self.kalman_filter(zk,U_vec,sensr)
                    self.filesave(str(final_xyz))
            finally:
                pass


obj = main_Class()
ini_time = time.time()
while(time.time()-ini_time<=1):
    obj.calibration()
print('Callibration Done')
obj.main()
