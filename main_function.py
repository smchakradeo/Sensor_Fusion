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
        self.ini_ori = np.identity(3)
        self.x_states = np.array([0, 0, 0, 0, 0, 0], float).transpose()
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
            vector_orig = np.array([float(data[3]), float(data[4]), float(data[5])]).transpose()
            vector_fin = np.array([0, 0, 9.8]).transpose()
            self.initial_orientation(vector_orig, vector_fin)
            # Update the state vector and control vector here
        else:
            self.first_init()

    def initial_orientation(self, vector_orig, vector_fin):

        """Calculate the rotation matrix required to rotate from one vector to another.
        For the rotation of one vector to another, there are an infinit series of rotation matrices
        possible.  Due to axially symmetry, the rotation axis can be any vector lying in the symmetry
        plane between the two vectors.  Hence the axis-angle convention will be used to construct the
        matrix with the rotation axis defined as the cross product of the two vectors.  The rotation
        angle is the arccosine of the dot product of the two unit vectors.
        Given a unit vector parallel to the rotation axis, w = [x, y, z] and the rotation angle a,
        the rotation matrix R is::
                      |  1 + (1-cos(a))*(x*x-1)   -z*sin(a)+(1-cos(a))*x*y   y*sin(a)+(1-cos(a))*x*z |
                R  =  |  z*sin(a)+(1-cos(a))*x*y   1 + (1-cos(a))*(y*y-1)   -x*sin(a)+(1-cos(a))*y*z |
                      | -y*sin(a)+(1-cos(a))*x*z   x*sin(a)+(1-cos(a))*y*z   1 + (1-cos(a))*(z*z-1)  |
            @param R:           The 3x3 rotation matrix to update.
            @type R:            3x3 numpy array
            @param vector_orig: The unrotated vector defined in the reference frame.
            @type vector_orig:  numpy array, len 3
            @param vector_fin:  The rotated vector defined in the reference frame.
            @type vector_fin:   numpy array, len 3
        """

        # Convert the vectors to unit vectors.
        vector_orig = vector_orig / np.linalg.norm(vector_orig)
        vector_fin = vector_fin / np.linalg.norm(vector_fin)

        # The rotation axis (normalised).
        axis = np.cross(vector_orig, vector_fin)
        axis_len = np.linalg.norm(axis)
        if axis_len != 0.0:
            axis = axis / axis_len

        # Alias the axis coordinates.
        x = axis[0]
        y = axis[1]
        z = axis[2]

        # The rotation angle.
        angle = math.acos(np.dot(vector_orig, vector_fin))

        # Trig functions (only need to do this maths once!).
        ca = math.cos(angle)
        sa = math.sin(angle)

        # Calculate the rotation matrix elements.
        self.ini_ori[0, 0] = 1.0 + (1.0 - ca) * (x ** 2 - 1.0)
        self.ini_ori[0, 1] = -z * sa + (1.0 - ca) * x * y
        self.ini_ori[0, 2] = y * sa + (1.0 - ca) * x * z
        self.ini_ori[1, 0] = z * sa + (1.0 - ca) * x * y
        self.ini_ori[1, 1] = 1.0 + (1.0 - ca) * (y ** 2 - 1.0)
        self.ini_ori[1, 2] = -x * sa + (1.0 - ca) * y * z
        self.ini_ori[2, 0] = -y * sa + (1.0 - ca) * x * z
        self.ini_ori[2, 1] = x * sa + (1.0 - ca) * y * z
        self.ini_ori[2, 2] = 1.0 + (1.0 - ca) * (z ** 2 - 1.0)

    def motion_model(self, X_states, U, sensr):
        T = sensr.DT
        A = np.array([[1, 0, 0, T, 0, 0],
                      [0, 1, 0, 0, T, 0],
                      [0, 0, 1, 0, 0, T],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1]], float)

        B = np.array([[((T ** 2) / 2), 0, 0],
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

        U_vec = 0.5 * ((self.U + U))
        self.U = U
        orientation = sensr.Orientation
        self.x_states = np.matmul(A, np.matmul(rotation, self.x_states)) + np.matmul(B, np.matmul(orientation, U_vec))
        # print('states: ', self.x_states)

    def main(self):
        sensr = sensor_fusion(self.angle1, self.angle2, self.angle3, self.ini_ori, time.time())
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
                    sensr.set_angles(alpha=float(data[6]), phi=float(data[7]), theta=float(data[8]), time_T=time.time())
                    U_vec = np.subtract(np.array([float(data[3]), float(data[4]), float(data[5])], float).transpose(),
                                        sensr.gravity)
                    # print('Gravity: ', sensr.gravity)
                    print('U_Vec: ', U_vec)
                    # print(sensr.Orientation)
                    self.motion_model(self.x_states, U_vec, sensr)
                    time.sleep(0.1)
            finally:
                pass


obj = main_Class()
obj.main()
