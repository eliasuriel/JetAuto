#!/usr/bin/env python3 

#from types import NoneType
import rospy
import numpy as np

class InverseKinematics:
    def __init__(self, jointSizes) -> None:
        #Obtener valores de los parametros (longitudes de los brazos)
        self.L1 = jointSizes[0]
        self.L2 = jointSizes[1]
        self.L3 = jointSizes[2]
        self.L4 = jointSizes[3]
        self._joints_size = [self.L1, self.L2, self.L3, self.L4]

    def link_sizes(self, L1, L2, L3, L4):
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.L4 = L4
        self._joints_size = [self.L1, self.L2, self.L3, self.L4]

    #INVERSE KINEMATICS!
    def ik_solver(self, xyz, rpy):
        
        #Valores deseados
        x = xyz[0]
        y = xyz[1]
        yaw = xyz[2]

        pwx = np.sqrt(x**2 + y**2) - self.L4
        pwy = yaw - self.L1

        q1 = np.arctan2(y, x)

        s = np.sqrt(pwx**2 + pwy**2)

        D = (s**2 - self.L2**2 - self.L3**2) / (2 * self.L2 * self.L3)

        if abs(D) < 1:
            q3 = -np.arctan2(np.sqrt(1 - D**2), D)
        else:
            D = 0.99
            q3 = -np.arctan2(np.sqrt(1 - D**2), D)

        alfa = np.arctan2(pwy, pwx)
        gamma = np.arctan2(self.L3 * np.sin(q3), self.L2 + self.L3 * np.cos(q3))
        q2 = alfa - gamma
        q4 = -(q2 + q3)

        q1 = q1
        q2 = 1.57-q2
        q3 = -q3
        q4 = -q4

        if (q1 < -2.35 or q1 > 2.35):
            q1 = None
        if (q2 < -1.23 or q2 > 1.54):
            q2 = None
            if (q3 < -1.2 or q3 > 1.2):
                q3 = None
        if (q3 < -0.1 or q3 > 2.3):
            q3 = None
        if (q4 < -1.45 or q4 > 1.4):
            q4 = None

        if (q1 == None or q2==None or q3==None or q4==None):
            qs= None
        else:
            qs = [q1, q2, q3, q4]
        #print("ángulo cinematica inversa: " + str(qs))
        
        return qs
 