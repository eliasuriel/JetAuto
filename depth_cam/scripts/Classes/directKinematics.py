#!/usr/bin/env python3 

import numpy as np

class directKinematics:
    def __init__(self, q, jointSizes) -> None:
        self.q1 = q[0]
        self.q2 = q[1]
        self.q3 = q[2]
        self.q4 = q[3]

        self.L1 = jointSizes[0]
        self.L2 = jointSizes[1]
        self.L3 = jointSizes[2]
        self.L4 = jointSizes[3]

    def rotar_x(alphai):
        ca = np.cos(alphai)
        sa = np.sin(alphai)
        
        R_x = np.array([[1, 0, 0, 0],
                        [0, ca, -sa, 0],
                        [0, sa, ca, 0],
                        [0, 0, 0, 1]])
    
        return R_x
    
    def rotar_z(thetai):
        cth = np.cos(thetai)
        sth = np.sin(thetai)
        
        R_z = np.array([[cth, -sth, 0, 0],
                        [sth, cth, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
    
        return R_z
    
    def tras_x(ai):
        T_x = np.array([[1, 0, 0, ai],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        
        return T_x


    def tras_z(di):
        T_z = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, di],
                        [0, 0, 0, 1]])
    
        return T_z
    
    def calc_Ai(ai, alphai, di, thetai):
        R_z = directKinematics.rotar_z(thetai)
        T_z = directKinematics.tras_z(di)
        T_x = directKinematics.tras_x(ai)
        R_x = directKinematics.rotar_x(alphai)
    
        Ai = np.dot(R_z, np.dot(T_z, np.dot(T_x, R_x)))
    
        return Ai
    
    def cinematica_directa_arm(self):
        a1 = 0
        alpha1 = np.pi/2
        d1 = self.L1
        theta1 = self.q1

        a2 = self.L2
        alpha2 = 0
        d2 = 0
        theta2 = 1.57-self.q2

        a3 = self.L3
        alpha3 = 0
        d3 = 0
        theta3 = -self.q3

        a4 = self.L4
        alpha4 = 0
        d4 = 0
        theta4 = -self.q4

        A1 = directKinematics.calc_Ai(a1, alpha1, d1, theta1)
        A2 = directKinematics.calc_Ai(a2, alpha2, d2, theta2)
        A3 = directKinematics.calc_Ai(a3, alpha3, d3, theta3)
        A4 = directKinematics.calc_Ai(a4, alpha4, d4, theta4)

        Trans_total = np.dot(np.dot(A1, A2), np.dot(A3, A4))

        return Trans_total[0:3, 3]