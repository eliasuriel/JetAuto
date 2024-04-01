#!/usr/bin/env python3

import rospy
import numpy as np
from threading import Thread
from std_msgs.msg import Float64
from geometry_msgs.msg import Point
from hiwonder_servo_msgs.msg import JointState
from Classes.inverse_kinematics import InverseKinematics
from Classes.directKinematics import directKinematics

class ArmController:
    def __init__(self):
        
        # JOINTS
        self.joint1_value = 0
        self.joint2_value = 0
        self.joint3_value = 0
        self.joint4_value = 0
        self.actualPos = [0, 0, 0]
        self.finalPos = [0, 0, 0]
        self.finalPosOld = [0, 0, 0]
        self.finalPosReach = False

        rospy.Subscriber('joint1_controller/state', JointState, self.joint1_callback)
        rospy.Subscriber('joint2_controller/state', JointState, self.joint2_callback)
        rospy.Subscriber('joint3_controller/state', JointState, self.joint3_callback)
        rospy.Subscriber('joint4_controller/state', JointState, self.joint4_callback)
        rospy.Subscriber('arm/finalPos', Point, self.finalPos_callback)

        self.joint1_publisher = rospy.Publisher('/joint1_controller/command', Float64, queue_size=10)
        self.joint2_publisher = rospy.Publisher('/joint2_controller/command', Float64, queue_size=10)
        self.joint3_publisher = rospy.Publisher('/joint3_controller/command', Float64, queue_size=10)
        self.joint4_publisher = rospy.Publisher('/joint4_controller/command', Float64, queue_size=10)

        self.armRate = rospy.get_param("samples", default = 100)   #Rate of arm movement
        self.jointSizes = rospy.get_param("links/allLinks/length", default = [0.01, 0.13, 0.13, 0.05]) 
        self.ik_handler = InverseKinematics(self.jointSizes)

        #GRIPPER
        self.gripperPub = rospy.Publisher("/r_joint_controller/command", Float64, queue_size=10)


    def joint1_callback(self, msg):
        self.joint1_value = msg.current_pos
    
    def joint2_callback(self, msg):
        self.joint2_value = msg.current_pos

    def joint3_callback(self, msg):
        self.joint3_value = msg.current_pos

    def joint4_callback(self, msg):
        self.joint4_value = msg.current_pos

    def finalPos_callback(self, msg):
        self.finalPos = [msg.x, msg.y, msg.z]

    def obtainActualPos(self):
        qS = [self.joint1_value, self.joint2_value, self.joint3_value, self.joint4_value]
        qS = [0, -0.7, 2, -1]
        if qS == [0, 0, 0, 0]:
            qS = [0, -0.8, 2.0, -0.1]
        dk = directKinematics(qS, self.jointSizes)
        self.actualPos = dk.cinematica_directa_arm()

    def move(self):
        # self.actualPos = [-.785, ]
        print(self.finalPos)
        print(self.finalPosOld)
        if (self.finalPos != self.finalPosOld):
            trayectory = np.linspace(self.actualPos, self.finalPos, self.armRate)

            index = 0

            while (index != len(trayectory)):
                
                q_vals = self.ik_handler.ik_solver(trayectory[index], [0, 0, 0])

                if q_vals != None:
                    msg_joint1 = Float64()
                    msg_joint2 = Float64()
                    msg_joint3 = Float64()
                    msg_joint4 = Float64()

                    msg_joint1.data = q_vals[0]
                    msg_joint2.data = q_vals[1]
                    msg_joint3.data = q_vals[2]
                    msg_joint4.data = q_vals[3]

                    self.joint1_publisher.publish(msg_joint1)
                    self.joint2_publisher.publish(msg_joint2)
                    self.joint3_publisher.publish(msg_joint3)
                    self.joint4_publisher.publish(msg_joint4)

                index += 1
                rate.sleep()
            self.finalPosOld = self.finalPos
            self.finalPosReach = True

    def closeGripper(self):
        self.closeGrip = 3.14
        self.gripperPub.publish(self.closeGrip)

    def openGripper(self):
        self.openGrip = -3.14
        self.gripperPub.publish(self.openGrip)



if __name__ == '__main__':
    rospy.init_node("ik_node", anonymous=True)
    rate = rospy.Rate(rospy.get_param("rate", default = 1))  # Hz

    arm_controller = ArmController()
    arm_controller.obtainActualPos()
    while not rospy.is_shutdown():
        try:
            print("entre")
            arm_controller.move()
        except rospy.ROSInterruptException as ie:
            rospy.loginfo(ie) # Catch an Interruption
        
        rate.sleep()
