#!/usr/bin/python3
import cv2 as cv
from modelPredict import modelPredict

def main():
    try:
        modelo = modelPredict("/home/jetauto/cf_ws/src/depth_cam/scripts/best.onnx",["Naranja","Roja","Verde"],0.6,True)
        cam = cv.VideoCapture(0)

        while cam.isOpened():
            ret,frame = cam.read()

            if not ret:
                print("No se pudo joven")

            elif frame.size > 0:
                modelo._startDetection(frame,"Roja", 0.5)
            else:
                print("No se pudo abrir")
            
            if cv.waitKey(1) == ord('q'):
                cam.release()
                print("stop")
                exit(1)
    except KeyboardInterrupt as ki:
        print(ki)
        cam.release()
        print("stop")
        exit(1)


if __name__ == '__main__':
    main()
