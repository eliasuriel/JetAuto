#!/usr/bin/python3
import cv2 as cv
import numpy as np
import onnxruntime as ort

# Classificator class
class modelPredict:
    def __init__(self, model:str, class_list:list, conf_thres:float, cuda:bool) -> None:
        #Initialize attributes
        self.__model = model
        self.__class_list = class_list
        self.__colors = np.random.uniform(0, 255, size=(len(self.__class_list), 3))
        self.__conf = conf_thres
        self.__buildModel(cuda) # Build the model for inference

        # Depth of the object from the camera (m)
        self.__depth = None
        # Horizontal distance of the object from the camera (m)
        self.__horizontal = None
        # Focal distance of the camera (pixels)
        self.__focalLength = 514

    # Define if opencv runs with CUDA or CPU (False = CPU, True = CUDA)
    def __buildModel(self, is_cuda:bool) -> None:
        if is_cuda:
            print("Attempting to use CUDA")
            self.__session = ort.InferenceSession(self.__model, providers = ['CUDAExecutionProvider'])
        else:
            print("Running on CPU")
            self.__session = ort.InferenceSession(self.__model, providers = ['CPUExecutionProvider'])
        # Get the input image shape for the model (width, height)
        shape = self.__session.get_inputs()[0].shape
        self.__inputWidth, self.__inputHeight = shape[2:4]

    # Format image to be used by the model
    def __formatImg(self, img:cv.Mat) -> np.ndarray:
        image = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        image = np.array(cv.resize(image, [self.__inputWidth, self.__inputHeight])) / 255.0 # Resize (input shape) and normalize (0-1)
        image = np.transpose(image, (2, 0, 1)) # Transpose to have: (channels, width, height)
        return np.expand_dims(image, axis=0).astype(np.float32) # Add batch dimension to create tensor (b, c, w, h)

    # Detect objects and get the raw output from the model
    def __detect(self, img:cv.Mat) -> np.ndarray:
        inputs = {self.__session.get_inputs()[0].name: img} # Prepare the input for the model
        preds = self.__session.run(None, inputs) # Perform inference
        return np.squeeze(preds[0]) # Remove batch dimension

    # Wrap the detection processing
    def __wrapDetection(self, modelOutput:np.ndarray, object:str) -> tuple:
        # Initialize lists
        class_ids, boxes, scores = [], [], []

        # Calculate the scaling factor
        x_factor = self.__imgWidth / self.__inputWidth
        y_factor = self.__imgHeight / self.__inputHeight

        # Iterate over the model output
        rows = modelOutput.shape[0]
        for r in range(rows):
            row = modelOutput[r]
            
            # Check if the object confidence is greater than the threshold
            if row[4] > self.__conf:
                classes_scores = row[5:]
                class_id = np.argmax(classes_scores)
                max_score = classes_scores[class_id]
                
                # Check if the score is greater than the threshold and if the detected object is the desired one
                if (max_score > self.__conf) and (self.__class_list[class_id] == object):
                    x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() # Get the bounding box coordinates

                    # Scale the bounding box coordinates
                    left = (x - 0.5 * w) * x_factor
                    top = (y - 0.5 * h) * y_factor
                    width, height = w*x_factor, h*y_factor

                    # Append the results to the lists
                    class_ids.append(class_id)
                    scores.append(max_score)
                    boxes.append(np.array([int(left), int(top), int(width), int(height)]))

        # Apply non-maximum suppression to suppress overlapping boxes
        indices = cv.dnn.NMSBoxes(boxes, scores, self.__conf, 0.5)

        # Get the final results
        final_class_ids, final_boxes, final_scores = [], [], []
        for i in indices:
            final_class_ids.append(class_ids[i])
            final_boxes.append(boxes[i])
            final_scores.append(scores[i])
        return final_class_ids, final_boxes, final_scores

    # Start the detection process
    def _startDetection(self, imgData:list, object:str, width:float) -> tuple:
        # Decode the image
        img = cv.imdecode(np.frombuffer(imgData, np.uint8), cv.IMREAD_COLOR)
        # Get the image shapes
        self.__imgHeight, self.__imgWidth = img.shape[:2]

        # Perform the detection
        formatImg = self.__formatImg(img)
        outs = self.__detect(formatImg)
        class_ids, boxes, scores = self.__wrapDetection(outs, object)

        if class_ids:
            # Get the detected object with the highest score
            index = np.argmax(scores)
            # Decompress the bounding box coordinates
            x, y, w, h = boxes[index]
            color = self.__colors[class_ids[index]]
            # Draw the bounding box for the object
            cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
            # Draw the label background
            cv.rectangle(img, (x, y - 15), (x + w, y + 15), color, -1)
            # Draw the label and confidence of the object
            cv.putText(img, f"{object}: {scores[index]:.3f}", (x, y + 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, cv.LINE_AA)

            # Calculate the distance of the object from the camera
            self.__depth = width * self.__focalLength / w

            # Calculate the horizontal distance of the object from the camera
            self.__horizontal = (x + (w - self.__imgWidth)/2) * self.__depth / self.__focalLength
        else:
            self.__depth = None
            self.__horizontal = None
        return cv.imencode('.jpg', img)[1].tobytes()

    # Get the X coordinate of the object
    def getDepth(self) -> float:
        return self.__depth

    # Get the Y coordinate of the object
    def getHorizontal(self) -> float:
        return self.__horizontal
    

    ############OBJECT CLASSIFICATORY


    #!/home/sr_tavo/venv/bin/python3
import rospy
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Point
from sensor_msgs.msg import CompressedImage
# import sys
# sys.path.append("/home/tavo/Ciberfisicos_ws/src/JetAuto2/scripts")
from Classes.modelPredict import modelPredict

class objectClassificator:
    def __init__(self, model:str, classes:list, conf_threshold:float, cuda:bool) -> None:
        # Instance of the class modelPredict
        self.__model = modelPredict(model, classes, conf_threshold, cuda)

        # Initialize the variables
        self.__img = None
        self.__object = None # Desired Class name of the object
        self.__objWidth, self.__objHeight = 0.06, 0.12 # Object dimensions (m)

        self.__armCoords = {"x":0.22, "y":0.0, "z":self.__objHeight/2 - 0.035} # Current end effector coordinates of the arm (m)
        self.__lastY = 0.0 # Last y coordinate of the object (m)
        self.__tolerance = 10 # Tolerance for capturing the object movement (iterations)
        self.__errorTolerance = 1e-2 # Tolerance for the error (m)
        self.__grab = False # Flag for check if the object is grabbed

        self.__kp = 0.7 # Proportional constant for the arm movement

        # Compressed image message
        self.__compressedImg = CompressedImage()
        self.__compressedImg.format = "jpeg"

        # Initialize the subscribers and publishers
        rospy.Subscriber("/usb_cam/image_raw/compressed", CompressedImage, self.__imageCallback) # Get the image from the camera
        rospy.Subscriber("/object/class", String, self.__classCallback) # Get the class of the object
        rospy.Subscriber("/object/drop", Bool, self.__dropCallback) # Get the drop flag
        self.__coord_pub = rospy.Publisher("/object/coords", Point, queue_size = 1) # Publish the coordinates of the object (m)
        self.__grab_pub = rospy.Publisher("/object/grab", Bool, queue_size = 1) # Publish if the object is ready to be grabbed
        self.__detection_pub = rospy.Publisher("/usb_cam/model_prediction/compressed", CompressedImage, queue_size = 10) # Publish the image with the prediction

    # Callback funtion for the image
    def __imageCallback(self, msg:CompressedImage) -> None:
        self.__img = msg.data

    # Callback function for the class name
    def __classCallback(self, msg:String) -> None:
        self.__object = msg.data
 
    def __dropCallback(self, msg:Bool) -> None:
        self.__grab = False if msg.data else self.__grab
        self.__armCoords["y"] = 0.0

    # Start the model classification
    def _startModel(self) -> None:
        if self.__img is not None and not self.__grab:
            # Detect on current frame
            decodedImg = self.__model._startDetection(self.__img, self.__object, self.__objWidth)
            y, depth = self.__model.getHorizontal(), self.__model.getDepth() # Get the coordinates of the object

            # Move the arm
            self.__moveArm(y, depth) if y is not None else None

            # Publish the compressed image
            self.__compressedImg.data = decodedImg
            self.__detection_pub.publish(self.__compressedImg)

    def __moveArm(self, y:float, depth:float) -> None:
        # Tolerance is increased if the object has not moved
        self.__tolerance += 1 if (-1e-3 < (self.__lastY - y) < 1e-3 or not self.__tolerance) else -1
        self.__lastY = y

        # Check if the object has not moved for "n" iterations
        if (self.__tolerance > 3):
            # Check if the y coordinate is close to zero (middle of the image)
            if not(-self.__errorTolerance < y < self.__errorTolerance):
                self.__armCoords["y"] += y*self.__kp
                self.__tolerance = 0
                self.__coord_pub.publish(self.__armCoords["x"], self.__armCoords["y"], self.__armCoords["z"])
            # Check if the x coordinate is close enough to grab the object
            else:
                x = (depth**2 - self.__armCoords["z"]**2)**0.5
                if x < 0.17:
                    self.__grab, self.__object = True, None
                    self.__tolerance = 0
                    self.__grab_pub.publish(True)
                    rospy.sleep(1.0)
                    self.__coord_pub.publish(self.__armCoords["x"]+(x-0.08), self.__armCoords["y"], self.__armCoords["z"]+0.025)

    # Stop Condition
    def _stop(self) -> None:
        print("Stopping classificator node")

if __name__ == '__main__':
    # Initialise and Setup node
    rospy.init_node("Classificator")

    # Initialize the rate
    rate = rospy.Rate(rospy.get_param("rateClass", default = 15))

    # Get the parameters
    model = rospy.get_param("model/path", default = "../Model/bestV5-25e.onnx")
    class_list = rospy.get_param("classes/list", default = ["Fanta", "Pepsi", "Seven"])
    conf = rospy.get_param("confidence/value", default = 0.5)
    cuda = rospy.get_param("isCuda/value", default = False)

    # Create the instance of the class
    classificator = objectClassificator(model, class_list, conf, cuda)

    # Shutdown hook
    rospy.on_shutdown(classificator._stop)

    # Run the node
    print("The Classificator is Running")
    while not rospy.is_shutdown():
        try:
            classificator._startModel()
        except rospy.ROSInterruptException as ie:
            rospy.loginfo(ie) # Catch an Interruption
        rate.sleep()