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

        # Y coordinate of the object
        self.__y = None

    # Define if opencv runs with CUDA or CPU (False = CPU, True = CUDA)
    def __buildModel(self, is_cuda:bool) -> None:
        if is_cuda:
            print("Attempting to use CUDA")
            self.__session = ort.InferenceSession(self.__model, providers=['CUDAExecutionProvider'])
        else:
            print("Running on CPU")
            self.__session = ort.InferenceSession(self.__model, providers=['CPUExecutionProvider'])
        # Get the input image shape for the model (width, height)
        shape = self.__session.get_inputs()[0].shape
        self.__inputWidth, self.__inputHeight = shape[2:4]

    # Format image to be used by the model
    def __formatImg(self, img:cv.Mat) -> np.ndarray:
        image = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        image = np.array(cv.resize(image, [self.__inputWidth, self.__inputHeight])) / 255.0 # Resize (input shape) and normalize (0-1)
        image = np.transpose(image, (2, 0, 1)) # Transpose to (channels, width, height)
        return np.expand_dims(image, axis=0).astype(np.float32) # Add batch dimension to create tensor (b, c, w, h)

    # Detect objects in the image
    def __detect(self, img:cv.Mat) -> np.ndarray:
        inputs = {self.__session.get_inputs()[0].name: img} # Get the input name of the model
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
        result_class_ids, result_boxes, result_scores = [], [], []
        for i in indices:
            result_class_ids.append(class_ids[i])
            result_boxes.append(boxes[i])
            result_scores.append(scores[i])
        return result_class_ids, result_boxes, result_scores

    # Start the detection process
    def _startDetection(self, imgData:list, object:str, width:float) -> None:
        # Decode the imagesudo apt-get remove python3.8
        img = cv.imdecode(np.frombuffer(imgData, np.uint8), cv.IMREAD_COLOR)
      
        # Get the image shapes
        self.__imgHeight, self.__imgWidth = img.shape[:2]
        
        # Perform the detection
        formatImg = self.__formatImg(img)
        outs = self.__detect(formatImg)
        class_ids, boxes, scores = self.__wrapDetection(outs, object)
        
        # Draw the bounding boxes and labels
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

            ref_pix = 155
            ref_meters = 0.245
            line_hor = int(x + w) - int(x)
            #print(line_hor)
            
            self.__depth = ( ref_pix* ref_meters) / line_hor
            print(self.__depth)

            # Calculate pixels to meters ratio
            PixToMeters = width / w
            # Calculate the Y coordinate of the object
            self.__y = (x + (w - self.__imgWidth)/2)*PixToMeters
            print(self.__y)
        else:
            self.__depth = None
            self.__y = None
        return cv.imencode('.jpg', img)[1].tobytes()

        

    # Get the Y coordinate of the object
    def getY(self) -> float:
        return self.__y
    
    # Get the Y coordinate of the object
    def getdepth(self) -> float:
        return self.__depth
    
  
