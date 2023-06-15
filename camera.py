import numpy as np
import cv2
from threading import Thread
import time
import jetson.inference
import jetson.utils

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(predictions, frameWidth, frameHeight, confThreshold, nmsThreshold):
    """
    This function removes low-confidence bounding boxes from network 
    predictions using non-maxima suppression. It iterates over the predictions,
    filters out boxes below the confidence threshold, and stores the relevant
    information. Afterward, it applies non-maxima suppression to eliminate 
    redundant boxes based on the provided thresholds. The function returns 
    the count of remaining bounding boxes. Example usage:
    
    count = postprocess(predictions, frameWidth, frameHeight, confThreshold, nmsThreshold)

    """
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in predictions:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - frameWidth / 2)
                top = int(center_y - frameHeight / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences.
    result = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

    return len(result)


def get_output_layers(net):
    """
    Get the names of the output layers from a neural network.
    
    Parameters:
    - net: neural network object (e.g., OpenCV DNN network)
    
    Returns:
    - output_layers: list of output layer names
    
    This function retrieves the names of the output layers from a neural
    network that uses the OpenCV DNN framework. It obtains the layer names
    using the getLayerNames method of the network object and retrieves the 
    indices of the unconnected output layers using the getUnconnectedOutLayers 
    method. Finally, it returns the corresponding output layer names as a list.
    
    Example usage:
    ```
    net = cv2.dnn.readNet(model_file, config_file)
    output_layers = get_output_layers(net)
    ```
    """
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def get_class_from_preds(predictions):
    """
    Parameters
    ----------
    predictions : scores for the predictions from yolo model

    Returns
    -------
    str
        the class predicted as a string

    """
    highest_index = np.argmax(predictions)
    if highest_index == 0:
        return "Class 0"
    elif highest_index == 1:
        return "Class 1"
    elif highest_index == 2:
        return "Class 2"
    
class Camera(Thread):
    def __init__(self):
        # Call the Thread class's init function
        Thread.__init__(self)

        self.img_class_left = "other"
        self.img_class_right = "other"

        self.last_trigger_time = 0

        self.ready = False

        ####  RESOLUTION FOR IMAGE CLASSIFICATION - Resnet18
        self.classification_img_width = 96  # Your image classification image width
        self.classification_img_height = 228  # Your image classification image height


        ####  RESOLUTION FOR OBJECT DETECTION - Yolo
        self.detection_img_width = 224  # Your object detection image width
        self.detection_img_height = 352  # Your object detection image height

        #### RESOLUTION FOR CAMERA
        self.camera_width_resolution = 1280
        self.camera_heigth_resolution = 720
        
        # Initialize the parameters for the object detection model
        self.detThreshold = 0.2  # Confidence threshold
        self.nmsThreshold = 0.4  # Non-maximum suppression threshold

        #### USE A COUNTER FOR WHEN IMAGE IS NOT RECEIVED, GIVES WARNING ONCE IT REACHES A MAX COUNT
        self.condition_count = 0
        self.condition_count_max = 300
        self.count_condition = True

        self.trigger_detection = False

	# Uses a yolo model for object detection of citrus fruits
        self.yoloweights = "<path to model .weights>"
        self.yoloconfig = "<path to model .cfg>"

	# Connect to NVIDIA cameras in CSI MIPI ports
        self.cam0 = "nvarguscamerasrc sensor_id=0 ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, width=(int)1280, height=(int)720, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"
        self.cam1 = "nvarguscamerasrc sensor_id=1 ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, width=(int)1280, height=(int)720, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"

    def load_classification_model(self):

        #### CLASSIFICATION MODEL
        self.classificator = jetson.inference.imageNet(
            argv=[
                "--model=<path to .onnx file>",
                "--labels=<path to labels.txt file>",
                "--input-blob=input_0",
                "--output-blob=output_0",
            ]
        )
        print("Loaded classification model")

        self.ready = True

    def load_detection_model(self):
        # read pre-trained model and config file
        self.detector = cv2.dnn.readNet(self.yoloweights, self.yoloconfig)
        self.detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        print("Loaded detection model")

    def disconnect(self):
        self.cap_left.release()
        self.cap_right.release()

    def connect(self):
        self.cap_left = cv2.VideoCapture(self.cam1, cv2.CAP_GSTREAMER)
        self.cap_right = cv2.VideoCapture(self.cam0, cv2.CAP_GSTREAMER)

        # Check if the webcam is opened correctly
        if not self.cap_left.isOpened() or not self.cap_right.isOpened():
            ##            raise IOError("Cannot open camera in address: "+str(self.adr))
            print("Cannot open cameras")
            self.condition = False
        else:
            print("Cameras connected")
            self.condition = True

    def classification_resize(self):
        # RESIZES FRAME FOR IMAGE CLASSIFICATION
        classification_image_left = cv2.resize(
            self.frame_left, (self.classification_img_width, self.classification_img_height)
        )
        classification_image_right = cv2.resize(
            self.frame_right, (self.classification_img_width, self.classification_img_height)
        )

        self.classification_image_left = (
            cv2.cvtColor(classification_image_left, cv2.COLOR_RGB2RGBA).astype(np.float32)
        )
        self.classification_image_right = (
            cv2.cvtColor(classification_image_right, cv2.COLOR_RGB2RGBA).astype(np.float32)
        )

    def detection_resize(self):
        # RESIZES FRAME FOR OBJECT DETECTION
        self.detection_image_left = cv2.resize(
            self.frame_left, (self.detection_img_width, self.detection_img_height)
        )
        self.detection_image_right = cv2.resize(
            self.frame_right, (self.detection_img_width, self.detection_img_height)
        )

    def run_classification(self):
        self.classification_resize()

        img = jetson.utils.cudaFromNumpy(self.classification_image_left)
        predictions = self.classificator.Classify(img)

        self.img_class_left = get_class_from_preds(predictions)

        img = jetson.utils.cudaFromNumpy(self.classification_image_right)
        predictions = self.classificator.Classify(img)

        self.img_class_right = get_class_from_preds(predictions)

    def get_classification(self):
        return self.img_class_left, self.img_class_right

    def get_detection(self, last):
        if self.detections != last:
            return self.detections
        else:
            return [None, None]

    def run_detection(self):


        scale = 0.00392

        self.detection_resize()

        # create input blob
        blob = cv2.dnn.blobFromImage(
            self.detection_image_left,
            scale,
            (self.detection_img_width, self.detection_img_height),
            (0, 0, 0),
            True,
            crop=False,
        )
        # set input blob for the network
        self.detector.setInput(blob)
        # run inference through the network
        # and gather predictions from output layers
        predictions = self.detector.forward(get_output_layers(self.detector))

        count_left = postprocess(
            predictions,
            self.detection_img_width,
            self.detection_img_height,
            self.detThreshold,
            self.nmsThreshold,
        )

        # create input blob
        blob = cv2.dnn.blobFromImage(
            self.detection_image_right,
            scale,
            (self.detection_img_width, self.detection_img_height),
            (0, 0, 0),
            True,
            crop=False,
        )
        # set input blob for the network
        self.detector.setInput(blob)
        # run inference through the network
        # and gather predictions from output layers
        predictions = self.detector.forward(get_output_layers(self.detector))

        count_right = postprocess(
            predictions,
            self.detection_img_width,
            self.detection_img_height,
            self.detThreshold,
            self.nmsThreshold,
        )

        self.detections = [count_left, count_right]

        self.trigger_detection = False
        self.last_trigger_time = time.perf_counter()

    def update_trigger(self, position = None):
        
        if position == None:
            #### IF NO SPEED MEASUREMENT AVAILABLE, DO BY TIME
            newtime = time.perf_counter() - self.last_trigger_time
            if self.gps == False and newtime >= self.triggerDelay:
                self.trigger_detection = True
        else:
            #### IF SPEED MEASUREMENT AVAILABLE
            self.travel_distance += position
            if self.travel_distance >= self.required_distance and self.trigger_detection == False:
                self.trigger_detection = True

    def run(self):
        time.sleep(2)
        print("Cameras started")
        while True:
            if self.condition:
                ret0, frame_left = self.cap_left.read()
                ret1, frame_right = self.cap_right.read()

                if ret0 and ret1:
                    self.frame_left = cv2.rotate(frame_left, cv2.ROTATE_90_CLOCKWISE)
                    self.frame_right = cv2.rotate(frame_right, cv2.ROTATE_90_CLOCKWISE)

                    self.run_classification()

                    self.condition_count = 0

                    if self.trigger_detection:
                        self.run_detection()

                else:
                    self.condition_count += 1
                    time.sleep(1)

                    if self.condition_count > self.condition_count_max:
                        self.condition = False
            else:
                print("Cameras disconnected. Trying to reconnect...")
                self.disconnect()
                time.sleep(3)

                self.connect()


if __name__ == "__main__":

    cap = Camera()
    cap.connect()
    cap.load_classification_model()
    cap.load_detection_model()

    cap.start()
