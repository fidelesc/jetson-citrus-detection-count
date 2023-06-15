## Camera Object Detection and Classification

This repository contains a Python script for object detection and classification using a YOLO (You Only Look Once) model and a ResNet18 model on an NVIDIA Jetson Xavier NX modeule. The script utilizes two cameras to capture real-time images and applies object detection to count citrus fruits and classification to determine if the image contains a citrus tree or another object.

### Smart Sprayer System

The script in this repository was used in the development of an updated version of a smart sprayer system, as described in the paper by Partel, V., Costa, L., and Ampatzidis, Y. (2021) titled "Smart tree crop sprayer utilizing sensor fusion and artificial intelligence" published in Computers and Electronics in Agriculture.

The smart sprayer system is a groundbreaking innovation that uses sensor fusion and artificial intelligence (AI) to optimize the application of crop sprays in tree plantations. It collects data from a range of sensors including the SICK TiM561-2050101 LiDAR sensor, as well as two cameras for real-time crop imaging. The AI analyzes this data to determine the presence and structure of the tree canopy, counts the fruits in real-time, and then adjusts the operation of the sprayer in real time, enabling it to apply pesticides and fertilizers more efficiently.

By controlling the spray application based on actual tree presence and structure, the smart sprayer system minimizes overspray, reduces the amount of chemicals used, improves the accuracy of application, and can significantly reduce the environmental impact of tree crop farming.

A video showcasing the system in action can be found [here](https://www.youtube.com/watch?v=qRd4g44b2lk). This video demonstrates how the system effectively controls the sprayer operation based on the real-time sensor data, validating the value and effectiveness of sensor fusion and AI in precision agriculture.

### NVIDIA Jetson Xavier NX

This script was created for a sensor fusion embedded system on an [NVIDIA Jetson Xavier NX](https://developer.nvidia.com/embedded/jetson-xavier-nx-devkit). Jetson Xavier NX is a small, powerful computer designed for AI applications and edge computing. The system uses the data processed by this script in conjunction with other sensor data to understand and interact with its environment.

### Dependencies

This script requires the following libraries to be installed:

- [numpy](https://numpy.org/) - for data processing and mathematical operations
- [cv2](https://pypi.org/project/opencv-python/) - for computer vision tasks and image processing
- [jetson.inference](https://github.com/dusty-nv/jetson-inference) - for using AI models on NVIDIA Jetson platforms
- [threading](https://docs.python.org/3/library/threading.html) - for concurrent execution
- [time](https://docs.python.org/3/library/time.html) - for controlling the timing and execution of code

### How to Use

Please note that the script can be customized and integrated into a larger smart farming system to further enhance the automation and efficiency of farming operations.

The script uses information from other sensors (like GPS) to trigger the detection to garantee every analysis in on a new image and avoid double counting. It can work without it if there is no signal or gps sensor.
