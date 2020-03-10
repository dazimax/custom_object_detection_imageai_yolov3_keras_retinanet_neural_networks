# DETECTION OF FACE COVERS INCLUDING FULL FACE HELMETS VIA CCTV CAMERAS WITH COMPUTER VISION AND MACHINE LEARNING 

The purpose of this project was to verify User Authentication and reduce false-positive predictions by improving the effectiveness of detection full-face helmets with computer vision and machine learning technologies. Also, send emergency alerts to the relevant authorities to take necessary actions before crime or robbery occurs in secure premises such as Banks, ATM, Jewellery shops, etc.

# METHODOLOGY

ImageAI & Keras Frameworks will use to build a YoloV3 and RetinaNet custom data-set model base on Region Convolutional Neural Networks (R-CNN) by using computer vision and machine learning technologies. 

Custom dataset model trained and validated with sample image datasets of full face helmet images 

As the result, this application will detect and provide prediction result for input image frames of CCTV camera, video footage.

# OBJECT DETECTION ALGORITHM

```
Algorithm 1 - The proposed object detection algorithm
Inputs: Video Path, Output Video Path, Minimum Percentage of Detection Probability
Outputs: Detected Objects, Email Notification send status, SMS Notification send status, Incident Log status

1. Import python libraries for object detection and send notifications
2. Initialize global variables with configuration values
3. Define send SMS alert functionality
4. Define send Email alert functionality
5. Define incident log maintain functionality
6. Define object detection functionality
7.   for end of video input image frames:
8.     process the full face object detection in neural network with custom dataset model
9.     if object detection == True:
10.      call the send SMS alert function
11.      call the send Email alert function
12.    end if
13.    call the log maintain function
14.    end for
15.Call the object detection functionality with created custom dataset model and video input path
```
