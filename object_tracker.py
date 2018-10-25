# USAGE
# python object_tracker.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", default="deploy.prototxt",
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default="res10_300x300_ssd_iter_140000.caffemodel",
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
(H, W) = (None, None)
# init age and gender parameters 
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list=['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
gender_list = ['Male', 'Female']

# load our serialized model from disk
print("[INFO] loading models...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

age_net = cv2.dnn.readNetFromCaffe(
                        "age_gender_models/deploy_age.prototxt", 
                        "age_gender_models/age_net.caffemodel")
gender_net = cv2.dnn.readNetFromCaffe(
                        "age_gender_models/deploy_gender.prototxt", 
                        "age_gender_models/gender_net.caffemodel")

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")

cap =cv2.VideoCapture(0)


# vs = VideoStream(src=0).start()
time.sleep(2.0)

font = cv2.FONT_HERSHEY_SIMPLEX
# loop over the frames from the video stream
person = 0
while True:
	# read the next frame from the video stream and resize it
    ret,frame = cap.read()
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #frame = imutils.resize(frame, width=400)
    #print(frame.shape)
	# if the frame dimensions are None, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    
	# construct a blob from the frame, pass it through the network,
	# obtain our output predictions, and initialize the list of
	# bounding box rectangles
    blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),(104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    rects = []

	# loop over the detections
    for i in range(0, detections.shape[2]):
		# filter out weak detections by ensuring the predicted
		# probability is greater than a minimum threshold
        if detections[0, 0, i, 2] > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the object, then update the bounding box rectangles list
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            rects.append(box.astype("int"))

			# draw a bounding box surrounding the object so we can
			# visualize it
            (startX, startY, endX, endY) = box.astype("int")
            
            if(startX<0 or startY<0):
                break
            
            face_img = frame[startY:endY, startX:endX].copy()
            
            blob2 = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            
            
            # Predict gender
            gender_net.setInput(blob2)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]
            # Predict age
            age_net.setInput(blob2)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]
            
            overlay_text = "%s, %s" % (gender, age)
            cv2.putText(frame, overlay_text ,(startX,startY), font, 1,(255,0,0),2,cv2.LINE_AA)
            
            
            cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255, 0), 2)
            
    save_frame = frame.copy()       
	# update our centroid tracker using the computed set of bounding
	# box rectangles
    objects = ct.update(rects)

	# loop over the tracked objects
    for (objectID, centroid) in objects.items():
		# draw both the ID of the object and the centroid of the
		# object on the output frame
        text = "ID {}".format(objectID)
        if(objectID>person):
            cv2.imwrite('img/'+str(person)+'.jpg',save_frame)
            person = person+1
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

	# show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cap.release()
cv2.destroyAllWindows()
