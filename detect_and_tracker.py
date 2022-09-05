# USAGE
# python object_tracker.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
import numpy as np
import argparse
from imutils import face_utils
import imutils
import time
import cv2
import dlib

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", default="deploy.prototxt",
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default="res10_300x300_ssd_iter_140000.caffemodel",
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.8,
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
time.sleep(1.0)

font = cv2.FONT_HERSHEY_SIMPLEX
# loop over the frames from the video stream
person = 0
cnt = 0
gender = "Undefined"
age = "Undefined"
while True:
	# read the next frame from the video stream and resize it
    ret,frame = cap.read()
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame = imutils.resize(frame, width=400)
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    
    detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(gray, 0)

    new_rects = []

    for rect in rects:
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        if y<0:
            print("a")
            continue
        new_rects.append((x, y, x + w, y + h))

        face_img = frame[y:y+h, x:x+w].copy()

        blob2 = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        
        if cnt%10 == 0:
            # Predict gender
            gender_net.setInput(blob2)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]
            # Predict age
            age_net.setInput(blob2)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]
        
        overlay_text = "%s, %s" % (gender, age)
        cv2.putText(frame, overlay_text ,(x,y), font, 1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h),(0, 255, 0), 2)

    save_frame = frame.copy()       
	# update our centroid tracker using the computed set of bounding
	# box rectangles
    objects = ct.update(new_rects)

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
    cnt += 1
    key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cap.release()
cv2.destroyAllWindows()
