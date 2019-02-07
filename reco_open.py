# python3 reconnaissance_objets.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
 
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
 
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
 

CLASSES = ["arriere-plan", "avion", "velo", "bateau",
	"autobus", "moto", "train",]

# Classes avion
UNDER_CLASSES_PLANES = ["Commercial", "Tourisme", "Militaire"]
COLORS_planes = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Classes bateau
UNDER_CLASSES_BOAT = ["Commercial", "Tourisme", "Militaire"]
COLORS_boat = np.random.uniform(0, 128, size=(len(CLASSES), 3))

# Classes autobus
UNDER_CLASSES_TRUCK = ["Autobus", "Bus", "Camion"]
COLORS_truck = np.random.uniform(0, 64, size=(len(CLASSES), 3))

# Classes moto
UNDER_CLASSES_MOTO = ["Autobus", "Bus", "Camion"]
COLORS_moto = np.random.uniform(0, 32, size=(len(CLASSES), 3))

print(" ...chargement du modèle...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("...démarrage de la Picamera...")
vs = VideoStream(usePiCamera=True, resolution=(1600, 1200)).start()
time.sleep(2.0)
fps = FPS().start()
 
while True:

	frame = vs.read()
	frame = imutils.resize(frame, width=800)
 
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)
 
	net.setInput(blob)
	detections = net.forward()
 

	for i in np.arange(0, detections.shape[2]):

		confidence = detections[0, 0, i, 2]
		
		if confidence > args["confidence"]:

			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
 
			label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
	 
			cv2.imwrite("detection.png", frame)
 			 
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	if key == ord("q"):
		break
 
	fps.update()
 
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
 
cv2.destroyAllWindows()
vs.stop()
