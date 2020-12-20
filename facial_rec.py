# John Strenio
# facial recognition attempt using keras and a cnn

# ================== IMPORT LIBRARIES ==============================

import tensorflow as tf
import numpy as np
import imutils
import cv2

# ignore warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# ================== APPLY TO VIDEO STREAM ==============================

model = tf.keras.models.load_model('saved_model1')

# initialize the video stream and allow the camera sensor to warmup
cap = cv2.VideoCapture(0)

# load face and eye detectors
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
eyes_cascade_glasses = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

# loop over the frames from the video stream
while True:

	# grab the frame 
	ret, output_frame = cap.read()
	detect_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2GRAY)

	# find the faces
	faces_detected = face_cascade.detectMultiScale(detect_frame, scaleFactor=1.1, minNeighbors=5)

	# box faces, set color based on model results
	for face in faces_detected:
		(x, y, w, h) = face

		# cutout and resize faces
		proc_face = detect_frame[y:y+h, x:x+w]
		proc_face = cv2.resize(proc_face, (128,128))

		# feed them to the model
		np_img = np.asarray(proc_face)
		images = list()
		images.append(np_img / 255.0)
		images = np.array(images)
		images = images.reshape(-1, 128, 128, 1)
		probs = model.predict(images)

		if probs[0][0] < 0.5:
			red_sq = cv2.rectangle(output_frame, (x, y), (x+w, y+h), (30,30,240), 1)
			cv2.putText(red_sq, 'stranger confidence: ' + str(round(1-probs[0][0],2)), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30,30,240))
		else:
			green_sq = cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0,240,0), 1)
			cv2.putText(green_sq, 'target confidence: ' + str(round(probs[0][0],2)), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,240,0))

	# show the output frame
	frame = imutils.resize(output_frame, width=1000)
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# close camera
cap.release()
cv2.destroyAllWindows()