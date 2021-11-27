#import the necessary packages
import numpy
import imutils, time, cv2, os
import imutils.video as Video
import tensorflow.keras.applications.mobilenet_v2  as mobilenet 
import tensorflow.keras.preprocessing.image as Preprocess
import tensorflow.keras.models as ML
import threading
# load our serialized face detector model from disk
pPath = r"face_detector\deploy.prototxt"
wPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
FaceReg = cv2.dnn.readNet(pPath, wPath)
import multiprocessing as mp
# load the face mask detector model from disk
maskMdl = ML.load_model("FaceMaskDetection.model")
def grabFrame(vid_str):
	frames = vid_str.read()
	frames = imutils.resize(frames, width=400)
	return frames

def Process(frame,startY,endY,startX,endX):
		face = frame[startY:endY, startX:endX]
		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		face = cv2.resize(face, (224, 224))
		face = Preprocess.img_to_array(face)
		return mobilenet.preprocess_input(face)

def FDHelper(detections,h,w,frame):
	
	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locations = []
	predictions = []
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * numpy.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = Process(frame,startY,endY,startX,endX)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locations.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:

		faces = numpy.array(faces, dtype="float32")
		predictions = maskMdl.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locationss
	return (locations, predictions)

def detect_and_predict_mask(frame, FaceReg, maskMdl):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	FaceReg.setInput(blob)
	detections = FaceReg.forward()
	print(detections.shape)


	return FDHelper(detections,h,w,frame)
	
def displayBoundingbox(frame,label,startX,startY,endx,endy,color):
	cv2.putText(frame, label, (startX, startY - 10),
	cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
	cv2.rectangle(frame, (startX, startY), (endx, endy), color, 2)
	# show the output frame
	cv2.imshow("Frame", frame)
	return cv2.waitKey(1) & 0xFF

def FaceDetection(locations,predictions,frame): 
	for (box, pred) in zip(locations, predictions):
			# unpack the bounding box and predictions
			(startX, startY, endX, endY) = box
			(mask, withoutMask) = pred

			# determine the class label and color we'll use to draw
			# the bounding box and text
			label = "Mask" if mask > withoutMask else "No Mask"
			color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

			# include the probability in the label
			label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)


			key = displayBoundingbox(frame,label,startX,startY,endX,endY,color)

			# if the `q` key was pressed, break from the loop
			if key == ord("q"):
				break

def runProgram(vid_str):
	# loop over the frames from the video stream
	while True:
		# grab the frame from the threaded video stream and resize it
		# to have a maximum width of 400 pixels
		frame = grabFrame(vid_str)

		# detect faces in the frame and determine if they are wearing a
		# face mask or not
		(locations, predictions) = detect_and_predict_mask(frame, FaceReg, maskMdl)
		# loop over the detected face locations and their corresponding
		# locations
		FaceDetection(locations,predictions,frame)







if __name__=="__main__":
	print("Starting Video Stream--")
	vid_str = Video.VideoStream(src=0).start()
	t1 = threading.Thread(target=runProgram, args=(vid_str,))
	t1.start()
	t1.join()
	#runProgram(vid_str)

	# do a bit of cleanup
	cv2.destroyAllWindows()
	vid_str.stop()
