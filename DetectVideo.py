#import the necessary packages
import numpy
import imutils, time, cv2, os
import imutils.video as Video
import tensorflow.keras.applications.mobilenet_v2  as mobilenet 
import tensorflow.keras.preprocessing.image as Preprocess
import tensorflow.keras.models as ML
import threading
import multiprocessing as mp

class Detect():
	def __init__(self):
		self.pPath = r"Detector\initialize.prototxt"
		self.wPath = r"Detector\wPath.caffemodel"
		self.maskMdl = ML.load_model("./Output/FaceMaskDetection.model")
		self.frames = []
		self.locations, self.predictions = [], []

	def Process(self,frame,startY,endY,startX,endX):
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (230, 230))
			face = Preprocess.img_to_array(face)
			return mobilenet.preprocess_input(face)
	def FDHelper(self,detections,height,width,frame,maskMdl):
	
		# initialize our list of faces, their corresponding locations,
		# and the list of predictions from our face mask network
		faces = []
		locations = []
		predictions = []
		for i in range(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with
			# the detection
			temp = detections[0, 0, i, 2]

			# filter out weak detections by ensuring the confidence is
			# greater than the minimum confidence
			if temp > 0.5:
				# compute the (x, y)-coordinates of the bounding box for
				# the object
				display = detections[0, 0, i, 3:7] * numpy.array([width, height, width, height])
				(startX, startY, endX, endY) = display.astype("int")

				# ensure the bounding boxes fall within the dimensions of
				# the frame
				startX = max(0, startX)
				startY = max(0, startY)
				endX = min(width - 1, endX)
				endY = min(height - 1, endY)
				# extract the face ROI, convert it from BGR to RGB channel
				# ordering, resize it to 224x224, and preprocess it
				face = self.Process(frame,startY,endY,startX,endX)

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

	def detect_and_predict_mask(self,frame, FaceReg, maskMdl):
		# grab the dimensions of the frame and then construct a blob
		# from it
		(height, width) = frame.shape[:2]
		image = cv2.dnn.blobFromImage(frame, 1.0, (230, 230),
			(104.0, 177.0, 123.0))

		# pass the blob through the network and obtain the face detections
		FaceReg.setInput(image)
		detections = FaceReg.forward()


		return self.FDHelper(detections,height,width,frame,maskMdl)
	
	def displayBoundingbox(self,frame,label,startX,startY,endx,endy,color):
		cv2.putText(frame, label, (startX, startY - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endx, endy), color, 2)
		# show the output frame
		cv2.imshow("Frame", frame)
		return cv2.waitKey(1) & 0xFF

	def FaceDetection(self,locations,predictions,frame): 
		for (display, pred) in zip(locations, predictions):
				# unpack the bounding box and predictions
				(startX, startY, endX, endY) = display
				(mask, withoutMask) = pred

				# determine the class label and color we'll use to draw
				# the bounding box and text
				label = "Mask" if mask > withoutMask else "No Mask"
				color = (50, 68, 230) if label == "Mask" else (250, 36, 7)

				# include the probability in the label
				label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)


				self.key = self.displayBoundingbox(frame,label,startX,startY,endX,endY,color)

				# if the `q` key was pressed, break from the loop
				if self.key == ord("q"):
					return False


def runProgram(vid_str):
	DetectUser = Detect() 

	FaceReg = cv2.dnn.readNet(DetectUser.pPath, DetectUser.wPath)
	cv2.startWindowThread()
	run = True

	# loop over the frames from the video stream
	while run:
		# grab the frame from the threaded video stream and resize it
		# to have a maximum width of 400 pixels
		DetectUser.frames = vid_str.read()
		DetectUser.frames  = imutils.resize(DetectUser.frames , width=800)

		# detect faces in the frame and determine if they are wearing a
		# face mask or not
		(DetectUser.locations, DetectUser.predictions) = DetectUser.detect_and_predict_mask(DetectUser.frames , FaceReg, DetectUser.maskMdl)
		# loop over the detected face locations and their corresponding
		# locations
		if (DetectUser.FaceDetection(DetectUser.locations,DetectUser.predictions,DetectUser.frames) == False):
			run = False
		else: 
			(DetectUser.locations, DetectUser.predictions) = DetectUser.detect_and_predict_mask(DetectUser.frames , FaceReg, DetectUser.maskMdl)
	cv2.destroyAllWindows()
	vid_str.stop()






if __name__=="__main__":
	print("Starting Detect Mask script")
	vid_str = Video.VideoStream(src=0).start()
	t1 = threading.Thread(target=runProgram, args=(vid_str,))
	t1.start()
	t1.join()


