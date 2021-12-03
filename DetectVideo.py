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

		faces,locations,predictions = [],[],[]
		for i in range(0, detections.shape[2]):

			temp = detections[0, 0, i, 2]


			if temp > 0.5:

				display = detections[0, 0, i, 3:7] * numpy.array([width, height, width, height])
				(startX, startY, endX, endY) = display.astype("int")


				startX = max(0, startX)
				startY = max(0, startY)
				endX = min(width - 1, endX)
				endY = min(height - 1, endY)

				face = self.Process(frame,startY,endY,startX,endX)


				faces.append(face)
				locations.append((startX, startY, endX, endY))

		if len(faces) > 0:

			faces = numpy.array(faces, dtype="float32")
			predictions = maskMdl.predict(faces, batch_size=32)


		return (locations, predictions)

	def Face_Mask_Detection(self,frame, FaceReg, maskMdl):

		(height, width) = frame.shape[:2]
		image = cv2.dnn.blobFromImage(frame, 1.0, (230, 230),
			(104.0, 177.0, 123.0))

		FaceReg.setInput(image)
		detections = FaceReg.forward()


		return self.FDHelper(detections,height,width,frame,maskMdl)
	
	def displayBoundingbox(self,frame,label,startX,startY,endx,endy,color):
		cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endx, endy), color, 2)
		cv2.imshow("Frame", frame)
		return cv2.waitKey(1) & 0xFF

	def FaceDetection(self,locations,predictions,frame): 
		for (display, pred) in zip(locations, predictions):
				(startX, startY, endX, endY) = display
				(mask, withoutMask) = pred

				label = "No Mask" if mask > withoutMask else "Mask"
				color = (0, 0, 255) if label == "Mask" else (255, 0, 0)

				label = f"{label}: {round(max(mask, withoutMask) * 100,2)}%"


				self.key = self.displayBoundingbox(frame,label,startX,startY,endX,endY,color)

				if self.key == ord("q"):
					return False


def runProgram(vid_str):
	DetectUser = Detect() 

	FaceReg = cv2.dnn.readNet(DetectUser.pPath, DetectUser.wPath)
	cv2.startWindowThread()
	run = True

	while run:

		DetectUser.frames = vid_str.read()
		DetectUser.frames  = imutils.resize(DetectUser.frames , width=800)

		(DetectUser.locations, DetectUser.predictions) = DetectUser.Face_Mask_Detection(DetectUser.frames , FaceReg, DetectUser.maskMdl)

		if (DetectUser.FaceDetection(DetectUser.locations,DetectUser.predictions,DetectUser.frames) == False):
			run = False
		else: 
			(DetectUser.locations, DetectUser.predictions) = DetectUser.Face_Mask_Detection(DetectUser.frames , FaceReg, DetectUser.maskMdl)
	cv2.destroyAllWindows()
	vid_str.stop()


if __name__=="__main__":
	print("Starting Detect Mask script")
	vid_str = Video.VideoStream(src=0).start()
	t1 = threading.Thread(target=runProgram, args=(vid_str,))
	t1.start()
	t1.join()


