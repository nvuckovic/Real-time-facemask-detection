import os
import numpy 
import matplotlib.pyplot as plt
import tensorflow.keras.preprocessing.image as Preprocess
import tensorflow.keras.applications as App
import tensorflow.keras.models as ML
import tensorflow.keras.optimizers as optimizer
import sklearn.metrics as SKMetrics
import sklearn.model_selection as SKSelecton 
import sklearn.preprocessing as SKPreprocess
import tensorflow.keras.applications.mobilenet_v2  as mobilenet 
import tensorflow.keras.layers as Layers
import tensorflow.keras.utils as utils 
import time
import sys
class TrainModel():
	def __init__(self):
		self.InitialLR = .0005
		self.EPO = 24
		self.batch_size = 36
		self.Folders = r"D:\Grad School\Fall 2021\Data mining\Group Project\Face-Mask-Detection\dataset"
		self.Divisions = ["mask", "No_mask"]
		self.data = []
		self.labels = []
		self.Processes = None
		self.Hd = None
		self.xTrain, self.xTest, self.yTrain, self.yTest = None, None,None,None
	def StoreInfo(self):
		for D in self.Divisions:
			P = os.path.join(self.Folders, D)
			for im in os.listdir(P):
				Path = os.path.join(P, im)
				content = Preprocess.load_img(Path, target_size=(230, 230))
				content = Preprocess.img_to_array(content)
				content = mobilenet.preprocess_input(content)
				self.data.append(content)
				self.labels.append(D)


	def Construct_Head_model(self,Model): 
		Processing = Layers.AveragePooling2D(pool_size=(8, 8))(Model.output)
		Processing = Layers.Flatten(name="flatten")(Processing)
		Processing = Layers.Dropout(0.5)(Layers.Dense(128, activation="relu")(Processing))		
		self.Processes = Layers.Dense(2, activation="softmax")(Processing)
		self.mdl = ML.Model(inputs=self.bMdl.input, outputs=self.Processes )
		for layer in self.bMdl.layers:
			layer.trainable = False
		op = optimizer.Adam(learning_rate=self.InitialLR, decay=(self.InitialLR /self.EPO))
		self.TrainHelper(op)

	def plot(self):
		arr = numpy.arange(0, self.EPO)
		plt.style.use("ggplot")
		plt.figure()
		plt.plot(arr, self.Hd.history["loss"], label="trainLoss")
		plt.plot(arr, self.Hd.history["accuracy"], label="trainAccuracy")
		plt.plot(arr, self.Hd.history["val_accuracy"], label="valueAccuracy")
		plt.plot(arr, self.Hd.history["val_loss"], label="valueLoss")
		plt.legend(loc="center right")
		plt.title("Loss Versus Accuracy")
		plt.xlabel("Epoch #")
		plt.ylabel("Accuracy per Loss")

		plt.savefig("ModelOutput.png")

	def LabelBinarizer(self):
		self.lb = SKPreprocess.LabelBinarizer()


		self.data = numpy.array(self.data, dtype="float32")
		self.labels = numpy.array(utils.to_categorical(self.lb.fit_transform(self.labels)))

		(self.xTrain, self.xTest, self.yTrain, self.yTest) = SKSelecton.train_test_split(self.data, self.labels,
			test_size=0.25, stratify=self.labels, random_state=48)
	def TrainHelper(self,op):

		op = optimizer.Adam(learning_rate=self.InitialLR, decay=(self.InitialLR /self.EPO))
		self.mdl.compile(loss="binary_crossentropy", optimizer=op,metrics=["accuracy"])

		print("Training...")
		batch_size = self.batch_size
		self.Hd = self.mdl.fit(self.agm.flow(self.xTrain, self.yTrain, batch_size=batch_size),steps_per_epoch=len(self.xTrain) // batch_size,validation_data=(self.xTest, self.yTest),validation_steps=len(self.xTest) // batch_size,epochs=self.EPO)
		self.predict = self.mdl.predict(self.xTest, batch_size=self.batch_size)
		print("Almost done :)")


		self.predict = numpy.argmax(self.predict, axis=1)

		print(SKMetrics.classification_report(self.yTest.argmax(axis=1), self.predict,
			target_names=self.lb.classes_))

	def TrainModelandExport(self):
		self.agm = Preprocess.ImageDataGenerator(rotation_range=25,zoom_range=0.18,width_shift_range=0.25,height_shift_range=0.22,
			shear_range=0.18,horizontal_flip=True,fill_mode="nearest")

		self.bMdl = App.MobileNetV2(include_top=False, input_tensor=Layers.Input(shape=(230, 230, 3)))


		self.Construct_Head_model(self.bMdl)


		
		
		print("Saving FaceMaskDetection Model--")
		self.mdl.save("./Output/FaceMaskDetection.model", save_format="h5")

		self.plot()
def runprogram():

	Model = TrainModel()

	Model.StoreInfo()
	Model.LabelBinarizer()
	Model.TrainModelandExport()

if __name__=="__main__":
	runprogram()



