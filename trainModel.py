# import the necessary packages
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

class TrainModel():
	def __init__(self):
		self.InitialLR = .0005
		self.EPO = 24
		self.batch_size = 36
		self.Directory = r"D:\Grad School\Fall 2021\Data mining\Group Project\Face-Mask-Detection\dataset"
		self.Categories = ["with_mask", "without_mask"]
		self.data = []
		self.labels = []
		self.Processes = None
		self.Hd = None
		self.xTrain, self.xTest, self.yTrain, self.yTest = None, None,None,None
	def StoreInfo(self):
		for category in self.Categories:
			P = os.path.join(self.Directory, category)
			for im in os.listdir(P):
				Image_Path = os.path.join(P, im)
				images = Preprocess.load_img(Image_Path, target_size=(230, 230))
				images = Preprocess.img_to_array(images)
				images = mobilenet.preprocess_input(images)
				self.data.append(images)
				self.labels.append(category)
		#return data, labels

	def Construct_Head_model(self,Model): 
		Processing = Layers.AveragePooling2D(pool_size=(8, 8))(Model.output)
		Processing = Layers.Flatten(name="flatten")(Processing)
		Processing = Layers.Dropout(0.5)(Layers.Dense(128, activation="relu")(Processing))		
		self.Processes = Layers.Dense(2, activation="softmax")(Processing)

	def plot(self):
		arr = numpy.arange(0, self.EPO)
		plt.style.use("ggplot")
		plt.figure()
		plt.plot(arr, self.Hd.history["loss"], label="trainLoss")
		plt.plot(arr, self.Hd.history["accuracy"], label="trainAccuracy")
		plt.plot(arr, self.Hd.history["val_accuracy"], label="valueAccuracy")
		plt.plot(arr, self.Hd.history["val_loss"], label="valueLoss")
		plt.title("Loss and Accuracy")
		plt.xlabel("Epoch No.")
		plt.ylabel("Accuracy/Loss")
		plt.legend(loc="lower left")
		plt.savefig("plot.png")

	def LabelBinarizer(self):
		self.lb = SKPreprocess.LabelBinarizer()
		self.labels = self.lb.fit_transform(self.labels)
		self.labels = utils.to_categorical(self.labels)

		self.data = numpy.array(self.data, dtype="float32")
		self.labels = numpy.array(self.labels)

		(self.xTrain, self.xTest, self.yTrain, self.yTest) = SKSelecton.train_test_split(self.data, self.labels,
			test_size=0.25, stratify=self.labels, random_state=48)
	def TrainHelper(self,op):
		print("Compiling")
		op = optimizer.Adam(learning_rate=self.InitialLR, decay=(self.InitialLR /self.EPO))
		self.mdl.compile(loss="binary_crossentropy", optimizer=op,metrics=["accuracy"])

		# train the head of the network
		print("Training...")
		#check here
		batch_size = self.batch_size
		self.Hd = self.mdl.fit(self.agm.flow(self.xTrain, self.yTrain, batch_size=batch_size),steps_per_epoch=len(self.xTrain) // batch_size,validation_data=(self.xTest, self.yTest),validation_steps=len(self.xTest) // batch_size,epochs=self.EPO)
		# make predictions on the testing set
		print("Evaluating")
		self.predict = self.mdl.predict(self.xTest, batch_size=self.batch_size)
		print("Almost done :)")

		# for each image in the testing set we need to find the index of the
		# label with corresponding largest predicted probability
		self.predict = numpy.argmax(self.predict, axis=1)

		# show a nicely formatted classification report
		print(SKMetrics.classification_report(self.yTest.argmax(axis=1), self.predict,
			target_names=self.lb.classes_))

	def TrainModelandExport(self):
		self.agm = Preprocess.ImageDataGenerator(rotation_range=25,zoom_range=0.18,width_shift_range=0.25,height_shift_range=0.22,
			shear_range=0.18,horizontal_flip=True,fill_mode="nearest")

		# load the MobileNetV2 network, ensuring the head FC layer sets are
		# left off
		self.bMdl = App.MobileNetV2(include_top=False, input_tensor=Layers.Input(shape=(230, 230, 3)))


		self.Construct_Head_model(self.bMdl)


		# place the head FC model on top of the base model (this will become
		# the actual model we will train)

		self.mdl = ML.Model(inputs=self.bMdl.input, outputs=self.Processes )

		# loop over all layers in the base model and freeze them so they will
		# *not* be updated during the first training process
		for layer in self.bMdl.layers:
			layer.trainable = False

		# compile our model
		#print("Compiling")
		op = optimizer.Adam(learning_rate=self.InitialLR, decay=(self.InitialLR /self.EPO))
		self.TrainHelper(op)
		
		# serialize the model to disk
		print("Saving FaceMaskDetection Model--")
		self.mdl.save("./Output/FaceMaskDetection.model", save_format="h5")

		# plot the training loss and accuracy
		self.plot()
def runprogram():
	# initialize the initial learning rate, number of epochs to train for,
	# and batch size
	Model = TrainModel()


	# grab the list of images in our dataset directory, then initialize
	# the list of data (i.e., images) and class images

	Model.StoreInfo()
	Model.LabelBinarizer()
	Model.TrainModelandExport()

if __name__=="__main__":
	runprogram()



