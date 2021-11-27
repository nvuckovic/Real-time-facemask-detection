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

def StoreInfo(CATEG,DIR):
	data = []
	labels = []
	for cat in CATEG:
		P = os.path.join(DIR, cat)
		for im in os.listdir(P):
			Image_Path = os.path.join(P, im)
			images = Preprocess.load_img(Image_Path, target_size=(230, 230))
			images = Preprocess.img_to_array(images)
			images = mobilenet.preprocess_input(images)
			data.append(images)
			labels.append(cat)
	return data, labels

def Construct_Head_model(Model):
	Processing = Model.output
	Processing = Layers.AveragePooling2D(pool_size=(8, 8))(Processing)
	Processing = Layers.Flatten(name="flatten")(Processing)
	Processing = Layers.Dense(128, activation="relu")(Processing)
	Processing = Layers.Dropout(0.5)(Processing)
	Processing = Layers.Dense(2, activation="softmax")(Processing)
	return Processing

def plot(N):
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(numpy.arange(0, N), Hd.history["loss"], label="train_loss")
	plt.plot(numpy.arange(0, N), Hd.history["val_loss"], label="v_loss")
	plt.plot(numpy.arange(0, N), Hd.history["accuracy"], label="train_accuracy")
	plt.plot(numpy.arange(0, N), Hd.history["val_accuracy"], label="v_accuracy")
	plt.title("Training (Loss and Accuracy)")
	plt.xlabel("Epoch No.")
	plt.ylabel("Accuracy/Loss")
	plt.legend(loc="lower left")
	plt.savefig("plot.png")
if __name__=="__main__":
	# initialize the initial learning rate, number of epochs to train for,
	# and batch size
	InitialLR = .0005
	EPO = 25
	BAT_SIZ = 36

	DIR = r"D:\Grad School\Fall 2021\Data mining\Group Project\Face-Mask-Detection\dataset"
	CATEG = ["with_mask", "without_mask"]

	# grab the list of images in our dataset directory, then initialize
	# the list of data (i.e., images) and class images
	print("loading images.")

	data = []
	labels = []

	data,labels = StoreInfo(CATEG,DIR)

	# perform one-hot encoding on the labels
	lb = SKPreprocess.LabelBinarizer()
	labels = lb.fit_transform(labels)
	labels = utils.to_categorical(labels)

	data = numpy.array(data, dtype="float32")
	labels = numpy.array(labels)

	(X_train, X_test, Y_train, Y_test) = SKSelecton.train_test_split(data, labels,
		test_size=0.25, stratify=labels, random_state=48)

	# construct the training image generator for data augmentation
	agm = Preprocess.ImageDataGenerator(rotation_range=25,zoom_range=0.18,width_shift_range=0.25,height_shift_range=0.22,
		shear_range=0.18,horizontal_flip=True,fill_mode="nearest")

	# load the MobileNetV2 network, ensuring the head FC layer sets are
	# left off
	bMdl = App.MobileNetV2( include_top=False, input_tensor=Layers.Input(shape=(230, 230, 3)))


	Processing = Construct_Head_model(bMdl)


	# place the head FC model on top of the base model (this will become
	# the actual model we will train)
	mdl = ML.Model(inputs=bMdl.input, outputs=Processing)

	# loop over all layers in the base model and freeze them so they will
	# *not* be updated during the first training process
	for layer in bMdl.layers:
		layer.trainable = False

	# compile our model
	print("Compiling")
	op = optimizer.Adam(learning_rate=InitialLR, decay=(InitialLR / EPO))
	mdl.compile(loss="binary_crossentropy", optimizer=op,metrics=["accuracy"])

	# train the head of the network
	print(" Training...")
	Hd = mdl.fit(agm.flow(X_train, Y_train, batch_size=BAT_SIZ),steps_per_epoch=len(X_train) // BAT_SIZ,validation_data=(X_test, Y_test),validation_steps=len(X_test) // BAT_SIZ,epochs=EPO)

	# make predictions on the testing set
	print("Evaluating")
	PID = mdl.predict(X_test, batch_size=BAT_SIZ)

	# for each image in the testing set we need to find the index of the
	# label with corresponding largest predicted probability
	PID = numpy.argmax(PID, axis=1)

	# show a nicely formatted classification report
	print(SKMetrics.classification_report(Y_test.argmax(axis=1), PID,
		target_names=lb.classes_))

	# serialize the model to disk
	print("Saving FaceMaskDetection Model--")
	mdl.save("FaceMaskDetection.model", save_format="h5")

	# plot the training loss and accuracy
	N = EPO
	plot(N)
