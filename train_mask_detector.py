# import the necessary packages
import tensorflow.keras.preprocessing.image as Preprocess
import tensorflow.keras.applications as App
import tensorflow.keras.layers as Layers
import tensorflow.keras.models as ML
import tensorflow.keras.optimizers as optimizer
import matplotlib.pyplot as plt
import numpy 
import os
import tensorflow.keras.applications.mobilenet_v2  as mobilenet 
import tensorflow.keras.utils as utils 
import sklearn.preprocessing as SKPreprocess
import sklearn.model_selection as SKSelecton 
import sklearn.metrics as SKMetrics

def StoreInfo(CATEGORIES,DIRECTORY):
	data = []
	labels = []
	for cat in CATEGORIES:
		P = os.path.join(DIRECTORY, cat)
		for im in os.listdir(P):
			Image_Path = os.path.join(P, im)
			images = Preprocess.load_img(Image_Path, target_size=(224, 224))
			images = Preprocess.img_to_array(images)
			images = mobilenet.preprocess_input(images)
			data.append(images)
			labels.append(cat)
	return data, labels

def Construct_Head_model(baseModel):
	headModel = baseModel.output
	headModel = Layers.AveragePooling2D(pool_size=(7, 7))(headModel)
	headModel = Layers.Flatten(name="flatten")(headModel)
	headModel = Layers.Dense(128, activation="relu")(headModel)
	headModel = Layers.Dropout(0.5)(headModel)
	headModel = Layers.Dense(2, activation="softmax")(headModel)
	return headModel

def plot(N):
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(numpy.arange(0, N), H.history["loss"], label="train_loss")
	plt.plot(numpy.arange(0, N), H.history["val_loss"], label="val_loss")
	plt.plot(numpy.arange(0, N), H.history["accuracy"], label="train_acc")
	plt.plot(numpy.arange(0, N), H.history["val_accuracy"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig("plot.png")

# initialize the initial learning rate, number of epochs to train for,
# and batch size
InitialLR = .0001
EP = 17
BS = 32

DIRECTORY = r"D:\Grad School\Fall 2021\Data mining\Group Project\Face-Mask-Detection\dataset"
CATEGORIES = ["with_mask", "without_mask"]

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("loading images...")

data = []
labels = []

data,labels = StoreInfo(CATEGORIES,DIRECTORY)

# perform one-hot encoding on the labels
lb = SKPreprocess.LabelBinarizer()
labels = lb.fit_transform(labels)
labels = utils.to_categorical(labels)

data = numpy.array(data, dtype="float32")
labels = numpy.array(labels)

(X_train, X_test, Y_train, Y_test) = SKSelecton.train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# construct the training image generator for data augmentation
aug = Preprocess.ImageDataGenerator(rotation_range=20,zoom_range=0.15,width_shift_range=0.2,height_shift_range=0.2,
	shear_range=0.15,horizontal_flip=True,fill_mode="nearest")

# load the MobileNetV2 network, ensuring the head FC layer sets are
# left off
baseModel = App.MobileNetV2(weights="imagenet", include_top=False, input_tensor=Layers.Input(shape=(224, 224, 3)))


headModel = Construct_Head_model(baseModel)


# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = ML.Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

# compile our model
print("compiling model...")
opt = optimizer.Adam(learning_rate=InitialLR, decay=(InitialLR / EP))
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])

# train the head of the network
print(" training head...")
H = model.fit(aug.flow(X_train, Y_train, batch_size=BS),steps_per_epoch=len(X_train) // BS,validation_data=(X_test, Y_test),validation_steps=len(X_test) // BS,epochs=EP)

# make predictions on the testing set
print("evaluating network...")
PredictID = model.predict(X_test, batch_size=BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
PredictID = numpy.argmax(PredictID, axis=1)

# show a nicely formatted classification report
print(SKMetrics.classification_report(Y_test.argmax(axis=1), PredictID,
	target_names=lb.classes_))

# serialize the model to disk
print("saving mask detector model...")
model.save("mask_detector.model", save_format="h5")

# plot the training loss and accuracy
N = EP
plot(N)
