# USAGE
# python knn.py --dataset dataset/blocks

# import the necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from imutils import paths
import argparse
import pickle 

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1,
	help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
	help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())

# grab the list of images that we'll be describing
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the image preprocessor, load the dataset from disk,
# and reshape the data matrix
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))

# show some information on memory consumption of the images
print("[INFO] features matrix: {:.1f}MB".format(
	data.nbytes / (1024 * 1024.0)))

# encode the labels as integers
print("labels before = ", labels)
le = LabelEncoder()
labels = le.fit_transform(labels)
print("labels after transform = ", labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
# train_test_split: Split arrays or matrices into random train and test subsets

(trainX, testX, trainY, testY) = train_test_split(data, labels,	test_size=0.25, random_state=42)

# train a k-NN classifier on the raw pixel intensities
print("[INFO] evaluating k-NN classifier...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"],
	n_jobs=args["jobs"])
model.fit(trainX, trainY)

# evaluate a k-NN classifier on the raw pixel intensities
print(classification_report(testY, model.predict(testX),
	target_names=le.classes_))

# Save the trained model as a pickle string.
# saved_model = pickle.dumps(model) 

# Store data (serialize)
pickle.dump(model, open("knn-model.pkl", "wb"))  # save it into a file named save.p

# """
# ## Run inference on new data
# Note that data augmentation and dropout are inactive at inference time.
# """
# image_size = (180, 180)

# img = keras.preprocessing.image.load_img(
#     "dataset/blocks/blue/001.jpg", target_size=image_size
# )
# img_array = keras.preprocessing.image.img_to_array(img)
# img_array = tf.expand_dims(img_array, 0)  # Create batch axis

# predictions = model.predict(img_array)
# score = predictions[0]
# print(
#     "This image is %.2f percent cat and %.2f percent dog."
#     % (100 * (1 - score), 100 * score)
# )

################

