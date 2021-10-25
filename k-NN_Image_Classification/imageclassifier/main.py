import tensorflow as tf
from tensorflow import keras
import pickle 

from src.preprocessing import SimplePreprocessor
from src.datasets import SimpleDatasetLoader
from imutils import paths
import argparse

# load the model from disk
model = pickle.load(open("knn-model.pkl", 'rb'))

# grab the list of images that we'll be describing
print("[INFO] processing the image...")

# initialize the image preprocessor, load the dataset from disk,
# and reshape the data matrix
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data) = sdl.loadSingleImage("../inputimg/ddd.jpg")
data1 = data.reshape((data.shape[0], 3072))

predictions = model.predict(data1)

score = predictions[0]

print("score = ", score)

if(score == 0):
    print("This is a blue block")
elif(score == 1):
    print("This is a red block")
elif(score == 2):
    print("This is a yellow block")


