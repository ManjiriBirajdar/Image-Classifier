# image augmentation
# code reference: https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/

from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
#from matplotlib import pyplot

# load the image
img = load_img('C:/Users/Manjiri/Documents/AI_Demostrator/code/DataAugmentation/orginal_images/yellow_2.jpg')

# convert to numpy array
data = img_to_array(img)

# expand dimension to one sample
samples = expand_dims(data, 0)

##### create image data augmentation generator #####

# horizontal shift image augmentation
datagen1 = ImageDataGenerator(width_shift_range=[-200,200])

# vertical shift image augmentation
datagen2 = ImageDataGenerator(height_shift_range=0.2)

# horizontal and vertical flip image augmentation
datagen3 = ImageDataGenerator(horizontal_flip=True)

# random rotation image augmentation
datagen4 = ImageDataGenerator(rotation_range=90)

# prepare iterator
#it = datagen.flow(samples, batch_size=1)

save_here = 'C:/Users/Manjiri/Documents/AI_Demostrator/code/DataAugmentation/datagen/yellow'
img_prefix = 'yellow'

#red_save_here = 'C:/Users/Manjiri/Documents/AI_Demostrator/code/DataAugmentation/datagen/red'
#red_img_prefix = 'red'

#yellow_save_here = 'C:/Users/Manjiri/Documents/AI_Demostrator/code/DataAugmentation/datagen/yellow'
#yellow_img_prefix = 'yellow'

# code reference: https://stackoverflow.com/questions/47826730/how-to-save-resized-images-using-imagedatagenerator-and-flow-from-directory-in-k
for x, val in zip(datagen1.flow(samples,                    #image we chose
        save_to_dir=save_here,     #this is where we figure out where to save
         save_prefix=img_prefix,        # it will save the images as 'aug_0912' some number for every new augmented image
        save_format='jpg'),range(50)) :     # here we define a range because we want 10 augmented images otherwise it will keep looping forever I think
		pass

for x, val in zip(datagen2.flow(samples,                    #image we chose
        save_to_dir=save_here,     #this is where we figure out where to save
         save_prefix=img_prefix,        # it will save the images as 'aug_0912' some number for every new augmented image
        save_format='jpg'),range(50)) :     # here we define a range because we want 10 augmented images otherwise it will keep looping forever I think
		pass

for x, val in zip(datagen3.flow(samples,                    #image we chose
        save_to_dir=save_here,     #this is where we figure out where to save
         save_prefix=img_prefix,        # it will save the images as 'aug_0912' some number for every new augmented image
        save_format='jpg'),range(50)) :     # here we define a range because we want 10 augmented images otherwise it will keep looping forever I think
		pass

for x, val in zip(datagen4.flow(samples,                    #image we chose
        save_to_dir=save_here,     #this is where we figure out where to save
         save_prefix=img_prefix,        # it will save the images as 'aug_0912' some number for every new augmented image
        save_format='jpg'),range(50)) :     # here we define a range because we want 10 augmented images otherwise it will keep looping forever I think
		pass

# generate samples and plot
# for i in range(9):
# 	# define subplot
# 	pyplot.subplot(330 + 1 + i)
# 	# generate batch of images
# 	batch = it.next()
# 	# convert to unsigned integers for viewing
# 	image = batch[0].astype('uint8')    
# 	# plot raw pixel data
# 	pyplot.imshow(image)
# # show the figure
# pyplot.show()