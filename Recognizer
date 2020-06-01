from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, GlobalAveragePooling2D
import numpy as np
import tensorflow as tf
seed_value = 56#for reproducibilty
np.random.seed(seed_value)# Set `numpy` pseudo-random generator at a fixed value

# Load the training data from the corresponding npz files
train_images = np.load('C:\\Users\\ADMIN\\Desktop\\Deep Learning Projects\\Handwriting\\kmnist-train-imgs.npz')['arr_0']
train_labels = np.load('C:\\Users\\ADMIN\\Desktop\\Deep Learning Projects\\Handwriting\\kmnist-train-labels.npz')['arr_0']

# Load the test data from the corresponding npz files
test_images =  np.load('C:\\Users\\ADMIN\\Desktop\\Deep Learning Projects\\Handwriting\\kmnist-test-imgs.npz')['arr_0']
test_labels =  np.load('C:\\Users\\ADMIN\\Desktop\\Deep Learning Projects\\Handwriting\\kmnist-test-labels.npz')['arr_0']

random_indices = np.random.choice(2000, size=2000, replace=False)

    # Get the data corresponding to these indices
train_images = train_images[random_indices]#taking small subset of train due to CPU limitations
train_labels = train_labels[random_indices]

from keras.utils import np_utils
train_labels = np_utils.to_categorical(train_labels, 10)#one hot encoding train_labels, total labels of kmnist=10

random_indices = np.random.choice(400, size=400, replace=False)
test_images = test_images[random_indices]#taking small subset of test due to CPU limitations
test_labels = test_labels[random_indices]
test_labels = np_utils.to_categorical(test_labels, 10)#one hot encoding train_labels, total labels of kmnist=10

# converting grayscale train and test images to RGB as required by ResNet
train_RGB = np.ndarray(shape=(train_images.shape[0], train_images.shape[1], train_images.shape[2], 3), dtype= np.uint8)
train_RGB[:, :, :, 0] = train_images[:, :, :]
train_RGB[:, :, :, 1] = train_images[:, :, :]
train_RGB[:, :, :, 2] = train_images[:, :, :]

test_RGB = np.ndarray(shape=(test_images.shape[0], test_images.shape[1], test_images.shape[2], 3), dtype= np.uint8)
test_RGB[:, :, :, 0] = test_images[:, :, :]
test_RGB[:, :, :, 1] = test_images[:, :, :]
test_RGB[:, :, :, 2] = test_images[:, :, :]

from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath='model.best.hdf5', # checkpointer saves weights of model with lowest validation loss
                               verbose=1,save_best_only=True)
#Importing the ResNet50 model
from keras.applications.resnet50 import ResNet50, preprocess_input

#Loading the ResNet50 model with pre-trained ImageNet weights
model = ResNet50(weights='imagenet', include_top=False, input_shape=(200, 200, 3))

from PIL import Image

#Reshaping the training data according to ResNet minimum size requirement
X_train =  np.array([np.array(Image.fromarray(train_RGB[i]).resize((200,200))) for i in range(0, len(train_RGB))]).astype('float32')

#Preprocessing the train data, so that it can be fed to the pre-trained ResNet50 model.
resnet_train_input = preprocess_input(X_train)

#Reshaping the testing data according to ResNet minimum size requirement
X_test =np.array([np.array(Image.fromarray(test_RGB[i]).resize((200,200))) for i in range(0, len(test_RGB))]).astype('float32')

#Preprocessing the test data, so that it can be fed to the pre-trained ResNet50 model.
resnet_test_input = preprocess_input(X_test)

#Creating bottleneck features for the testing data
test_features = model.predict(resnet_test_input)

#Saving the bottleneck features
np.savez('resnet_features_test', features=test_features)

#Creating bottleneck features for the training data
train_features = model.predict(resnet_train_input)

#Saving the bottleneck features
np.savez('resnet_features_train', features=train_features)

model = Sequential()
model.add(GlobalAveragePooling2D(input_shape=train_features.shape[1:]))
model.add(Dropout(0.3,seed=seed_value))
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

model.fit(train_features, train_labels, batch_size=32, epochs=20,
          validation_split=0.2, callbacks=[checkpointer], verbose=1, shuffle=True)

#Evaluate the model on the test data
score  = model.evaluate(test_features, test_labels)

#Accuracy on test data
print('Accuracy on the Test Images: ', score[1])
