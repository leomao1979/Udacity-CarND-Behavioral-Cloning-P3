import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Cropping2D
from keras.layers.pooling import MaxPooling2D

def image_path(source_path):
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/' + filename
    return current_path

def load_driving_data():
    lines = []
    with open('data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for line in reader:
            lines.append(line)

    images = []
    steerings = []
    steering_adjustment = 0.2
    for line in lines:
        center_image_path = image_path(line[0])
        images.append(cv2.imread(center_image_path))
        steerings.append(float(line[3]))
        left_image_path = image_path(line[1])
        images.append(cv2.imread(left_image_path))
        steerings.append(float(line[3]) + steering_adjustment)
        right_image_path = image_path(line[2])
        images.append(cv2.imread(right_image_path))
        steerings.append(float(line[3]) - steering_adjustment)

    return (images, steerings)

def augment_training_data(images, steerings):
    augmented_images = []
    augmented_steerings = []
    for image, steering in zip(images, steerings):
        augmented_images.append(image)
        augmented_steerings.append(steering)
        augmented_images.append(np.fliplr(image))
        augmented_steerings.append(-steering)
    return (augmented_images, augmented_steerings)

def build_simple_model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160, 320, 3)))
    model.add(Flatten())
    model.add(Dense(1))
    return model

def resize_normalize(image, new_size):
    from keras.backend import tf
    resized = tf.image.resize_images(image, new_size)
    normalized = resized / 255.0 - 0.5
    return normalized

def build_lenet_model():
    model = Sequential()
    model.add(Cropping2D(cropping = ((75, 25), (0, 0)), input_shape = (160, 320, 3)))
    model.add(Lambda(resize_normalize, arguments = {'new_size': (32, 32)}))
    model.add(Conv2D(6, (5, 5), activation = 'relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(16, (5, 5), activation = 'relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    # dropout?
    # model.add(Dropout(rate=0.5))
    model.add(Dense(84))
    # model.add(Dropout(rate=0.5))
    model.add(Dense(1))
    return model

def build_nvidia_model():
    model = Sequential()
    model.add(Cropping2D(cropping = ((75, 25), (0, 0)), input_shape = (160, 320, 3)))
    model.add(Lambda(resize_normalize, arguments = {'new_size': (66, 200)}))
    model.add(Conv2D(24, kernel_size = (5, 5), strides=(2, 2), activation = 'relu'))
    model.add(Conv2D(36, kernel_size = (5, 5), strides=(2, 2), activation = 'relu'))
    model.add(Conv2D(48, kernel_size = (5, 5), strides=(2, 2), activation = 'relu'))
    model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
    model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

images, steerings = load_driving_data()
images, steerings = augment_training_data(images, steerings)
X_train = np.array(images)
y_train = np.array(steerings)
print('X_train.shape: {}, y_train.shape: {}'.format(X_train.shape, y_train.shape))

model = build_nvidia_model()
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, epochs = 8)
model.save('model.h5')
