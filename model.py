import csv
import cv2
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Dropout, SpatialDropout2D
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('data_folder', './data', 'Data folder for the training data.')
flags.DEFINE_string('batch_size', '128', 'Batch size for model training')
flags.DEFINE_string('epochs', '10', 'Number of epochs to run the model')

def generator(samples, data_folder, batch_size=64):
    num_samples = len(samples)

    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            steering_angles, car_images = [], []
            for car_image, steering_angle in batch_samples:
                image = cv2.cvtColor(car_image, cv2.COLOR_BGR2RGB)
                car_images.append(image)
                steering_angles.append(steering_angle)
                # Flip image
                car_images.append(cv2.flip(image,1))
                steering_angles.append(steering_angle*-1.0)

            # trim image to only see section with road
            X_train = np.array(car_images)
            y_train = np.array(steering_angles)

            yield shuffle(X_train, y_train)

def model_nvidia(input_shape):
    # model described in http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    
    model.add(Conv2D(24, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(optimizer=Adam(lr=0.001), loss='mse')

    return model

def lenet(input_shape):
    model = Sequential()
    # Normalize image
    model.add(Lambda(lambda x : x/ 255.0 - 0.5, input_shape=input_shape))
    # Cropping layer
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    model.add(Conv2D(6,5,5,activation="relu"))
    model.add(MaxPooling2D())
    model.add(Conv2D(6,5,5,activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))    
    model.compile(loss='mse', optimizer='adam')

    return model

def getImages(samples, data_folder, correction):
    image_path = data_folder + '/IMG/'
    car_images, steering_angles = [], []
    for sample in samples:
        steering_center = float(sample[3])
        # create adjusted steering measurements for the side camera images
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        # read in images from center, left and right cameras
        img_center = cv2.imread(image_path + sample[0].split('/')[-1])
        img_left = cv2.imread(image_path + sample[1].split('/')[-1])
        img_right = cv2.imread(image_path + sample[2].split('/')[-1])

        # add images and angles to data set
        car_images.extend((img_center, img_left, img_right))
        steering_angles.extend((steering_center, steering_left, steering_right)) 
      
    return (car_images, steering_angles)

def main(_):
    data_folder = FLAGS.data_folder
    batch_size = int(FLAGS.batch_size)
    epochs = int(FLAGS.epochs)

    print("Running model with data folder {}, batch size {} and epochs {}.".format(data_folder, batch_size, epochs))
    samples = []
    with open(data_folder + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    car_images, steering_angles = getImages(samples, data_folder, correction=0.2)
    samples = list(zip(car_images, steering_angles))

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    # compile and train the model using the generator function
    train_generator = generator(train_samples, data_folder, batch_size=batch_size)
    validation_generator = generator(validation_samples, data_folder, batch_size=batch_size)     

    model = model_nvidia((160, 320, 3))
    steps_per_epoch=len(train_samples)//batch_size
    validation_steps=len(validation_samples)//batch_size

    print("Fit generator with {} steps per epochs and {} validation steps".format(steps_per_epoch, validation_steps))
    history_object = model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, validation_data=validation_generator, validation_steps=validation_steps, epochs=epochs, verbose=1)

    print(history_object.history.keys())
    print('Loss')
    print(history_object.history['loss'])
    print('Validation Loss')
    print(history_object.history['val_loss'])

    # Save the model
    model.save('model.h5')

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
