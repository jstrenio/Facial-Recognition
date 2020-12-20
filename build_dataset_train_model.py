# John Strenio
# build dataset and train CNN model on it

# import libraries
import os
import math
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import cv2
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import save_img
from keras.preprocessing.image import img_to_array

# ignore warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def proc_image(filename, dest_dir):

    # read the greyscaled image into a variable
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    # load face and eye detectors
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes_cascade_glasses = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

    # get array of faces
    faces_detected = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

    # box faces
    print(str(len(faces_detected)) + " faces detected")
    for face in faces_detected:
        (x, y, w, h) = face
        #cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 1)

        # detect eyes within face
        #eyes = eyes_cascade.detectMultiScale(img[y:y+h, x:x+w])
        eyes = eyes_cascade_glasses.detectMultiScale(img[y:y+h, x:x+w])

        # box eyes
        eye_centers = list()
        for (ex, ey, ew, eh) in eyes:
            # draw rectangle around eye
            #cv2.rectangle(img, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255, 255, 255), 1)

            # find the center of the eye
            new_center = (int(x+ex+ew/2), int(y+ey+eh/2))

            # make sure left eye is first eye and right eye is second
            if eye_centers and new_center[0] < eye_centers[0][0]:
                eye_centers.insert(0, new_center)
            else:
                eye_centers.append(new_center)

            # draw circle for the eye
            #cv2.circle(img, new_center, 5, (155,255,255), 1)

        if len(eye_centers) == 2:
        
            # calculate angle between 2 eyes
            # eye1.y - eye2.y, eye1.x - eye2.x                        
            angle = math.atan2(eye_centers[1][1] - eye_centers[0][1], eye_centers[1][0] - eye_centers[0][0])
            angle *= (180 / math.pi)

            # straighten face
            rows, cols = img.shape[:2]
            M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
            img_rotated = cv2.warpAffine(img, M, (cols, rows))
        
        else:
            img_rotated = img  

        # crop, pad with p
        p = 1
        img_rotated = img_rotated[y-p+1:y+h+p, x-p+1:x+w+p]
        if img.shape[1] < 128:
            img_resized = cv2.resize(img_rotated, (128, 128), interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = cv2.resize(img_rotated, (128, 128), interpolation=cv2.INTER_AREA)

        norm_img = np.zeros((128, 128))
        norm_img = cv2.normalize(img_resized, norm_img, 0, 255, cv2.NORM_MINMAX)

        path, dirs, files = next(os.walk(dest_dir))
        n = len(files)

        cv2.imwrite(dest_dir + 'proc_face' + str(n) + '.jpg', norm_img)
    
    # cv2.imshow('', img)
    # cv2.waitKey()

def aug_image(filename, n_augs, f, dest_dir):
    # load the image
    img = load_img(filename)
    # convert to numpy array
    data = img_to_array(img)
    # expand dimension to one sample
    samples = expand_dims(data, 0)
    # create image data augmentation generator
    datagen = ImageDataGenerator(
                                 width_shift_range=[-5,5], 
                                 height_shift_range=[-5,5], 
                                 horizontal_flip=True, 
                                 rotation_range=15,
                                 brightness_range=[0.8,1.2]
                                 )
    # prepare iterator
    it = datagen.flow(samples, batch_size=1)
    # generate samples and plot
    for i in range(n_augs):
        # define subplot
        #pyplot.subplot(330 + 1 + i)
        # generate batch of images
        batch = it.next()
        # convert to unsigned integers for viewing
        image = batch[0].astype('uint8')
        # plot raw pixel data
        #pyplot.imshow(image)
        save_img(dest_dir + f + 'aug_' + str(i) + '.jpg', image)
    # show the figure
    #pyplot.show()

def build_dataset():

    print("building dataset...")

    # just set the source directory and destination directory and process or augment as needed
    source_dir_target = 'raw_images/' #'other_data_sets/faces_processed_not_augmented/lj/'
    aug_dir_target = 'unaugmented_images/'
    dest_dir_target = 'processed_images/target/' 

    print("processing images...")
    for f in os.listdir(source_dir_target):
        proc_image(source_dir_target + f, aug_dir_target)
    print("done.")

    print("augmenting images...")
    for f in os.listdir(aug_dir_target):
        aug_image(aug_dir_target + f, 4, f, dest_dir_target)

    for f in os.listdir(aug_dir_target):
        img = load_img(aug_dir_target + f)
        save_img('valid_face/target/' + f + '.jpg', img)
    print("done.")

    print("finished buidling dataset.")

def train_model():
    # ================== TRAIN MODEL ==============================

    print("training model on dataset...")

    # import the data and rescale from 255 grayscale values to decimal 0 - 1.0
    train_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)

    # preprocess images in batches
    train_generator = train_datagen.flow_from_directory(
        # source dir
        'processed_images',
        classes = ['stranger', 'target'],
        target_size=(128,128),
        batch_size=1, #104, # batch_size * steps_per_epoch in model.fit below should = num_images
        # binary labels
        class_mode='binary',
        color_mode='grayscale',
        #shuffle=True
        )

    validation_generator = validation_datagen.flow_from_directory(
        # source dir
        'valid_face',
        classes = ['stranger', 'target'],
        target_size=(128,128),
        batch_size=1, #10, # batch_size * steps_per_epoch in model.fit below should = num_images
        # binary labels
        class_mode='binary',
        color_mode='grayscale',
        shuffle=False
        )

    # build model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(128, 128, 1)))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(2,2))
    model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(2,2))
    model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(2,2))
    model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(2,2))
    # Flatten the results to feed into a DNN
    model.add(tf.keras.layers.Flatten())
    # 512 neuron hidden layer # dense ouput = (batch_size, units)
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('other') and 1 for the other ('lj')
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0001),
                loss = 'binary_crossentropy',
                metrics=['accuracy'])

    # train model
    history = model.fit(train_generator,
                        # DON'T SET BATCH SIZE IF USING DATA GENERATORS SINCE THEY SET BATCH SIZE
                        #  steps_per_epoch * batch_size in generator above should = num_images
                        steps_per_epoch=1000,
                        epochs=30,
                        verbose=1,
                        validation_data=validation_generator,
                        validation_steps=200
                        ) 

    model.save("saved_model1")

    print("model trained and saved.")


build_dataset()

train_model()













