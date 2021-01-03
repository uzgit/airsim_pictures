from keras import backend as K
from keras import layers
from keras.layers import Add, Input, UpSampling2D, Concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.callbacks import Callback, ModelCheckpoint
import matplotlib.pyplot as plt

import tensorflow
import pandas
import cv2
import numpy
import random

from sklearn.model_selection import train_test_split

HEIGHT = 480
WIDTH = 640

# get system info for tensorflow
print("Devices: {}".format(tensorflow.config.experimental.list_logical_devices()))
gpu_devices = tensorflow.config.list_physical_devices("GPU")
print("GPU device : {}".format(gpu_devices[0]))

# set directories and filenames
image_directory = "images"
data_csv = "data.csv"

# get data and filter empty masks
data = pandas.read_csv("{}/{}".format(image_directory, data_csv), index_col=0)
data = data[data["mask"].str.len() > 0]

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return (1 - dice_coef(y_true, y_pred))


def tversky(y_true, y_pred, smooth=1e-6):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = tensorflow.reduce_sum(y_true_pos * y_pred_pos)
    false_neg = tensorflow.reduce_sum(y_true_pos * (1 - y_pred_pos))
    false_pos = tensorflow.reduce_sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def focal_tversky_loss(y_true, y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return tensorflow.keras.backend.pow((1 - pt_1), gamma)


def build_model(input_shape, loss_function='binary_crossentropy', metrics_=[]):
    inputs = Input(input_shape)

    c1 = Conv2D(8, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(8, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(16, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(c5)
    p5 = MaxPooling2D(pool_size=(2, 2))(c5)

    c55 = Conv2D(128, (3, 3), activation='relu', padding='same')(p5)
    c55 = Conv2D(128, (3, 3), activation='relu', padding='same')(c55)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2))(c55)
    u6 = concatenate([u6, c5])
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(c6)

    u71 = Conv2DTranspose(32, (2, 2), strides=(2, 2))(c6)
    u71 = concatenate([u71, c4])
    c71 = Conv2D(32, (3, 3), activation='relu', padding='same')(u71)
    c61 = Conv2D(32, (3, 3), activation='relu', padding='same')(c71)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2))(c61)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(c7)

    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2))(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same')(c8)

    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2))(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same')(c9)

    outputs = Conv2D(4, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss=loss_function, metrics=metrics_)

    return model

def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [numpy.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = numpy.zeros(shape[0]*shape[1], dtype=numpy.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

def load_image_train(index):

    filename = data.iloc[index].filename

    path = "{}/{}".format(image_directory, filename)
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(src=image, dsize=(HEIGHT, WIDTH), interpolation=cv2.INTER_AREA)


    mask = rle_decode(data.iloc[index]["mask"], (image.shape[0], image.shape[1]))

    image = numpy.reshape(image, newshape=(480, 640, 3)).astype(numpy.float32) / 255.0
    mask = numpy.reshape(mask, newshape=(480, 640, 1))

    return image, mask

model = build_model((HEIGHT, WIDTH, 3), loss_function=dice_loss, metrics_=[dice_coef])
model.summary()







images = []
masks = []
max_index = len(data) - 1
for i in range(100):
    index = random.randint(0, max_index)
    # print("index: {}".format(index))
    image, mask = load_image_train(index)

    images.append(image)
    masks.append(mask)



random_state = random.randint(0, 10000)
train_images, test_images, train_masks, test_masks = train_test_split(images, masks, test_size=0.2, random_state=random_state)

print("images: {}, {}".format(len(images), images[0].shape))
print("masks: {}, {}".format(len(masks), masks[0].shape))
print("training images: {}, {}".format(len(train_images), train_images[0].shape))
print("training masks: {}, {}".format(len(train_masks), train_masks[0].shape))
print("testing images: {}, {}".format(len(test_images), test_images[0].shape))
print("testing masks: {}, {}".format(len(test_masks), test_masks[0].shape))

EPOCHS = 20
BATCH_SIZE = 64
model_history = model.fit(x=train_images,
                          y=train_masks,
                          batch_size=BATCH_SIZE,
                          epochs=EPOCHS,
                          verbose=1,
                          validation_data=(test_images, test_masks))