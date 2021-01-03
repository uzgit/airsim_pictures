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

# import theano
# theano.config.exception_verbosity='high'

from tensorflow_examples.models.pix2pix import pix2pix
from sklearn.model_selection import train_test_split
pandas.set_option("display.max_columns", 500)
pandas.set_option("display.width", 1000)
HEIGHT = 224
WIDTH = 224

# get system info for tensorflow
print("Devices: {}".format(tensorflow.config.experimental.list_logical_devices()))
gpu_devices = tensorflow.config.list_physical_devices("GPU")
print("GPU device : {}".format(gpu_devices[0]))

# set directories and filenames
image_directory = "images"
data_csv = "data.csv"

# get data and filter empty masks
data = pandas.read_csv("{}/{}".format(image_directory, data_csv), index_col=0)
data = data.iloc[:1000]
data = data[data["mask"].str.len() > 0]

def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [numpy.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = numpy.zeros(shape[0]*shape[1], dtype=numpy.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

OUTPUT_CHANNELS = 2
base_model = tensorflow.keras.applications.MobileNetV2(input_shape=[224, 224, 3], include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
layers = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tensorflow.keras.Model(inputs=base_model.input, outputs=layers)

down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

def unet_model(output_channels):
  inputs = tensorflow.keras.layers.Input(shape=[224, 224, 3])
  x = inputs

  # Downsampling through the model
  skips = down_stack(x)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tensorflow.keras.layers.Concatenate()
    x = concat([x, skip])

  # This is the last layer of the model
  last = tensorflow.keras.layers.Conv2DTranspose(
      output_channels, 3, strides=2,
      padding='same')  #64x64 -> 128x128

  x = last(x)

  return tensorflow.keras.Model(inputs=inputs, outputs=x)

model = unet_model(OUTPUT_CHANNELS)
model.compile(optimizer='adam',
              loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

# shuffle data
validation_size = 0.2
training_size = 1 - validation_size
random_state = random.randint(0, 10000)
training_data = data.sample(frac=training_size, random_state=random_state)
validation_data = data.drop(training_data.index)

num_training_images = len(training_data)
num_validation_images = len(validation_data)

print("{} training data points (training_size = {})".format(num_training_images, training_size))
print("{} validation data points (validation_size = {})".format(num_validation_images, validation_size))

def get_image_and_mask(index, dataframe=data):
    row = dataframe.iloc[index]

    # get the image of the current row
    filename = row["filename"]
    path = "{}/{}".format(image_directory, filename)
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(src=image, dsize=(640, 480), interpolation=cv2.INTER_AREA)

    # decode the image's mask
    mask = rle_decode(row["mask"], (image.shape[0], image.shape[1], 1))

    # resize
    image = tensorflow.cast(image, tensorflow.float32) / 255.0
    image = tensorflow.image.resize(image, (224, 224))
    mask = numpy.reshape(mask, newshape=(mask.shape[0], mask.shape[1], 1))
    mask = tensorflow.cast(mask, tensorflow.uint8)
    mask = tensorflow.image.resize(mask, (224, 224))
    # mask = numpy.reshape(mask, newshape=(mask.shape[0], mask.shape[1]))

    return image, mask



def data_gen(dataframe, batch_size=128):

    while( True ):

        total_images_processed = 0
        images = numpy.zeros((batch_size, HEIGHT, WIDTH, 3)).astype('float')
        masks = numpy.zeros((batch_size, HEIGHT, WIDTH, 1)).astype('float')

        # sequentially iterate through the already randomized dataframe
        for index in range(batch_size):
            index += total_images_processed
            # row = dataframe.iloc[total_images_processed + index]

            # # get the image of the current row
            # filename = row["filename"]
            # path = "{}/{}".format(image_directory, filename)
            # image = cv2.imread(path)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image = cv2.resize(src=image, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
            #
            # # decode the image's mask
            # mask = rle_decode(row["mask"], (image.shape[0], image.shape[1], 1))

            image, mask = get_image_and_mask(index=index, dataframe=dataframe)

            images[index] = image
            masks[index] = mask

        yield images, masks

        total_images_processed += batch_size

        # may experience problems here
        if( total_images_processed >= len(dataframe) ):
            total_images_processed -= total_images_processed
            dataframe = dataframe.sample(frac=1)

BATCH_SIZE = 64
training_data_generator = data_gen(training_data, batch_size=BATCH_SIZE)
validation_data_generator = data_gen(validation_data, batch_size=BATCH_SIZE)

def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tensorflow.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

def display_predictions():
    num_displays = 1
    indices = random.sample(range(len(data)), num_displays)

    for i in indices:
        image, mask = get_image_and_mask(i)
        print("image shape: {}".format(image.shape))

        # prediction_input = numpy.expand_dims( numpy.array(image), 0 )
        prediction_input = numpy.reshape(image, (224, 224*32))
        print("stuff: {}".format(prediction_input.shape))
        prediction = numpy.zeros(shape=(image.shape[0], image.shape[1], 1))
        prediction = model.predict(prediction_input, batch_size=1)
        display([image, mask, prediction])

class DisplayCallback(tensorflow.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        display_predictions()
        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

checkpoint_callback = tensorflow.keras.callbacks.ModelCheckpoint(
    filepath="model.h5",
    verbose=1,
    # save_weights_only=True,
    save_freq="epoch")

EPOCHS = 1
model_history = model.fit_generator(training_data_generator,
                                    steps_per_epoch=(num_training_images // BATCH_SIZE),
                                    epochs=EPOCHS,
                                    validation_data=validation_data_generator,
                                    validation_steps=(num_validation_images // BATCH_SIZE),
                                    verbose=1,
                                    callbacks=[checkpoint_callback])
                                    # callbacks=[DisplayCallback()])