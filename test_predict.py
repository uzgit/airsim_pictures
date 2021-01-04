RESIZE = True

import matplotlib.pyplot as plt
import pandas
pandas.set_option("display.max_columns", 500)
pandas.set_option("display.width", 1000)
import cv2
import numpy
import random

import tensorflow

image_directory = "images"
data_csv = "data.csv"

data = pandas.read_csv("{}/{}".format(image_directory, data_csv), index_col=0)
# filters
# remove rows where the mask is empty (where the marker doesn't appear)
data = data[data["mask"].str.len() > 0]

model = tensorflow.keras.models.load_model("model.h5")

def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [numpy.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = numpy.zeros(shape[0]*shape[1], dtype=numpy.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

WIDTH = 224
HEIGHT = 224
def get_image_and_mask(index):
    row = data.iloc[index]

    # get the image of the current row
    filename = row["filename"]
    path = "{}/{}".format(image_directory, filename)
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(src=image, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    image = numpy.array(image).astype("float32") / 255

    print(image)

    # decode the image's mask
    mask = rle_decode(row["mask"], (image.shape[0], image.shape[1]))

    return image, mask

def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tensorflow.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

def display_predictions(num_displays=1):

    indices = random.sample(range(len(data)), num_displays)

    for i in indices:
        image, mask = get_image_and_mask(i)
        print("image shape: {}".format(image.shape))

        # prediction_input = numpy.expand_dims( numpy.array(image), 0 )
        # prediction_input = numpy.reshape(image, newshape=(1, 224, 224, 3))

        # prediction_input = numpy.reshape( image, newshape=(1, 224, 224, 3) )
        # prediction_input = tensorflow.reshape(image, shape=(1, 224, 224, 3))
        # prediction_input = numpy.random.random(size=(1, 224, 224, 3))
        # prediction_input = numpy.expand_dims(prediction_input, axis=0)
        # prediction_input = numpy.expand_dims(prediction_input, axis=0)

        prediction_input = image.reshape((32,) + image.shape)

        # prediction_input = image
        # prediction_input = numpy.reshape(image, newshape=(1, 224, 224, 3))
        # prediction_input = numpy.expand_dims( prediction_input, axis=0)

        print("stuff: {}".format(prediction_input.shape))
        # prediction = numpy.zeros(shape=(image.shape[0], image.shape[1], 1))
        prediction = model.predict(x=prediction_input, batch_size=32)
        display([image, mask, prediction])

display_predictions()