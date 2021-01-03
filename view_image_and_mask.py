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

# print(data.head(5))

# filters
# remove rows where the mask is empty (where the marker doesn't appear)
data = data[data["mask"].str.len() > 0]

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

def plot_masked_image(index):

    filename = data.iloc[index].filename

    path = "{}/{}".format(image_directory, filename)
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = rle_decode(data.iloc[index]["mask"], (image.shape[0], image.shape[1]))
    # print("mask shape: {}".format(mask.shape))
    contours, hierarchy = cv2.findContours(mask, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    plt.imshow(image)
    plt.show()

def plot_resized_masked_image(datapoint):

    filename = datapoint["filename"]

    path = "{}/{}".format(image_directory, filename)
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = rle_decode(datapoint["mask"], (image.shape[0], image.shape[1], 1))
    # print("mask shape: {}".format(mask.shape))
    contours, hierarchy = cv2.findContours(mask, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

    if( RESIZE ):
        image = tensorflow.cast(image, tensorflow.float32) / 255.0
        image = tensorflow.image.resize(image, (224, 224))
        mask = numpy.reshape(mask, newshape=(mask.shape[0], mask.shape[1], 1))
        mask = tensorflow.cast(mask, tensorflow.uint8)
        mask = tensorflow.image.resize(mask, (224, 224))
        mask = numpy.reshape(mask, newshape=(mask.shape[0], mask.shape[1]))
    plt.imshow(image)
    plt.show()
    plt.imshow(mask)
    plt.show()

    return image, mask

# without filtering this one doesn't work because it doesn't have a mask
# plot_masked_image(4507)

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

    # decode the image's mask
    mask = rle_decode(row["mask"], (image.shape[0], image.shape[1]))

    return image, mask

max_index = len(data) - 1
for i in range(10):
    index = random.randint(0, max_index)
    print("index: {}".format(index))
    # plot_masked_image(index)
    # plot_resized_masked_image(data.iloc[index])
    image, mask = get_image_and_mask(index)
    plt.imshow(image)
    plt.show()
    plt.imshow(mask)
    plt.show()

print("{} valid rows".format(len(data)))
