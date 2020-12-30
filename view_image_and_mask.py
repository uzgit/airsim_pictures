import matplotlib.pyplot as plt
import pandas
pandas.set_option("display.max_columns", 500)
pandas.set_option("display.width", 1000)
import cv2
import numpy
import random

image_directory = "images"
data_csv = "data.csv"

data = pandas.read_csv("{}/{}".format(image_directory, data_csv), index_col=0)

print(data.head(5))

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
    colors = [(249, 192, 12), (0, 185, 241), (114, 0, 218), (249, 50, 12)]

    channels = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]]
    intensities = [255, 255, 255, 125]

    filename = data.iloc[index].filename

    path = "{}/{}".format(image_directory, filename)
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # print(data.iloc[index]["mask"])

    mask = rle_decode(data.iloc[index]["mask"], (480, 640))
    # print("mask shape: {}".format(mask.shape))
    contours, hierarchy = cv2.findContours(mask, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    plt.imshow(image)
    plt.show()

# plot_masked_image(4507)

max_index = len(data) - 1
for i in range(10):
    index = random.randint(0, max_index)
    print("index: {}".format(index))
    plot_masked_image(index)

print("{} valid rows".format(len(data)))