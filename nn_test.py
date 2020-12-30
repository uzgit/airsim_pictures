image_directory = "images"
image_height = 480
image_width  = 640

import keras
import glob
import numpy
import matplotlib.pyplot as plt
import matplotlib.image as mplimage
import pandas
pandas.set_option("display.max_columns", 500)
pandas.set_option("display.width", 1000)

def get_single_image(filename):
    return mplimage.imread("{}/{}".format(image_directory, filename))

train_test_split = 0.8
data = pandas.read_csv("images/data.csv", index_col=0)

num_rows       = len(data)
training_end   = int(num_rows * train_test_split)
testing_points = int(num_rows - training_end)

print(data.head(5))
print("Number of data points:\t\t{}".format(num_rows))
print("\tTraining data points:\t{}".format(training_end))
print("\tTesting data points:\t{}".format(testing_points))

training_data = data.iloc[0 : training_end]
testing_data  = data.iloc[training_end : ]

test = get_single_image(data.iloc[0].filename)
test_image = plt.imshow(test)

plt.show()