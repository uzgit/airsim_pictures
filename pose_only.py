# set global directories and filenames
image_directory = "images"
labels_csv = "data.csv"

# imports
import glob
import pandas
pandas.set_option("display.max_columns", 500)
pandas.set_option("display.width", 1000)
from sklearn.model_selection import train_test_split
import keras


# get image filenames as a glob
images = glob.glob("{}/*.png".format(image_directory))

# get labels as a pandas dataframe
labels = pandas.read_csv("{}/{}".format(image_directory, labels_csv), index_col=0)
print("Labels:")
print(labels.head(5))
print()

# split into testing and validation sets
training_images, validation_images = train_test_split(images, test_size=0.2)
print("Total images: {}".format(len(images)))
print("Total training images: {}".format(len(training_images)))
print("Total validation images: {}".format(len(validation_images)))

# create a DataGenerator class