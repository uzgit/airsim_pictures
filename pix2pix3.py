#!/usr/bin/python3

# imports
import tensorflow
import tensorflow_examples
import pandas
import numpy
from tensorflow_examples.models.pix2pix import pix2pix
import cv2
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

# configuration
pandas.set_option("display.max_columns", 500)
pandas.set_option("display.width", 1000)
params = {"axes.titlesize" : "small"}
pylab.rcParams.update(params)
plt.subplots_adjust(top=2, bottom=1)

# get GPU information and assert that we have a GPU
print("Devices: {}".format(tensorflow.config.experimental.list_logical_devices()))
gpu_devices = tensorflow.config.list_physical_devices("GPU")
assert len(gpu_devices) > 0
print("GPU device : {}".format(gpu_devices[0]))

# global variables
NATIVE_HEIGHT = 480
NATIVE_WIDTH  = 640
NATIVE_DEPTH  = 3
NATIVE_DIMENSION = (NATIVE_HEIGHT, NATIVE_WIDTH, NATIVE_DEPTH)
NATIVE_DIMENSION_2D = (NATIVE_HEIGHT, NATIVE_WIDTH)
HEIGHT = 224
WIDTH  = 224
DEPTH  = 3
IMAGE_RESIZE_DIMENSION = (HEIGHT, WIDTH, DEPTH)
MASK_RESIZE_DIMENSION = (HEIGHT, WIDTH)
RESIZE_DIMENSION = (HEIGHT, WIDTH)
EPOCHS = 10
DATA_GENERATION_BATCH_SIZE = 32
OUTPUT_CHANNELS = 2
NUM_CLASSES = 2
IMAGE_DIRECTORY = "images"
DATA_CSV_FILE = "data.csv"
VALIDATION_SIZE = 0.2
TRAINING_SIZE   = 1 - VALIDATION_SIZE

# get data and filter empty masks
data = pandas.read_csv("{}/{}".format(IMAGE_DIRECTORY, DATA_CSV_FILE), index_col=0)
data = data[data["mask"].str.len() > 0]
data = data.iloc[:1000]

# split data
random_state = numpy.random.randint(0, 10000)
training_data = data.sample(frac=TRAINING_SIZE, random_state=random_state)
validation_data = data.drop(training_data.index)

# notify for data size and train-test-split
num_images = len(data)
num_training_images = len(training_data)
num_validation_images = len(validation_data)
print("{} total data points".format(num_images))
print("{} training data points (training_size = {})".format(num_training_images, TRAINING_SIZE))
print("{} validation data points (validation_size = {})".format(num_validation_images, VALIDATION_SIZE))

# build unet model
# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]
base_model = tensorflow.keras.applications.MobileNetV2(input_shape=[224, 224, 3], include_top=False)
layers = [base_model.get_layer(name).output for name in layer_names]
down_stack = tensorflow.keras.Model(inputs=base_model.input, outputs=layers)
down_stack.trainable = False

def unet_model(output_channels):
    # inputs = tensorflow.keras.layers.Input(shape=[224, 224, 3])
    inputs = tensorflow.keras.layers.Input(shape=(224, 224, 3))#, batch_size=DATA_GENERATION_BATCH_SIZE)
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

# define DataGenerator
class ImageMaskDataGenerator(tensorflow.keras.utils.Sequence):

    def __init__(self,
                 dataframe,
                 name,
                 mode="fit",
                 base_path=IMAGE_DIRECTORY,
                 batch_size=DATA_GENERATION_BATCH_SIZE,
                 dimension=NATIVE_DIMENSION,
                 resize_dimension=None,
                 num_input_channels=3,
                 num_output_channels=1,
                 num_classes=NUM_CLASSES,
                 random_state=0,
                 shuffle=True):
        self.dataframe = dataframe
        self.name = name
        self.mode = mode
        self.base_path = base_path
        self.batch_size = batch_size
        self.dimension = dimension
        self.resize_dimension = resize_dimension
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.num_classes = num_classes
        self.random_state = random_state
        self.shuffle = shuffle

        self.on_epoch_end()

    # get the number of batches per epoch
    def __len__(self):
        return len(self.dataframe) // self.batch_size

    def on_epoch_end(self):
        # shuffle the data
        if( self.shuffle ):
            self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)

    # get a batch of data
    def __getitem__(self, item):

        # index the i^{th} batch where i=item
        start_index = self.batch_size * item
        end_index = self.batch_size * (item + 1)

        # print("indexing [{} : {}] of data with length {}".format(start_index, end_index, len(self.dataframe)))
        current_batch_data = self.dataframe.iloc[ start_index : end_index ]

        # reset indices for quicker iteration
        current_batch_data = current_batch_data.reset_index(drop=True)

        # retrieve images
        images = self.__get_images(current_batch_data)

        # generate masks
        masks = self.__get_masks(current_batch_data)

        return images, masks

    def __get_images(self, batch_data):

        images = numpy.empty((self.batch_size, *self.resize_dimension, self.num_input_channels))
        for index, row in batch_data.iterrows():

            path = "{}/{}".format(self.base_path, row["filename"])
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image = cv2.resize(src=image, dsize=RESIZE_DIMENSION, interpolation=cv2.INTER_AREA)
            image = tensorflow.image.resize(image, self.resize_dimension)
            image = numpy.array(image).astype("float32") / 255

            images[index,] = image

        return images

    def rle_decode(self, mask_rle, shape):
        '''
        mask_rle: run-length as string formated (start length)
        shape: (height,width) of array to return
        Returns numpy array, 1 - mask, 0 - background

        '''
        s = mask_rle.split()
        starts, lengths = [numpy.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = numpy.zeros(shape[0] * shape[1], dtype=numpy.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(shape)

    def __get_masks(self, batch_data):

        # masks = numpy.empty((self.batch_size, *self.resize_dimension, self.num_output_channels))
        masks = numpy.empty((self.batch_size, *self.resize_dimension))
        for index, row in batch_data.iterrows():

            mask = self.rle_decode(row["mask"], self.dimension)
            mask = numpy.reshape(mask, newshape=(*mask.shape, 1))
            mask = tensorflow.cast(mask, tensorflow.uint8)
            mask = tensorflow.image.resize(mask, self.resize_dimension)
            mask = numpy.reshape(mask, newshape=(mask.shape[0], mask.shape[1]))

            # threshold
            mask = numpy.where(mask > 0, 1, 0)

            masks[index,] = mask

        masks = numpy.reshape(masks, newshape=(self.batch_size, *self.resize_dimension, 1))
        return masks

    # string representation
    def __str__(self):
        return "ImageMaskDataGenerator '{}' with {} batches.".format(self.name, self.__len__())

# instantiate model
model = unet_model(OUTPUT_CHANNELS)
model.compile(optimizer=tensorflow.keras.optimizers.Adam(), loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
# model.summary()

# instantiate data generators
training_data_generator = ImageMaskDataGenerator(dataframe=training_data, name="training", dimension=NATIVE_DIMENSION_2D, resize_dimension=RESIZE_DIMENSION)
validation_data_generator = ImageMaskDataGenerator(dataframe=validation_data, name="validation", dimension=NATIVE_DIMENSION_2D, resize_dimension=RESIZE_DIMENSION)
print(training_data_generator)
print(validation_data_generator)

# visualize
def plot_masked_image(image, mask):

    mask = mask.astype(numpy.uint8)
    contours, hierarchy = cv2.findContours(mask, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(image, contours, -1, (255, 0, 0), 1)
    plt.imshow(image)
    plt.show()

# training_batch_images, training_batch_masks = training_data_generator.__getitem__(20)
# validation_batch_images, validation_batch_masks = validation_data_generator.__getitem__(0)

# plot_masked_image(training_batch_images[0], training_batch_masks[0])

# instantiate checkpoint callback
checkpoint_callback = tensorflow.keras.callbacks.ModelCheckpoint(
    filepath="model.h5",
    verbose=1,
    # save_weights_only=True,
    save_freq="epoch")

# define/instantiate display callback and corresponding display function (called on epoch end)
def display_predictions(model=model, data_generator=validation_data_generator, num_predictions=1):
    assert num_predictions < DATA_GENERATION_BATCH_SIZE
    images, masks = data_generator.__getitem__(0)

    columns = num_predictions
    rows = 3

    figure, axes = plt.subplots(rows, columns)
    figure.set_figheight(3)
    figure.set_figwidth(num_predictions)
    # figure.tight_layout()

    for axis in axes:
        # axis.axis("tight")
        # axis.autoscale_view("tight")
        axis.axis("off")

    for i in range(num_predictions):
        prediction = model.predict(images[0].reshape(1, 224, 224, 3), batch_size=1)
        prediction = numpy.where(numpy.isnan(prediction), 0, prediction)
        prediction = numpy.where(prediction == 0, 0, 1)

        axes[0].imshow(images[i])
        axes[0].set_title("Input Image", pad=1)

        axes[1].imshow(prediction[0, :, :, 0])
        axes[1].set_title("Predicted Mask", pad=1)

        axes[2].imshow(masks[i])
        axes[2].set_title("True Mask", pad=1)

    plt.show()
class DisplayCallback(tensorflow.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        display_predictions()
        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))
display_callback = DisplayCallback()

# train
history = model.fit(training_data_generator,
                    steps_per_epoch=(num_training_images // DATA_GENERATION_BATCH_SIZE),
                    epochs=EPOCHS,
                    validation_data=validation_data_generator,
                    validation_steps=(num_validation_images // DATA_GENERATION_BATCH_SIZE),
                    verbose=1,
                    callbacks=[checkpoint_callback, display_callback])
