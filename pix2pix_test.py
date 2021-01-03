import tensorflow
from tensorflow_examples.models.pix2pix import pix2pix
import pandas
import cv2
import numpy
import random
from sklearn.model_selection import train_test_split

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

# import existing network
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

def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [numpy.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = numpy.zeros(shape[0]*shape[1], dtype=numpy.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

@tensorflow.function
def load_image_train(index):

    filename = data.iloc[index].filename

    path = "{}/{}".format(image_directory, filename)
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(src=image, dsize=(224, 224), interpolation=cv2.INTER_AREA)

    mask = rle_decode(data.iloc[index]["mask"], (image.shape[0], image.shape[1]))

    return image, mask

OUTPUT_CHANNELS = 2
model = unet_model(OUTPUT_CHANNELS)
model.compile(optimizer="adam", loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

# draw the model
# tensorflow.keras.utils.plot_model(model, show_shapes=True, to_file="model_diagram.png")

images = []
masks = []
max_index = len(data) - 1
for i in range(100):
    index = random.randint(0, max_index)
    # print("index: {}".format(index))
    image, mask = load_image_train(index)

    images.append(image)
    masks.append(mask)

print("images: {}, {}".format(len(images), images[0].shape))
print("masks: {}, {}".format(len(masks), masks[0].shape))

random_state = random.randint(0, 10000)
train_images, train_masks, test_images, test_masks = train_test_split(images, masks, test_size=0.2, random_state=random_state)

EPOCHS = 20
BATCH_SIZE = 64
model_history = model.fit(x=train_images,
                          y=train_masks,
                          batch_size=BATCH_SIZE,
                          epochs=EPOCHS,
                          verbose=1,
                          validation_data=(test_images, test_masks))