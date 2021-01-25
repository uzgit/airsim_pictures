#!/usr/bin/python3

# from https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/segmentation.ipynb#scrollTo=MQmKthrSBCld

import tensorflow
from tensorflow_examples.models.pix2pix import pix2pix
import tensorflow_datasets

from IPython.display import clear_output
import matplotlib.pyplot as plt

dataset, info = tensorflow_datasets.load("oxford_iiit_pet:3.*.*", with_info=True)

def normalize(input_image, input_mask):
    input_image = tensorflow.cast(input_image, tensorflow.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask

@tensorflow.function
def load_image_train(datapoint):
    input_image = tensorflow.image.resize(datapoint["image"], (128, 128))
    input_mask  = tensorflow.image.resize(datapoint["segmentation_mask"], (128, 128))

    if( tensorflow.random.uniform(()) > 0.5 ):
        input_image = tensorflow.image.flip_left_right(input_image)
        input_mask  = tensorflow.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

def load_image_test(datapoint):
    input_image = tensorflow.image.resize(datapoint["image"], (128, 128))
    input_mask = tensorflow.image.resize(datapoint["segmentation_mask"], (128, 128))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


TRAIN_LENGTH = info.splits["train"].num_examples
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

train = dataset["train"].map(load_image_train, num_parallel_calls=tensorflow.data.experimental.AUTOTUNE)
test = dataset["test"].map(load_image_test)

train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tensorflow.data.experimental.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)

def display(display_list):
    plt.figure(figsize=(15,15))

    title = ["Input Image", "True Mask", "Predicted Mask"]

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tensorflow.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis("off")

    plt.show()

for image, mask in train.take(1):
    sample_image, sample_mask = image, mask
display([sample_image, sample_mask])

OUTPUT_CHANNELS = 3

base_model = tensorflow.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
layers = [base_model.get_layer(name).output for name in layer_names]

down_stack = tensorflow.keras.Model(inputs=base_model.input, outputs=layers)
down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

def unet_model(output_channels):
    inputs = tensorflow.keras.layers.Input(shape=[128, 128, 3])
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
model.compile(optimizer="adam", loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

tensorflow.keras.utils.plot_model(model, show_shapes=True)

def create_mask(mask):
    mask = tensorflow.argmax(mask, axis=-1)
    mask = mask[..., tensorflow.newaxis]
    return mask

def show_predictions(dataset=None, num_predictions=1):
    if( dataset ):
        for image, mask in dataset.take(num_predictions):
            predicted_mask = model.predict(image)
            predicted_mask = create_mask(predicted_mask)

            display([image[0], mask[0], predicted_mask[0]])
    else:
        predicted_mask = model.predict(sample_image[tensorflow.newaxis, ...])
        predicted_mask = create_mask(predicted_mask)

        display([sample_image, sample_mask, predicted_mask[0]])

show_predictions()

class DisplayCallback(tensorflow.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions(dataset=test_dataset, num_predictions=3)
        print("\nSample prediction after epoch {}\n".format(epoch + 1))

EPOCHS = 20
VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits["test"].num_examples // BATCH_SIZE // VAL_SUBSPLITS

model.history = model.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_dataset,
                          callbacks=[DisplayCallback()])