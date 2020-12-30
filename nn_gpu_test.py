import keras
import keras.datasets.mnist as mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, TensorBoard
import keras.backend as backend
import tensorflow

print("Devices: {}".format(tensorflow.config.experimental.list_logical_devices()))
gpu_devices = tensorflow.config.list_physical_devices("GPU")

print("GPU device : {}".format(gpu_devices[0]))

batch_size = 128
num_classes = 10
epochs = 12

image_rows = 28
image_cols = 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()

if backend.image_data_format() == "channels_first":
    x_train = x_train.reshape(x_train.shape[0], 1, image_rows, image_cols)
    x_test  =  x_test.reshape( x_test.shape[0], 1, image_rows, image_cols)
    input_shape = (1, image_rows, image_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], image_rows, image_cols, 1)
    x_test  =  x_test.reshape( x_test.shape[0], image_rows, image_cols, 1)
    input_shape = (image_rows, image_cols, 1)

x_train = x_train.astype("float32")
x_test  = x_test.astype("float32")

x_train /= 255
x_test  /= 255

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test  =  keras.utils.to_categorical(y_test, num_classes)

print("checkpoint 1")

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation="relu", input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation="relu", input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation="softmax"))

print("checkpoint 2")

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=["accuracy"])

print("checkpoint 3")

best_check = ModelCheckpoint(filepath="model-best.h5", verbose=1, save_weights_only=True, save_best_only=True)

print("checkpoint 4")

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test), callbacks=[best_check])

print("checkpoint 5")

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])