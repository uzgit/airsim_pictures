import tensorflow
# gpu_devices = tensorflow.config.list_physical_devices("GPU")
gpu_devices = tensorflow.config.experimental.list_physical_devices('GPU')
print(gpu_devices)