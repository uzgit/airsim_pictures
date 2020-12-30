#!/usr/bin/python3

import airsim
from datetime import datetime
import numpy
import os
import uuid

import matplotlib.pyplot as plt

print("getting image...");

# for car use CarClient()
client = airsim.MultirotorClient()

# clear
success = client.simSetSegmentationObjectID("[\w]*", -1, True)
print(success)
success = client.simGetSegmentationObjectID("symbol_cube")
print(success)
# responses = client.simGetImages([
#     png format
    # airsim.ImageRequest(0, airsim.ImageType.Scene),
    # uncompressed RGB array bytes
    # airsim.ImageRequest(1, airsim.ImageType.Scene, False, False),
    # floating point uncompressed image
    # airsim.ImageRequest(1, airsim.ImageType.DepthPlanner, True)])

 # do something with response which contains image data, pose, timestamp etc

responses = client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Segmentation, False, False)])
print("response length: {}".format(len(responses)))
response = responses[0]
print(dir(response))

# print(responses)

# get numpy array
# img1d = numpy.fromstring(response.image_data_uint8, dtype=numpy.uint8)
img1d = numpy.frombuffer(response.image_data_uint8, dtype=numpy.uint8)

# reshape array to 4 channel image array H X W X 4
img_rgb = img1d.reshape(response.height, response.width, 3)
print("response dimensions: {} x {}".format(response.height, response.width))

# original image is flipped vertically
# img_rgb = numpy.flipud(img_rgb)

# filename = "airsim_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
# filename = str(uuid.uuid3(uuid.NAMESPACE_X500, "blocks,1.0,1.0,1.0")).replace("-", "") + ".png"
# filename = filename.replace(" ", "_")

filename = "segmentation.png"

# write to png
print("writing {} ...".format(filename))
airsim.write_png(os.path.normpath(filename), img_rgb)
print("dimensions: {} x {}".format(img_rgb.shape[0], img_rgb.shape[1]))