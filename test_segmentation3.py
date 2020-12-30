#!/usr/bin/python3

import sys
sys.path.insert(0, "/home/joshua/git/Airsim/PythonClient") # use the included AirSim Python client
import airsim
import numpy

client = airsim.MultirotorClient()

# try to exclude all objects from the segmentation image as per the documentation
result = client.simSetSegmentationObjectID("[\w]*", -1, True)
print("set all object segmentation ids to -1: {}".format(result))

# set the ID of the symbol cube to 99 instead of -1 to include it in the segmentation image
result = client.simSetSegmentationObjectID("symbol_cube", 99, True)
# result = client.simSetSegmentationObjectID("OrangeBall", 99, True)
symbol_cube_id = client.simGetSegmentationObjectID("symbol_cube")
print("set symbol_cube id: {} -> {}".format(result, symbol_cube_id))

# request Scene and Segmentation images
requests = [airsim.ImageRequest(camera_name=0, image_type=airsim.ImageType.Scene, pixels_as_float=False, compress=False),
            airsim.ImageRequest(camera_name=0, image_type=airsim.ImageType.Segmentation, pixels_as_float=False, compress=False)]
responses = client.simGetImages(requests)
print("response length: {}".format(len(responses)))

# for each returned image
for i in range(len(requests)):

    # make the image as a 1-dimensional numpy array
    img1d = numpy.frombuffer(responses[i].image_data_uint8, dtype=numpy.uint8)
    img_rgb = img1d.reshape(responses[i].height, responses[i].width, 3)

    # chatter about the image
    print("response dimensions: {} x {}".format(responses[i].height, responses[i].width))
    print("dimensions: {} x {}".format(img_rgb.shape[0], img_rgb.shape[1]))

    # save the image
    filename = "image_{}.png".format(i)
    airsim.write_png(filename, img_rgb)

    if( i == 1 ):
        print(img1d)