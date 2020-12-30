#!/usr/bin/python3

# indices
SCENE = 0
SEGMENTATION = 1
SYMBOL_CUBE_SEGMENTATION_ID = 99 # arbitrary value != -1 and in [0, 255]

from datetime import datetime
import numpy
import os
import uuid
import signal
import sys
sys.path.insert(0, "/home/joshua/git/Airsim/PythonClient")
import airsim
import pandas
import matplotlib.pyplot as plt

# rle functions taken from here: https://www.kaggle.com/paulorzp/run-length-encode-and-decode

def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask,
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = numpy.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs

def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)

def get_rle(mask_image):
    return rle_to_string(rle_encode(mask_image))

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

image_directory = "images"
scenario_name = "blocks"
output_csv_name = "data.csv"

columns = "x,y,z,yaw,pitch,roll,filename,mask".split(",")
data = pandas.DataFrame(columns=columns)

total_images = 50000
if( len(sys.argv) > 1 ):
    total_images = int(sys.argv[1])

print("Generating {} images with randomized pose.".format(total_images))

def sigint_handler(sig, frame):
    full_output_filename = "{}/{}".format(image_directory, output_csv_name)
    data.to_csv(full_output_filename)
    print("Saving output to {}".format(full_output_filename))
    sys.exit(0)
signal.signal(signal.SIGINT, sigint_handler)

client = airsim.MultirotorClient()

# exclude all objects from the segmentation image as per the documentation
result = client.simSetSegmentationObjectID("[\w]*", -1, True)
print("set all object segmentation ids to -1: {}".format(result))

# set the ID of the symbol cube to 99 instead of -1 to include it in the segmentation image
result = client.simSetSegmentationObjectID("symbol_cube", SYMBOL_CUBE_SEGMENTATION_ID, True)
symbol_cube_id = client.simGetSegmentationObjectID("symbol_cube")
print("set symbol_cube id: {} -> {}".format(result, symbol_cube_id))

pose = client.simGetObjectPose(object_name="symbol_cube")

print("Initial pose:")
print(pose)

pitch_magnitude = 0.5
roll_magitude   = 0.5
yaw_magnitude   = 0.5
pitch_offset    = 0
roll_offset     = 0
yaw_offset      = 0

pitches = numpy.random.uniform(-pitch_magnitude + pitch_offset, pitch_magnitude + pitch_offset, total_images)
rolls   = numpy.random.uniform(-roll_magitude + roll_offset, roll_magitude + roll_offset, total_images)
yaws    = numpy.random.uniform(-yaw_magnitude + yaw_offset, yaw_magnitude + yaw_offset, total_images)
# xs = numpy.full(total_images, pose.position.x_val)
# ys = numpy.full(total_images, pose.position.y_val)
# zs = numpy.full(total_images, pose.position.z_val)

xs = numpy.random.uniform(1, 5, total_images)
ys = numpy.random.uniform(-2.5, 2.5, total_images)
zs = numpy.random.uniform(-1, -3, total_images)

for i in range(total_images):
    yaw = yaws[i]
    pitch = pitches[i]
    roll = rolls[i]
    pose.orientation = airsim.to_quaternion(pitch, roll, yaw)

    x = xs[i]
    y = ys[i]
    z = zs[i]

    pose.position.x_val = x
    pose.position.y_val = y
    pose.position.z_val = z

    # set the pose of the symbol_cube and assert that it was successful
    assert client.simSetObjectPose(object_name="symbol_cube", pose=pose)

    # request Scene and Segmentation images
    request = [
        airsim.ImageRequest(camera_name=0, image_type=airsim.ImageType.Scene, pixels_as_float=False, compress=False),
        airsim.ImageRequest(camera_name=0, image_type=airsim.ImageType.Segmentation, pixels_as_float=False, compress=False)]
    response = client.simGetImages(request)

    # format Scene image
    img1d = numpy.frombuffer(response[SCENE].image_data_uint8, dtype=numpy.uint8)
    img_rgb = img1d.reshape(response[SCENE].height, response[SCENE].width, 3)

    # generate a unique filename
    id_encoding = "{},{},{},{},{},{},{}".format(scenario_name, x, y, z, pitch, roll, yaw)
    filename = str(uuid.uuid3(uuid.NAMESPACE_X500, id_encoding)).replace("-", "") + ".png"
    full_filename = "{}/{}".format(image_directory, filename)

    # write Scene image to png
    airsim.write_png(os.path.normpath(full_filename), img_rgb)

    # format Segmentation image
    img1d = numpy.frombuffer(response[SEGMENTATION].image_data_uint8, dtype=numpy.uint8)
    img_rgb = img1d.reshape(response[SEGMENTATION].height, response[SEGMENTATION].width, 3)

    # slice the Segmentation image to get rid of RGB
    # we know that this works with segmentation id of 99 (should work generally but depends on color palette)
    mask_image = img_rgb[:,:,0]

    # get the mask
    mask = get_rle(mask_image)

    # show reconstructed mask (for debugging)
    # reconstructed = rle_decode(mask, mask_image.shape)
    # plt.imshow(reconstructed)
    # plt.show()

    # enter all the data into a dictionary and append it to the dataframe
    row_data = [x, y, z, yaw, pitch, roll, filename, mask]
    row_dict = dict(zip(columns, row_data))
    data = data.append(row_dict, ignore_index=True)

    # notify periodically
    if( i % 20 == 0 ):
        # print("{}/{}={:0.2f}%: writing {} ({} x {} px) for x={}, y={}, z={}, pitch={}, roll={}, yaw={}".format(i, total_images, float(i) / total_images * 100, full_filename, img_rgb.shape[0], img_rgb.shape[1], x, y, z, pitch, roll, yaw))
        print("{}/{}={:0.2f}%: writing {}".format(i, total_images, float(i) / total_images * 100, full_filename))

full_output_filename = "{}/{}".format(image_directory, output_csv_name)
data.to_csv(full_output_filename)
print("Saving output to {}".format(full_output_filename))