#!/usr/bin/python3

# for use with AirSim in "ComputerVision" mode

# indices
SCENE = 0
SEGMENTATION = 1
SYMBOL_CUBE_SEGMENTATION_ID = 99    # arbitrary value != -1 and in [0, 255]

CLEAR_EXISTING_DATASET = True       # clears everything in the dataset directory before generating new data
DRAW_BOUNDING_BOX = False           # only for debugging, set to False for dataset generation

from datetime import datetime
import numpy
import os
import uuid
import signal
import sys
sys.path.insert(0, "/home/joshua/git/Airsim/PythonClient")
import airsim
import pandas
import cv2
import shutil

# file locations
dataset_directory = "dataset"
image_directory = "images"
mask_directory = "masks"
scenario_name = "blocks"
output_csv_name = "labels.csv"

if( CLEAR_EXISTING_DATASET and os.path.exists(dataset_directory) ):
    print("Removing all existing data!")
    shutil.rmtree(dataset_directory)

if not os.path.exists(dataset_directory):
    os.makedirs(dataset_directory)
if not os.path.exists("{}/{}".format(dataset_directory, image_directory)):
    os.makedirs("{}/{}".format(dataset_directory, image_directory))
if not os.path.exists("{}/{}".format(dataset_directory, mask_directory)):
    os.makedirs("{}/{}".format(dataset_directory, mask_directory))

# labels CSV
columns = "x,y,z,yaw,pitch,roll,bounding_box,rgb_filename,mask_filename".split(",")
data = pandas.DataFrame(columns=columns)

total_images = 50000
if( len(sys.argv) > 1 ):
    total_images = int(sys.argv[1])

print("Generating {} images with randomized pose.".format(total_images))

def save_metadata():

    # save labels
    full_output_filename = "{}/{}".format(dataset_directory, output_csv_name)
    data.to_csv(full_output_filename)

    # get/save camera info
    camera_info = client.simGetCameraInfo(camera_name=1)
    with open("{}/camera_info.txt".format(dataset_directory), "w") as camera_info_file:
        camera_info_file.write(str(camera_info))

    print("Saved metadata!")

def sigint_handler(sig, frame):
    save_metadata()
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
yaw_magnitude   = numpy.pi
pitch_offset    = 0
roll_offset     = 0
yaw_offset      = 0

# generate poses
pitches = numpy.random.uniform(-pitch_magnitude + pitch_offset, pitch_magnitude + pitch_offset, total_images)
rolls   = numpy.random.uniform(-roll_magitude + roll_offset, roll_magitude + roll_offset, total_images)
yaws    = numpy.random.uniform(-yaw_magnitude + yaw_offset, yaw_magnitude + yaw_offset, total_images)

# for camera pose
# xs = numpy.random.uniform(0, 25, total_images)
# ys = numpy.random.uniform(-10, 10, total_images)
# zs = numpy.random.uniform(-10, 10, total_images)

# for object pose
xs = numpy.random.uniform(-5, 5, total_images)
ys = numpy.random.uniform(-5, 5, total_images)
zs = numpy.random.uniform(1, 29, total_images)

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
    # client.simSetCameraPose(camera_name=0, pose=pose)

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
    rgb_filename = str(uuid.uuid3(uuid.NAMESPACE_X500, id_encoding)).replace("-", "")
    # mask_filename = rgb_filename + "_mask"
    mask_filename = rgb_filename
    rgb_filename += ".png"
    mask_filename += ".png"

    full_rgb_filename = "{}/{}/{}".format(dataset_directory, image_directory, rgb_filename)

    # write Scene image to png
    airsim.write_png(os.path.normpath(full_rgb_filename), img_rgb)

    # format Segmentation image
    img1d = numpy.frombuffer(response[SEGMENTATION].image_data_uint8, dtype=numpy.uint8)
    img_rgb = img1d.reshape(response[SEGMENTATION].height, response[SEGMENTATION].width, 3)

    # slice the Segmentation image to get rid of RGB
    # we know that this works with segmentation id of 99 (should work generally but depends on color palette)
    mask_image = img_rgb[:,:,0]
    bb_x, bb_y, bb_w, bb_h = cv2.boundingRect(mask_image)
    bounding_box = "{} {} {} {}".format(bb_x, bb_y, bb_w, bb_h)

    if( DRAW_BOUNDING_BOX ):
        print("(", bb_x, bb_y, bb_w, bb_h, ")")
        point_1 = (bb_x, bb_y)
        point_2 = (bb_x + bb_w, bb_y + bb_h)
        color = (0, 0, 255) # BGR
        cv2.rectangle(img_rgb, point_1, point_2, thickness=1, color=color)

    full_mask_filename = "{}/{}/{}".format(dataset_directory, mask_directory, mask_filename)
    airsim.write_png(os.path.normpath(full_mask_filename), img_rgb)

    # enter all the data into a dictionary and append it to the dataframe
    row_data = [x, y, z, yaw, pitch, roll, bounding_box, rgb_filename, mask_filename]
    row_dict = dict(zip(columns, row_data))
    data = data.append(row_dict, ignore_index=True)

    # notify periodically
    if( i % 20 == 0 ):
        print("{}/{}={:0.2f}%: writing {}".format(i, total_images, float(i) / total_images * 100, rgb_filename))

save_metadata()