#!/usr/bin/python3

# for use with AirSim in "ComputerVision" mode

# indices
SCENE = 0
SEGMENTATION = 1
SYMBOL_CUBE_SEGMENTATION_ID = 99    # arbitrary value != -1 and in [0, 255]

CLEAR_EXISTING_DATASET = True       # clears everything in the dataset directory before generating new data
DRAW_BOUNDING_BOX = False           # only for debugging, set to False for dataset generation

# to do!!!!!!!!!!!!!!!!!!!!!
DEPTH_SCALE = 1


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
from scipy.spatial.transform import Rotation as R
from sklearn.model_selection import train_test_split
import yaml
# from ruamel.yaml import YAML as yaml

# file locations
dataset_directory = "dataset/data"
image_directory = "rgb"
mask_directory = "mask"
object_id = "01"
scenario_name = "blocks"
output_csv_name = "labels.csv"

metadata_directory = "{}/{}/".format(dataset_directory, object_id)

obj_id = 1

if( CLEAR_EXISTING_DATASET and os.path.exists(dataset_directory) ):
    print("Removing all existing data!")
    shutil.rmtree(dataset_directory)

if not os.path.exists(dataset_directory):
    os.makedirs(dataset_directory)
if not os.path.exists("{}/{}/{}".format(dataset_directory, object_id, image_directory)):
    os.makedirs("{}/{}/{}".format(dataset_directory, object_id, image_directory))
if not os.path.exists("{}/{}/{}".format(dataset_directory, object_id, mask_directory)):
    os.makedirs("{}/{}/{}".format(dataset_directory, object_id, mask_directory))

# labels CSV
columns = "object_id,filename,translation_x,translation_y,translation_z,rotation_yaw,rotation_pitch,rotation_roll,bounding_box,cam_K,depth_scale,cam_R_m2c,cam_t_m2c,obj_bb,obj_id".split(",")
data = pandas.DataFrame(columns=columns)

total_images = 50000
if( len(sys.argv) > 1 ):
    total_images = int(sys.argv[1])

print("Generating {} images with randomized pose.".format(total_images))

def linemod_yml_dump(data, filename, create_arrays=True):

    data_dictionary = data.to_dict(orient="index")

    if( create_arrays ):
        for key in data_dictionary:
            data_dictionary[key] = [data_dictionary[key]]

    text = yaml.dump(
        data_dictionary, width=float("inf"), indent=2, default_flow_style=False
    )

    text = text.replace("'", "")

    with open(metadata_directory + filename, "w") as file:
        file.write(text)

def yml_int_array_to_string(array, separator=", "):
    str_rep = separator.join("{}".format(element) for element in array)
    str_rep = "[{}]".format(str_rep)
    return str_rep

def yml_float_array_to_string(array, separator=", "):
    str_rep = separator.join("{:.8f}".format(element) for element in array)
    str_rep = "[{}]".format(str_rep)
    return str_rep

# def linemod_yaml

def generate_gt(data):
    linemod_yml_dump(data[["cam_R_m2c","cam_t_m2c", "obj_bb", "obj_id"]], "gt.yml", create_arrays=True)

def generate_info(data):
    linemod_yml_dump(data[["cam_K","depth_scale"]], "info.yml", create_arrays=False)

def generate_train_test(data):

    output_data = data["filename"].apply(lambda x: "{:04d}".format(int(x.replace(".png", ""))))

    train = output_data.sample(frac=0.8, random_state=200)
    test  = output_data.drop(train.index)

    train.to_csv(metadata_directory + "train.txt", index=False, header=False)
    test.to_csv(metadata_directory + "test.txt", index=False, header=False)

def save_metadata():

    # save labels
    full_output_filename = "{}/{}".format(dataset_directory, output_csv_name)
    data.to_csv(full_output_filename)

    # get/save camera info
    camera_info = client.simGetCameraInfo(camera_name=1)
    with open("{}/camera_info.txt".format(dataset_directory), "w") as camera_info_file:
        camera_info_file.write(str(camera_info))

    generate_gt(data)
    generate_info(data)
    generate_train_test(data)

    print("Saved metadata!")

def sigint_handler(sig, frame):
    save_metadata()
    sys.exit(0)
signal.signal(signal.SIGINT, sigint_handler)

client = airsim.MultirotorClient()

object_names = client.simListSceneObjects('[\w]*')
for name in object_names:
    if "symbol" in name:
        print(name)

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
    # rgb_filename = str(uuid.uuid3(uuid.NAMESPACE_X500, id_encoding)).replace("-", "")
    rgb_filename = "{:04d}".format(i)
    # mask_filename = rgb_filename + "_mask"
    mask_filename = rgb_filename
    rgb_filename += ".png"
    mask_filename += ".png"

    full_rgb_filename = "{}/{}/{}/{}".format(dataset_directory, object_id, image_directory, rgb_filename)

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
    obj_bb = yml_int_array_to_string([bb_x, bb_y, bb_w, bb_h])

    (thresh, img_rgb) = cv2.threshold(img_rgb, 1, 255, cv2.THRESH_BINARY)

    if( DRAW_BOUNDING_BOX ):
        print("(", bb_x, bb_y, bb_w, bb_h, ")")
        point_1 = (bb_x, bb_y)
        point_2 = (bb_x + bb_w, bb_y + bb_h)
        color = (0, 0, 255) # BGR
        cv2.rectangle(img_rgb, point_1, point_2, thickness=1, color=color)

    full_mask_filename = "{}/{}/{}/{}".format(dataset_directory, object_id, mask_directory, mask_filename)
    airsim.write_png(os.path.normpath(full_mask_filename), img_rgb)


    rotation_matrix_raw = (R.from_quat([pose.orientation.x_val, pose.orientation.y_val, pose.orientation.z_val, pose.orientation.w_val])).as_matrix().flatten()
    rotation_matrix = yml_float_array_to_string(rotation_matrix_raw)
    translation_vector = yml_float_array_to_string([pose.position.x_val, pose.position.y_val, pose.position.z_val])

    # enter all the data into a dictionary and append it to the dataframe

    camera_matrix = [572.4114, 0.0, 325.2611, 0.0, 573.57043, 242.04899, 0.0, 0.0, 1.0]
    camera_matrix = yml_int_array_to_string(camera_matrix)

    row_data = [1, rgb_filename, x, y, z, yaw, pitch, roll, bounding_box, camera_matrix, DEPTH_SCALE, rotation_matrix, translation_vector, obj_bb, obj_id]
    row_dict = dict(zip(columns, row_data))
    data = data.append(row_dict, ignore_index=True)

    print("\r{}/{}={:0.2f}%: writing {}".format(i, total_images, float(i) / total_images * 100, rgb_filename), end="")

print("\nDone generating rgbs and masks!")

save_metadata()