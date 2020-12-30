#!/usr/bin/python3
import airsim
from datetime import datetime
import numpy
import os
import uuid
import signal
import sys
sys.path.insert(0, "/home/joshua/git/Airsim/PythonClient")
import airsim
import pandas

image_directory = "images"
scenario_name = "blocks"
output_csv_name = "data.csv"

columns = "x,y,z,yaw,pitch,roll,filename".split(",")
data = pandas.DataFrame(columns=columns)

def sigint_handler(sig, frame):
    full_output_filename = "{}/{}".format(image_directory, output_csv_name)
    data.to_csv(full_output_filename)
    print("Saving output to {}".format(full_output_filename))
    sys.exit(0)
signal.signal(signal.SIGINT, sigint_handler)

pitches = numpy.linspace(-80, 80, 50)
rolls   = numpy.linspace(-180, 180, 50)
yaws    = numpy.linspace(-80, 80, 50)

client = airsim.MultirotorClient()
initial_pose = actual_pose = client.simGetObjectPose(object_name="symbol_cube")
x = initial_pose.position.x_val
y = initial_pose.position.y_val
z = initial_pose.position.z_val

index = 0
for yaw in yaws:
    for pitch in pitches:
        for roll in rolls:

            index += 1

            pose = airsim.Pose()
            pose.orientation = airsim.to_quaternion(pitch, roll, yaw)
            pose.position.x_val = x
            pose.position.y_val = y
            pose.position.z_val = z
            result = client.simSetObjectPose(object_name="symbol_cube", pose=pose)
            # assert result == True

            # actual_pose = client.simGetObjectPose(object_name="symbol_cube")
            # print(actual_pose)

            responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
            response = responses[0]
            img1d = numpy.frombuffer(response.image_data_uint8, dtype=numpy.uint8)
            img_rgb = img1d.reshape(response.height, response.width, 3)

            id_encoding = "{},{},{},{},{},{},{}".format(scenario_name, x, y, z, pitch, roll, yaw)

            filename = str(uuid.uuid3(uuid.NAMESPACE_X500, id_encoding)).replace("-", "") + ".png"
            # filename = "blocks_{}.png".format(index)

            full_filename = "{}/{}".format(image_directory, filename)
            # filename = filename.replace(" ", "_")

            # write to png
            print("writing {} ({} x {} px) for pitch={}, roll={}, yaw={}".format(full_filename, img_rgb.shape[0], img_rgb.shape[1], pitch, roll, yaw))
            airsim.write_png(os.path.normpath(full_filename), img_rgb)

            row_data = [x, y, z, yaw, pitch, roll, filename]
            row_dict = dict(zip(columns, row_data))
            data = data.append(row_dict, ignore_index=True)

full_output_filename = "{}/{}".format(image_directory, output_csv_name)
data.to_csv(full_output_filename)
print("Saving output to {}".format(full_output_filename))

# print("dimensions: {} x {}".format(img_rgb.shape[0], img_rgb.shape[1]))

# for i in range(200):
#     pose = client.simGetObjectPose(object_name="symbol_cube")
#     pose.position.x_val -= 0.2
#     result = client.simSetObjectPose(object_name="symbol_cube", pose=pose)
#
#     responses = client.simGetImages([
#         # png format
#         airsim.ImageRequest(0, airsim.ImageType.Scene),
#         # uncompressed RGB array bytes
#         # airsim.ImageRequest(1, airsim.ImageType.Scene, False, False),
#         # floating point uncompressed image
#         # airsim.ImageRequest(1, airsim.ImageType.DepthPlanner, True)
#     ])
#
#     # do something with response which contains image data, pose, timestamp etc
#
#     responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
#     response = responses[0]
#
#     img1d = numpy.frombuffer(response.image_data_uint8, dtype=numpy.uint8)
#
#     # reshape array to 4 channel image array H X W X 4
#     img_rgb = img1d.reshape(response.height, response.width, 3)
#     # print("response dimensions: {} x {}".format(response.height, response.width))
#
#     # original image is flipped vertically
#     # img_rgb = numpy.flipud(img_rgb)
#
#     # filename = "airsim_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
#     # pose.orientation.
#     # id_encoding = "{},{},{},{},{},{},{}".format()
#     filename = str(uuid.uuid3(uuid.NAMESPACE_X500, "blocks,1.0,1.0,1.0")).replace("-", "") + ".png"
#     full_filename = "{}/{}".format(image_directory, filename)
#     # filename = filename.replace(" ", "_")
#
#     # write to png
#     print("writing {} ...".format(full_filename))
#     airsim.write_png(os.path.normpath(full_filename), img_rgb)
#     print("dimensions: {} x {}".format(img_rgb.shape[0], img_rgb.shape[1]))