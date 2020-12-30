#!/usr/bin/python3

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

    assert client.simSetObjectPose(object_name="symbol_cube", pose=pose)
    # result = client.simSetObjectPose(object_name="symbol_cube", pose=pose, teleport=False)
    # print(result)
    # assert result == True

    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
    response = responses[0]
    img1d = numpy.frombuffer(response.image_data_uint8, dtype=numpy.uint8)
    img_rgb = img1d.reshape(response.height, response.width, 3)

    id_encoding = "{},{},{},{},{},{},{}".format(scenario_name, x, y, z, pitch, roll, yaw)

    filename = str(uuid.uuid3(uuid.NAMESPACE_X500, id_encoding)).replace("-", "") + ".png"
    full_filename = "{}/{}".format(image_directory, filename)

    # write to png
    print("{}/{}={:0.2f}%: writing {} ({} x {} px) for x={}, y={}, z={}, pitch={}, roll={}, yaw={}".format(i, total_images, float(i) / total_images * 100, full_filename, img_rgb.shape[0], img_rgb.shape[1], x, y, z, pitch, roll, yaw))
    airsim.write_png(os.path.normpath(full_filename), img_rgb)

    row_data = [x, y, z, yaw, pitch, roll, filename]
    row_dict = dict(zip(columns, row_data))
    data = data.append(row_dict, ignore_index=True)

full_output_filename = "{}/{}".format(image_directory, output_csv_name)
data.to_csv(full_output_filename)
print("Saving output to {}".format(full_output_filename))
