#!/usr/bin/python3

import sys
sys.path.insert(0, "/home/joshua/git/Airsim/PythonClient")

import airsim

client = airsim.MultirotorClient()

object_names = client.simListSceneObjects('[\w]*')
# for object_name in object_names:
#     print(object_name)
    # pose = client.simGetObjectPose(object_name=object_name)
    # pose.position.x_val += 10
    # result = client.simSetObjectPose(object_name=object_name, pose=pose)

assert ("symbol_cube" in object_names)

for i in range(10):
    pose = client.simGetObjectPose(object_name="symbol_cube")
    print(pose)

    pose.position.x_val -= 0.2
    assert client.simSetObjectPose(object_name="symbol_cube", pose=pose)
    # print(result)