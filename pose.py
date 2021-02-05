#!/usr/bin/python3

import sys
sys.path.insert(0, "/home/joshua/git/Airsim/PythonClient")

import airsim

client = airsim.MultirotorClient()

pose = client.simGetObjectPose(object_name="symbol_cube")

print("Initial pose:")
print(pose)