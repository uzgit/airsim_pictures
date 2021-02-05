import numpy
import pandas
pandas.set_option('display.max_rows', 500)
pandas.set_option('display.max_columns', 500)
pandas.set_option('display.width', 1000)

from scipy.spatial.transform import Rotation as R

import yaml

from yaml import CDumper
from yaml.representer import SafeRepresenter

data = pandas.read_csv("/home/joshua/git/EfficientPose/dataset/data/labels.csv", index_col=0)

data["translation_x_str"] = data["translation_x"].astype(str)
data["translation_y_str"] = data["translation_y"].astype(str)
data["translation_z_str"] = data["translation_z"].astype(str)

data["cam_R_m2c"] = data.apply(lambda x : R.from_euler("zyx", [data["rotation_pitch"], data["rotation_roll"], data["rotation_yaw"]], degrees=False))

data["cam_t_m2c_s"] = "[" + data["translation_x_str"] + ", " + data["translation_y_str"] + ", " + data["translation_z_str"] + "]"
data["cam_t_m2c"] = numpy.array(data["cam_t_m2c_s"])

data["obj_bb"] = data["bounding_box"].astype(str)
data["obj_bb"] = data["obj_bb"].apply(lambda x: str(x.split(" ")))

data["obj_id"] = 1 # we only have one item

data = data.head()
print(data)

print(len(data))

gt_output_data = data[["cam_t_m2c", "obj_bb", "obj_id"]]

# yaml.dump()

gt_text = yaml.dump(
    gt_output_data.to_dict(orient="records"),
    sort_keys=True, width=72, indent=2,
    default_flow_style=False
)

gt_text = gt_text.replace("'", "")

# gt_text = text.replace("{", "")
# gt_text = text.replace("}", "")

with open("gt.yml", "w") as gt_file:
    gt_file.write(gt_text)

