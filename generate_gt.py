import pandas
import yaml
from yaml import CDumper
from yaml.representer import SafeRepresenter

data = pandas.read_csv("/home/joshua/PycharmProjects/airsim_pictures/dataset_linemod/data/labels.csv", index_col=0)

print(data.head())

text = yaml.dump(
    data.to_dict(orient="records"),
    sort_keys=False, width=72, indent=2,
    default_flow_style=None
)

text = text.replace("{", "")
text = text.replace("}", "")

print(text)