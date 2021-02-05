import yaml
import pandas

metadata_directory = ""

data = pandas.read_csv("dataset/data/labels.csv")

def linemod_yml_dump(data, filename):

    data_dictionary = data.to_dict(orient="index")
    output_dictionary = {}

    print(data_dictionary)
    for key in data_dictionary:
        data_dictionary[key] = [data_dictionary[key]]

    text = yaml.dump(
        data_dictionary, width=float("inf"), indent=2, default_flow_style=False
    )

    text = text.replace("'", "")

    with open(metadata_directory + filename, "w") as file:
        file.write(text)

linemod_yml_dump(data[["cam_R_m2c","cam_t_m2c", "obj_bb", "obj_id"]], "test.yml")