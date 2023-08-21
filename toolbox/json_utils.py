import json


def load_json(json_file):
    data = []
    with open(json_file, 'r') as fin:
        for line in fin.readlines():
            data.append(json.loads(line))

    if len(data) == 1:
        data = data[0]
    return data


def save_json(json_file, data):
    with open(json_file, 'w') as fout:
        json.dump(data, fout, indent=2)
