import json
import random
res = []

## after process reason -> FT
paths = ["step4-2.json"]

for path in paths:
    with open(path, "r") as f:
        data = json.load(f)
    for d in data:
        element = {}
        element[
            "instruction"] = d["instruction"]
        element["answer"] = d["answer"].replace('\\boxed{', '').replace('}', '')
        res.append(element)

random.shuffle(res)
random.shuffle(res)
random.shuffle(res)

with open("./ready-for-FT.json", 'w') as f:
    json.dump(res, f)