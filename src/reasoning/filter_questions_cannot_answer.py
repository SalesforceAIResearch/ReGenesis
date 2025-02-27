import json
from read_datasets import load_gsm8k, load_numglue, load_tracking_shuffled_objects, load_logical_deduction, load_arc_c, load_eurus

def filter_questions_cannot_answer(dataset_name, filtered_json, number=None):
    instructions = set()
    with open(filtered_json, 'r') as f:
        dataset_json = json.load(f)
    for data in dataset_json:
        instructions.add(data['instruction'])

    if dataset_name == "gsm8k":
        dataset = load_gsm8k()
    elif dataset_name == "numglue":
        dataset = load_numglue()
    elif "tracking" in dataset_name:
        dataset = load_tracking_shuffled_objects(split="train", number=number)
    elif "logical" in dataset_name:
        dataset = load_logical_deduction(split="train", number=number)
    elif "arc" in dataset_name:
        dataset = load_arc_c()
    elif "eurus" in dataset_name: #eurus_Logic_reclor
        _, task_remain, dataset_remain = dataset_name.split('_')
        dataset = load_eurus(task_remain=task_remain, dataset_remain=dataset_remain, clean=True)

    need_redone = dict()
    need_redone["instruction"] = []
    need_redone["answer"] = []
    need_redone_set = set()
    for i, ins in enumerate(dataset["instruction"][:27000]):
        if ins not in instructions and ins not in need_redone_set:
            need_redone_set.add(ins)
            need_redone["instruction"].append(ins)
            need_redone["answer"].append(dataset["answer"][i])

    path = "./step3.json.json"
    with open(path, 'w') as f:
        json.dump(need_redone, f)

if __name__ == "__main__":
    number = 5
    filter_questions_cannot_answer("eurus_Logic_reclor",  # dataset name
                                   "./step2.json",  # last round file
                                   number=number)