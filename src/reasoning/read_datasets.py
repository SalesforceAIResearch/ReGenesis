from datasets import load_dataset
import random
from datasets import Dataset
from collections import defaultdict, Counter
import string
import json
import os
import re
from tqdm import tqdm


def extract_boxed(text):
    match = re.search(r'\\boxed\{(.*)\}', text)
    if match:
        text = match.group(1)
        l_find = False
        return_idx = len(text)
        for i in range(len(text) - 1, 0, -1):
            if text[i] == '{':
                l_find = True
            if text[i] == '}':
                if l_find:
                    return_idx = i
                    l_find = False
                else:
                    return text[:return_idx]
        return text[:return_idx]

    else:
        return None


def filter_answer(answers):
    res = []
    length = len(answers)
    for a in answers:
        res += a.strip().split(' ')
    count = Counter(res)
    answers_res = []
    for a in answers:
        a = a.strip().split()
        a_new = ' '.join([_a for _a in a if count[_a] < length])
        answers_res.append(a_new)
    return answers_res


def load_copa(name="pkavumba/balanced-copa", type="llama", examples=5, instruction=True, load_part="test"):
    dataset = load_dataset(name,
                           split=load_part)  # ['label', 'id', 'premise', 'question', 'choice1', 'choice2', 'mirrored'
    dataset_train = load_dataset(name, split='train')
    question_map = {"cause": "What was the CAUSE of this? (Note: You must choose from Alternative 1 and Alternative 2)",
                    "effect": "What happened as a RESULT? (Note: You must choose from Alternative 1 and Alternative 2)"}
    add_on = 'Instruction: The following question is composed of a paragraph and two alternatives, where the task is to select the alternative that more plausibly has a causal relation with the premise.\n'
    label_map = {0: "Alternative 1", 1: "Alternative 2"}
    new_dataset = dict()
    new_dataset["instruction"] = []
    new_dataset["answer"] = []
    res = []
    example_string = ""
    if type == "llama":
        for i in range(len(dataset["label"])):
            d = ""
            if instruction:
                d += add_on
            d += 'Paragraph: ' + dataset["premise"][i] + '\n' + 'Alternative 1: ' + dataset["choice1"][
                i] + '\n' + 'Alternative 2: ' + dataset["choice2"][i]
            d += '\n' + 'Question: ' + question_map[dataset["question"][i]]
            new_dataset["instruction"].append(d)
            new_dataset["answer"].append(label_map[dataset["label"][i]])
        for i in range(len(dataset_train["label"])):
            d = ""
            if instruction:
                d += add_on
            d += 'Paragraph: ' + dataset_train["premise"][i] + '\n' + 'Alternative 1: ' + dataset_train["choice1"][
                i] + '\n' + 'Alternative 2: ' + dataset_train["choice2"][i]
            d += '\n' + 'Question: ' + question_map[dataset_train["question"][i]]
            d += '[/INST]Answer: ' + label_map[dataset_train["label"][i]]
            res.append(d)
        if examples > 0:
            indices = list(range(len(dataset_train["premise"])))
            # Shuffle the list
            random.shuffle(indices)
            # Return the first n indices from the shuffled list
            index = indices[:examples]
            # Instruction:\n{instruction}[/INST]
            for idx in index:
                example_string += res[idx] + '[INST]'

    return Dataset.from_dict(new_dataset), example_string, add_on


def load_reasoning(name="tasksource/bigbench", type="llama", examples=2, instruction=True):
    name = '/'.join(name.split('/')[:2])
    dataset = load_dataset(name, 'minute_mysteries_qa',
                           split="train")  # ['label', 'id', 'premise', 'question', 'choice1', 'choice2', 'mirrored'
    dataset_val = load_dataset(name, 'minute_mysteries_qa', split="validation")
    new_dataset = dict()
    new_dataset["instruction"] = []
    new_dataset["answer"] = []
    length_instruction = defaultdict(list)
    add_on = "Instruction: You will read a paragraph and at the end of the story, there is a question. then you will choose one from the given choices to answer the question.\n\n"

    for i in range(len(dataset["inputs"])):
        ins = dataset["inputs"][i].replace("Story:", "Paragraph:")
        if instruction:
            ins = add_on + ins
        ins = ins[:-7] + "Please choose one from the above choices."
        ans = dataset["targets"][i][0]
        if len(ans.split()) < 15 and "choice: " in ins:
            new_dataset["instruction"].append(ins)
            new_dataset["answer"].append(ans)
            length_instruction[len(ins.split())].append([ins, ans])

    for i in range(len(dataset_val["inputs"])):
        ins = dataset_val["inputs"][i].replace("Story:", "Paragraph:")
        if instruction:
            ins = add_on + ins
        ins = ins[:-7] + "Please choose one from the above choices."
        ans = dataset_val["targets"][i][0]
        if len(ans.split()) < 15 and "choice: " in ins:
            new_dataset["instruction"].append(ins)
            new_dataset["answer"].append(ans)
            length_instruction[len(ins.split())].append([ins, ans])
    example_string = ""

    count = 0
    for length in sorted(list(length_instruction.keys())):
        for ins, ans in length_instruction[length]:
            example_string += ins + '[/INST]choice:' + ans + '[INST]'
            count += 1
            if count >= examples:
                break
        if count >= examples:
            break

    return Dataset.from_dict(new_dataset), example_string, add_on


def load_yes_or_no(name="tasksource/bigbench", type="llama", examples=5, instruction=True):
    name, split_name = '/'.join(name.split('/')[:2]), name.split('/')[-1]
    dataset = load_dataset(name, split_name,
                           split="train")  # ['label', 'id', 'premise', 'question', 'choice1', 'choice2', 'mirrored'
    dataset_val = load_dataset(name, split_name, split="validation")
    new_dataset = dict()
    new_dataset["instruction"] = []
    new_dataset["answer"] = []
    add_on = "Instruction: You will read a question, then you will answer 'Yes' or 'No'.\n\n"

    for i in range(len(dataset["inputs"])):
        ins = dataset["inputs"][i]
        if instruction:
            ins = add_on + ins
        ins = ins.replace('Q:', 'Question:').replace('A:', '')
        ans = dataset["targets"][i][0].replace('.', '').strip()
        new_dataset["instruction"].append(ins.strip())
        new_dataset["answer"].append(ans)

    for i in range(len(dataset_val["inputs"])):
        ins = dataset_val["inputs"][i]
        if instruction:
            ins = add_on + ins
        ins = ins.replace('Q:', 'Question:').replace('A:', '')
        ans = dataset_val["targets"][i][0].replace('.', '').strip()
        new_dataset["instruction"].append(ins)
        new_dataset["answer"].append(ans)

    example_string = ""

    for i in range(examples):
        ins = new_dataset["instruction"].pop(-1)
        ans = new_dataset["answer"].pop(-1)
        example_string += ins + '[/INST]Answer: ' + ans + '[INST]'

    print(len(new_dataset["instruction"]))

    return Dataset.from_dict(new_dataset), example_string, add_on


def load_multiple_choice(name, type="llama", split="train", examples=0, instruction=True):
    name, split_name = '/'.join(name.split('/')[:2]), name.split('/')[-1]
    dataset = load_dataset(name, split_name, split=split)
    print(dataset)
    new_dataset = dict()
    new_dataset["instruction"] = []
    new_dataset["answer"] = []

    add_on = ""  # Instruction: You will read a paragraph and then deduce the order of a sequence of objects\n\n

    for i in range(len(dataset["inputs"])):
        ins = dataset["inputs"][i]
        add_on = ins.split('\n')[0]
        if not instruction:
            ins = "Paragraph: " + ins.split('\n')[-1]
        # ins = ins + "\nQuestion:Please choose one from the following choices."
        choices = dataset["multiple_choice_targets"][i]
        # letters = list(string.ascii_uppercase)
        # connected_string = "\n".join(f"{letters[i]}.{value}" for i, value in enumerate(choices))
        # ins += '\nChoices:\n' + connected_string
        if choices:
            # ans = letters[dataset["multiple_choice_scores"][i].index(1)]
            ans = dataset["targets"][i][0]
            print('ans', ans)
            new_dataset["instruction"].append(ins)
            new_dataset["answer"].append(ans)

    example_string = ""

    for i in range(examples):
        ins = new_dataset["instruction"].pop(-1)
        ans = new_dataset["answer"].pop(-1)
        example_string += ins + '[/INST]choice:' + ans + '[INST]'

    return Dataset.from_dict(new_dataset), example_string, add_on


def load_multiple_choice_add_choices(name, type="vllm", split="train", examples=0, instruction=True):
    name, split_name = '/'.join(name.split('/')[:2]), name.split('/')[-1]
    dataset = load_dataset(name, split_name, split=split)
    print(dataset)
    new_dataset = dict()
    new_dataset["instruction"] = []
    new_dataset["answer"] = []
    new_dataset["multiple_choice_targets"] = []

    for i in range(len(dataset["inputs"])):
        ins = dataset["inputs"][i].strip() + '?'
        choices = '\n\nPlease choose from ' + ';'.join(dataset["multiple_choice_targets"][i]).replace('.', '')
        ans = dataset["targets"][i][0].replace('.', '')
        new_dataset["instruction"].append(ins + choices)
        new_dataset["answer"].append(ans)
        new_dataset["multiple_choice_targets"].append(dataset["multiple_choice_targets"][i])

    example_string = []
    if type == 'llama':
        for i in range(examples):
            ins = new_dataset["instruction"].pop(-1)
            ans = new_dataset["answer"].pop(-1)
            example_string.append({"role": "user", "content": ins})
            example_string.append({"role": "assistant", "content": ans})
    else:
        for i in range(examples):
            ins = new_dataset["instruction"].pop(-1)
            ans = new_dataset["answer"].pop(-1)
            example_string.append(ins)
            example_string.append(ans)

    return Dataset.from_dict(new_dataset), example_string, ''


def load_strategy_qa(name="ChilleD/StrategyQA", split="train"):
    dataset = load_dataset(name, split=split)  # 1603/ 687

    new_dataset = dict()
    new_dataset["instruction"] = []
    new_dataset["answer"] = []
    # new_dataset["facts"] = []

    for i in range(len(dataset["question"])):
        ins = str(dataset["facts"][i]) + '\n\n' + dataset["question"][i]
        ans = str(dataset["answer"][i])
        new_dataset["instruction"].append(ins)
        new_dataset["answer"].append(ans)

    return Dataset.from_dict(new_dataset)


def load_multiple_choice_q(name="tasksource/bigbench", type="llama", examples=2, instruction=True):
    name, split_name = '/'.join(name.split('/')[:2]), name.split('/')[-1]
    dataset = load_dataset(name, split_name, split="train")

    new_dataset = dict()
    new_dataset["instruction"] = []
    new_dataset["answer"] = []
    add_on = "Instruction: You will read a question, then you will choose one from the following choices.\n\n" if instruction else ""

    for i in range(len(dataset["inputs"])):
        ins = dataset["inputs"][i]
        add_on = ins.split('\n')[0]
        if not instruction:
            ins = "Question: " + ins.split('\n')[-1]
        ins = ins + "\nQuestion:Please choose one from the following choices."
        choices = dataset["multiple_choice_targets"][i]
        letters = list(string.ascii_uppercase)
        connected_string = "\n".join(f"{letters[i]}.{value}" for i, value in enumerate(choices))
        ins += '\nChoices:\n' + connected_string
        ans = letters[dataset["multiple_choice_scores"][i].index(1)]
        new_dataset["instruction"].append(ins)
        new_dataset["answer"].append(ans)

    example_string = ""

    for i in range(examples):
        ins = new_dataset["instruction"].pop(-1)
        ans = new_dataset["answer"].pop(-1)
        example_string += ins + '[/INST]choice:' + ans + '[INST]'

    return Dataset.from_dict(new_dataset), example_string, add_on


def load_clutrr(name="CLUTRR/v1", type="llama", examples=5, instruction=True):
    dataset = load_dataset(name, split="train")

    new_dataset = dict()
    new_dataset["instruction"] = []
    new_dataset["answer"] = []
    add_on = "Instruction: You will read a context, then you will complete the sentence.\n\n" if instruction else ""

    for i in range(len(dataset["inputs"])):
        ins = 'Context: ' + dataset["clean_story"][i]
        ins += '\nPlease complete this sentence:  '
        ins = ins + "\nQuestion:Please choose one from the following choices."
        choices = dataset["multiple_choice_targets"][i]
        letters = list(string.ascii_uppercase)
        connected_string = "\n".join(f"{letters[i]}.{value}" for i, value in enumerate(choices))
        ins += '\nChoices:\n' + connected_string
        ans = letters[dataset["multiple_choice_scores"][i].index(1)]
        new_dataset["instruction"].append(ins)
        new_dataset["answer"].append(ans)

    example_string = ""

    for i in range(examples):
        ins = new_dataset["instruction"].pop(-1)
        ans = new_dataset["answer"].pop(-1)
        example_string += ins + '[/INST]choice:' + ans + '[INST]'

    return Dataset.from_dict(new_dataset), example_string, add_on


def letter_choices(string_list):
    choices = []
    for index, item in enumerate(string_list):
        choice_letter = string.ascii_uppercase[index]
        choices.append(f"({choice_letter}) {item}")
    return " ".join(choices)


def load_tracking_shuffled_objects(name="tasksource/bigbench", split="train", number=3):
    """
    number = 3,5,7
    """
    # 'tracking_shuffled_objects_five_objects', 'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_three_objects'
    # split_names = ['boolean_expressions', 'causal_judgement', 'date_understanding', 'disambiguation_qa', 'dyck_languages', 'formal_fallacies', 'geometric_shapes', 'hyperbaton', 'logical_deduction_five_objects', 'logical_deduction_seven_objects', 'logical_deduction_three_objects', 'movie_recommendation', 'multistep_arithmetic_two', 'navigate', 'object_counting', 'penguins_in_a_table', 'reasoning_about_colored_objects', 'ruin_names', 'salient_translation_error_detection', 'snarks', 'sports_understanding', 'temporal_sequences', 'web_of_lies', 'word_sorting']
    if split == "train":
        dataset = load_dataset(name, 'tracking_shuffled_objects', split="train")
    else:
        dataset = load_dataset(name, 'tracking_shuffled_objects', split="validation")

    new_dataset = dict()
    new_dataset["instruction"] = []
    new_dataset["answer"] = []
    new_dataset["answer_choice"] = []
    new_dataset["answer_letter"] = []
    new_dataset["answer_choice_filter"] = []
    for i in range(len(dataset["inputs"])):
        ins = dataset["inputs"][i].strip()
        ans = dataset["targets"][i][0]
        multiple_choice_targets = dataset["multiple_choice_targets"][i]
        multiple_choice_scores = dataset["multiple_choice_scores"][i]
        ans = ans[:-1] if ans[-1] == '.' else ans
        index = multiple_choice_scores.index(1)
        ins += '?\nOptions: ' + letter_choices(multiple_choice_targets)
        if len(multiple_choice_targets) == number:
            new_dataset["instruction"].append(ins)
            ans_filter = filter_answer(multiple_choice_targets)[index].replace('.', '')
            new_dataset["answer_choice_filter"].append(ans_filter)
            new_dataset["answer_choice"].append(ans)
            new_dataset["answer_letter"].append('(' + string.ascii_uppercase[index] + ')')
            new_dataset["answer"].append('(' + string.ascii_uppercase[index] + ') ' + ans)

    return Dataset.from_dict(new_dataset)


def load_logical_deduction(name="tasksource/bigbench", split="train", number=3):
    """
    number = 3,5,7
    """
    # 'tracking_shuffled_objects_five_objects', 'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_three_objects'
    # split_names = ['boolean_expressions', 'causal_judgement', 'date_understanding', 'disambiguation_qa', 'dyck_languages', 'formal_fallacies', 'geometric_shapes', 'hyperbaton', 'logical_deduction_five_objects', 'logical_deduction_seven_objects', 'logical_deduction_three_objects', 'movie_recommendation', 'multistep_arithmetic_two', 'navigate', 'object_counting', 'penguins_in_a_table', 'reasoning_about_colored_objects', 'ruin_names', 'salient_translation_error_detection', 'snarks', 'sports_understanding', 'temporal_sequences', 'web_of_lies', 'word_sorting']
    if split == "train":
        dataset = load_dataset(name, 'logical_deduction', split="train")
    else:
        dataset = load_dataset(name, 'logical_deduction', split="validation")

    new_dataset = dict()
    new_dataset["instruction"] = []
    new_dataset["answer_choice"] = []
    new_dataset["answer_letter"] = []
    new_dataset["answer_choice_filter"] = []
    new_dataset["answer"] = []
    for i in range(len(dataset["inputs"])):
        choices = dataset["multiple_choice_targets"][i]
        ins = dataset["inputs"][i].strip() + '\nWhich statement is correct? Options: ' + letter_choices(choices)
        ans = dataset["targets"][i][0].strip()
        multiple_choice_targets = dataset["multiple_choice_targets"][i]
        multiple_choice_scores = dataset["multiple_choice_scores"][i]
        index = multiple_choice_scores.index(1)
        score = string.ascii_uppercase[index]
        letter = '(' + score + ')'
        ans = ans[:-1] if ans[-1] == '.' else ans
        multiple_choice_targets = dataset["multiple_choice_targets"][i]
        if len(multiple_choice_scores) == number:
            new_dataset["instruction"].append(ins)
            ans_filter = filter_answer(multiple_choice_targets)[index].replace('.', '')
            new_dataset["answer_choice_filter"].append(ans_filter)
            new_dataset["answer_choice"].append(ans)
            new_dataset["answer_letter"].append(letter)
            new_dataset["answer"].append(letter + ' ' + ans)

    return Dataset.from_dict(new_dataset)


# def load_gsm8k(name="gsm8k", split=None):
#     new_dataset = dict()
#     split = 'train' if not split else split
#     dataset = load_dataset(name, "main", split=split)
#     new_dataset["instruction"] = dataset["question"][:]
#     new_dataset["answer"] = dataset["answer"][:]
#
#     return Dataset.from_dict(new_dataset)

def reclor_eurus_process(instruction, ans):
    instruction = instruction.replace("Solve the following problem step-by-step:\n", "").split("Step ")[0]
    ans = ans.split('Answer:')[-1].replace("\\boxed{", "").replace("\n", "").strip()
    ans = ans[:-1] if ans[-1] in ['}', '.'] else ans
    ans = ans[0]

    if "Options:" not in instruction:
        return None, None, None, None
    ins = instruction.split("Options:")[-1].split('Step ')[0].split('\n')
    ins = [_ins.strip() for _ins in ins if _ins.strip()]
    if ins:
        dict_keys = {string.ascii_uppercase[i]: "" for i in range(7)}
        for element in ins:
            if len(element) > 2 and element[0] in dict_keys and '.' == element[1]:
                dict_keys[element[0]] = element[2:].strip()
                instruction = instruction.replace(element[:2], '(' + element[0] + ')')
            else:
                break
        answer_letter = '(' + ans + ')'
        answer_choice = dict_keys[ans]
        answer = answer_letter + ' ' + answer_choice
    else:
        return None, None, None, None
    return instruction, answer, answer_letter, answer_choice


def load_eurus(task_remain="Math_CoT", dataset_remain="gsm8k", clean=False):
    new_dataset = dict()
    # dataset = load_dataset("parquet", data_files={
    #     'train': '/export/atlas-eval/huggingface/datasets/UltraInteract_sft/0000_sft.parquet'})["train"]
    dataset = load_dataset("openbmb/UltraInteract_sft")["train"]
    dataset = dataset.filter(lambda example: example["dataset"] == dataset_remain).filter(
        lambda example: example["task"] == task_remain)

    if task_remain == "Math_CoT":
        dataset = dataset.filter(lambda example: "\\boxed{" in example["response"])

    if not clean:
        new_dataset["instruction"] = dataset["instruction"][:]
        new_dataset["answer"] = dataset["response"][:]

    else:
        file_name = "/export/atlas-eval/huggingface/datasets/eurus_clean/" + dataset_remain.lower() + '.json'
        if os.path.isfile(file_name):
            with open(file_name, 'r') as f:
                new_dataset = json.load(f)[0]
        else:
            instruction_set = set()
            new_dataset["instruction"] = []
            new_dataset["answer"] = []
            if dataset_remain == "reclor":
                new_dataset["answer_letter"] = []
                new_dataset["answer_choice"] = []
            for i in tqdm(range(len(dataset["instruction"]))):
                if dataset_remain == "reclor":
                    ins, ans, answer_letter, answer_choice = reclor_eurus_process(dataset["instruction"][i],
                                                                                  dataset["response"][i])
                else:
                    ins = dataset["instruction"][i].split("Answer}.\n")[-1].split("Step ")[0]
                    if dataset_remain in ["strategyqa", "hotpotqa"]:
                        ins = ins.split("Solve the following problem step-by-step:\n")[1].strip()
                    ans = dataset["response"][i]
                    if 'Answer:' not in ans:
                        ins = None
                    else:
                        ans = ans.split('Answer:')[-1].replace("\\boxed{", "").replace("\n", "").strip()
                        ans = ans[:-1] if ans[-1] in ['}', '.'] else ans

                if ins and ins not in instruction_set:
                    instruction_set.add(ins)
                    new_dataset["instruction"].append(ins)
                    print(ins)
                    new_dataset["answer"].append(ans)
                    if dataset_remain == "reclor":
                        new_dataset["answer_letter"].append(answer_letter)
                        new_dataset["answer_choice"].append(answer_choice)

            with open(file_name, 'w') as f:
                json.dump([new_dataset], f)
        # new_dataset["answer"] = [extract_boxed(a) for a in new_dataset["answer"]]
    return Dataset.from_dict(new_dataset)


def load_gsm(name="qintongli/GSM-Plus", add_on=False):
    new_dataset = dict()
    dataset = load_dataset(name, split='test')
    dataset = dataset.filter(lambda example: example["answer"].lower().strip() != 'none')
    new_dataset["instruction"] = dataset["question"][:]
    new_dataset["answer"] = dataset["answer"][:]
    new_dataset["instruction"] = [i.strip() for i in new_dataset["instruction"]]
    if add_on:
        add_on = "Solve the following math problem step-by-step.\nSimplify your answer as much as possible. Present your final answer as \\boxed{Your Answer}.\n"
        new_dataset["instruction"] = [add_on + i.strip() for i in new_dataset["instruction"]]
    new_dataset["answer"] = [i.strip() for i in new_dataset["answer"]]
    return Dataset.from_dict(new_dataset)


def load_svamp(name="Dahoas/prompted_svamp", add_on=False):
    new_dataset = dict()
    dataset = load_dataset(name, split='train')
    new_dataset["instruction"] = dataset["question"][:]
    new_dataset["answer"] = dataset["answer"][:]
    dataset = load_dataset(name, split='test')
    new_dataset["instruction"] += dataset["question"][:]
    new_dataset["answer"] += dataset["answer"][:]
    new_dataset["answer"] = [a.replace('#### ', '') for a in new_dataset["answer"]]
    new_dataset["instruction"] = [i.strip() for i in new_dataset["instruction"]]
    if add_on:
        add_on = "Solve the following math problem step-by-step.\nSimplify your answer as much as possible. Present your final answer as \\boxed{Your Answer}.\n"
        new_dataset["instruction"] = [add_on + i.strip() for i in new_dataset["instruction"]]

    return Dataset.from_dict(new_dataset)


def load_asdiv(name="MU-NLPC/Calc-asdiv_a", add_on=False):
    new_dataset = dict()
    dataset = load_dataset(name)["test"]
    new_dataset["instruction"] = dataset["question"][:]
    new_dataset["answer"] = dataset["result"][:]
    new_dataset["instruction"] = [i.strip() for i in new_dataset["instruction"]]
    new_dataset["answer"] = [i.strip() for i in new_dataset["answer"]]
    if add_on:
        add_on = "Solve the following math problem step-by-step.\nSimplify your answer as much as possible. Present your final answer as \\boxed{Your Answer}.\n"
        new_dataset["instruction"] = [add_on + i.strip() for i in new_dataset["instruction"]]
    return Dataset.from_dict(new_dataset)


def load_math(name="lighteval/MATH", split=None, clean=True, add_on=False):
    new_dataset = dict()
    if not split:
        split_names = ["algebra", "counting & probability", "intermediate algebra", "number theory", "prealgebra",
                       "precalculus"]
    else:
        split_names = [split]
    dataset = load_dataset(name)["test"]
    dataset = dataset.filter(lambda example: example["type"].lower().strip() in split_names)
    dataset = dataset.filter(lambda example: "\\box" in example["solution"])
    new_dataset["instruction"] = dataset["problem"][:]
    new_dataset["answer"] = dataset["solution"][:]
    if clean:
        new_dataset["instruction"] = [ins.strip() for ins in new_dataset["instruction"]]
        new_dataset["answer"] = [extract_boxed(a) for a in new_dataset["answer"]]

    if add_on:
        add_on = "Solve the following math problem step-by-step.\nSimplify your answer as much as possible. Present your final answer as \\boxed{Your Answer}.\n"
        new_dataset["instruction"] = [add_on + i.strip() for i in new_dataset["instruction"]]

    return Dataset.from_dict(new_dataset)


def load_theoqa(name="TIGER-Lab/TheoremQA", add_on=False):
    new_dataset = dict()
    dataset = load_dataset(name, split='test')
    dataset = dataset.filter(lambda example: not example["Picture"])
    new_dataset["instruction"] = dataset["Question"][:]
    new_dataset["answer"] = dataset["Answer"][:]
    new_dataset["instruction"] = [i.strip() for i in new_dataset["instruction"]]
    new_dataset["answer"] = [i.strip() for i in new_dataset["answer"]]

    if add_on:
        add_on = "Solve the following math problem step-by-step.\nSimplify your answer as much as possible. Present your final answer as \\boxed{Your Answer}.\n"
        new_dataset["instruction"] = [add_on + i.strip() for i in new_dataset["instruction"]]

    return Dataset.from_dict(new_dataset)


def load_gsm8k(name="gsm8k", add_on=False, split="train"):
    new_dataset = dict()
    dataset = load_dataset(name, "main", split=split)
    dataset = dataset.filter(lambda example: example["answer"].split('####')[-1].strip())
    new_dataset["instruction"] = dataset["question"][:]
    new_dataset["answer"] = dataset["answer"][:]
    new_dataset["instruction"] = [i.strip() for i in new_dataset["instruction"]]
    new_dataset["answer"] = [i.split('####')[-1].strip() for i in new_dataset["answer"]]
    if add_on:
        add_on = "Question: "
        new_dataset["instruction"] = [add_on + i.strip() + '\nAnswer: ' for i in new_dataset["instruction"]]
    return Dataset.from_dict(new_dataset)


def word_to_number(word):
    word_dict = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
        "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
        "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
        # Add more if needed
    }

    return word_dict.get(word.lower())


def extract_value(input_a, input_b):
    if input_a.count(',') > 4 or ':' in input_a or '.' in input_a or '/' in input_a or '.' in input_b:
        return None
    # Define a dictionary to map the input_b choices to indices
    choice_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}

    # Split the input_a string into a list of choices
    choices = re.split(r'\s*,\s*', input_a)

    # Extract the chosen string based on input_b
    chosen_string = choices[choice_map[input_b]]

    # Find all numerical parts in the chosen string
    numerical_parts = re.findall(r'\d+\.?\d*', chosen_string)

    # Join the numerical parts with a space if there are multiple parts, else return the single part
    result = ' '.join(numerical_parts) if numerical_parts else chosen_string

    return result


def load_mathqa_raw(name="allenai/math_qa", add_on=False, split="train"):
    if split == "train":
        datasets = [load_dataset(name, split="train"), load_dataset(name, split="validation")]
    else:
        datasets = [load_dataset(name, split="test")]

    new_dataset = dict()
    new_dataset["instruction"] = []
    new_dataset["answer"] = []

    for dataset in datasets:
        for i in tqdm(range(len(dataset["Problem"]))):
            answer = extract_value(dataset["options"][i], dataset["correct"][i])
            if answer and ')' not in answer and len(str(answer)) <= 4:
                new_dataset["instruction"].append(dataset["Problem"][i])
                new_dataset["answer"].append(answer)

    with open("/export/atlas-eval/huggingface/datasets/math_qa_" + split + ".json", "w") as f:
        json.dump(new_dataset, f)

    return Dataset.from_dict(new_dataset)


def load_mathqa(add_on=False, split="train"):
    json_file_path = "/export/atlas-eval/huggingface/datasets/math_qa_" + split + ".json"
    with open(json_file_path, "r") as f:
        new_dataset = json.load(f)

    return Dataset.from_dict(new_dataset)


def load_numglue(add_on=False, split="train"):
    new_dataset = dict()
    new_dataset["instruction"] = []
    new_dataset["answer"] = []

    folder = "/export/atlas-eval/huggingface/datasets/num-glue/"
    if split == "train":
        path_lst = [folder + "NumGLUE_train.jsonl", folder + "NumGLUE_dev.jsonl"]
    else:
        path_lst = [folder + "NumGLUE_test.jsonl"]

    for path in path_lst:
        with open(path, 'r') as json_file:
            json_list = list(json_file)

        for json_str in json_list:
            result = json.loads(json_str)
            if result["type"] in ["Type_1", "Type_2", "Type_8"]:
                if '___' not in result["question"]:
                    new_dataset["instruction"].append(str(result["question"]))
                    new_dataset["answer"].append(str(result["answer"]))
            elif result["type"] == "Type_5":
                if result["answer"]["number"]:
                    new_dataset["instruction"].append(str(result["passage"]) + str(result["question"]))
                    new_dataset["answer"].append(str(result["answer"]["number"]))

            else:
                pass
    if add_on:
        add_on = "Question: "
        new_dataset["instruction"] = [add_on + i.strip() + '\nAnswer: ' for i in new_dataset["instruction"]]
    return Dataset.from_dict(new_dataset)


def load_asdiv_raw(name="EleutherAI/asdiv", add_on=False, split="train"):
    dataset = load_dataset(name, split="validation")
    print(dataset["body"][-1])
    print(dataset["question"][-1])
    print(dataset["answer"][-1])
    return dataset


def load_arc_c(name="allenai/ai2_arc", split="train"):
    """
    allenai/ai2_arc
    """
    # 'tracking_shuffled_objects_five_objects', 'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_three_objects'
    # split_names = ['boolean_expressions', 'causal_judgement', 'date_understanding', 'disambiguation_qa', 'dyck_languages', 'formal_fallacies', 'geometric_shapes', 'hyperbaton', 'logical_deduction_five_objects', 'logical_deduction_seven_objects', 'logical_deduction_three_objects', 'movie_recommendation', 'multistep_arithmetic_two', 'navigate', 'object_counting', 'penguins_in_a_table', 'reasoning_about_colored_objects', 'ruin_names', 'salient_translation_error_detection', 'snarks', 'sports_understanding', 'temporal_sequences', 'web_of_lies', 'word_sorting']
    if split == "train":
        dataset = load_dataset(name, 'ARC-Challenge', split="train+validation")
    else:
        dataset = load_dataset(name, 'ARC-Challenge', split="test")

    new_dataset = dict()
    new_dataset["instruction"] = []
    new_dataset["answer"] = []
    new_dataset["answer_choice"] = []
    new_dataset["answer_letter"] = []
    new_dataset["answer_choice_filter"] = []

    map_letter_inx = {k: v for v, k in enumerate(string.ascii_uppercase)}
    map_inx_letter = {str(v + 1): k for v, k in enumerate(string.ascii_uppercase)}

    for i in range(len(dataset["question"])):
        choices = dataset["choices"][i]['text']
        labels = dataset["choices"][i]['label']

        ans = dataset["answerKey"][i]
        if ans in map_inx_letter:
            ans = map_inx_letter[ans]
            labels = [map_inx_letter[l] for l in labels]
        new_dataset["answer_letter"].append('(' + ans + ')')
        choices_dict = {label: choice for label, choice in zip(labels, choices)}
        choice_string = '; '.join(['(' + labels[i] + ') ' + choices[i] for i in range(len(choices))])
        ins = dataset["question"][i].strip() + '\nChoose from the following options: ' + choice_string
        new_dataset["instruction"].append(ins)
        new_dataset["answer_choice"].append(choices_dict[ans])
        index = map_letter_inx[ans]
        ans_filter = filter_answer(choices)[index].replace('.', '')
        new_dataset["answer_choice_filter"].append(ans_filter)
        ans = '(' + ans + ') ' + choices_dict[ans]
        new_dataset["answer"].append(ans)

    return Dataset.from_dict(new_dataset)


def string_to_dict(input_str):
    keys = re.findall('\((.*?)\)', input_str)
    keys = ['(' + k + ')' for k in keys]
    values = re.split('\(.*?\)\s*', input_str)[1:]
    values = [v.strip() for v in values]
    return dict(zip(keys, values)), values


def ans_to_index(ans):
    res = dict()
    for i in range(10):
        key = '(' + string.ascii_uppercase[i] + ')'
        res[key] = i
    return res[ans]


def load_bb_hard(name="lukaemon/bbh", split="tracking", number=3):
    if split == "tracking":
        if number == 3:  # 250
            dataset = load_dataset(name, "tracking_shuffled_objects_three_objects", split="test")
        elif number == 5:  # 250
            dataset = load_dataset(name, "tracking_shuffled_objects_five_objects", split="test")
        else:  # 250
            dataset = load_dataset(name, "tracking_shuffled_objects_seven_objects", split="test")
    else:
        if number == 3:  # 250
            dataset = load_dataset(name, "logical_deduction_three_objects", split="test")
        elif number == 5:  # 250
            dataset = load_dataset(name, "logical_deduction_five_objects", split="test")
        else:  # 250
            dataset = load_dataset(name, "logical_deduction_seven_objects", split="test")
    new_dataset = dict()
    new_dataset["instruction"] = []
    new_dataset["answer_letter"] = []
    new_dataset["answer_choice"] = []
    new_dataset["answer_choice_filter"] = []
    for i in range(len(dataset["input"])):
        input = dataset["input"][i]
        input_str = input.split('Options:')[-1]
        answer_dict, values = string_to_dict(input_str)
        answer = dataset["target"][i]
        new_dataset["instruction"].append(input)
        index = ans_to_index(answer)
        ans_filter = filter_answer(values)[index].replace('.', '')
        new_dataset["answer_choice_filter"].append(ans_filter)
        new_dataset["answer_letter"].append(answer)
        new_dataset["answer_choice"].append(answer_dict[answer])

    return Dataset.from_dict(new_dataset)


def load_eurus_eval(dataset_name="gsmplus_test", split="train"):
    path = "/export/atlas-eval/Eurus/eval/Math/subset/data/" + dataset_name + '.json'
    new_dataset = defaultdict(list)
    with open(path, "r") as f:
        dataset = json.load(f)
    if dataset_name =="asdiv":
        for instance in dataset["Instances"]:
            if instance["split"] == split:
                new_dataset["instruction"].append(instance["Input"])
                new_dataset["answer"].append(instance['Output Answer'][0])
    elif dataset_name =="SVAMP":
        for instance in dataset:
            new_dataset["instruction"].append(instance["Body"]+'\n'+instance["Question"])
            new_dataset["answer"].append(instance['Answer'])
    elif dataset_name =="gsmplus_test":
        for instance in dataset:
            instance = dataset[instance]
            for subproblem_name in instance["perturbation_questions"]:
                p = instance["perturbation_questions"][subproblem_name]
                new_dataset["instruction"].append(p["question"])
                new_dataset["answer"].append(p['answer'])
    return Dataset.from_dict(new_dataset)