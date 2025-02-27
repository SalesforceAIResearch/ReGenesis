from collections import defaultdict
import random
import sys, json, os
from fastchat.conversation import get_conv_template, conv_templates
from tqdm import tqdm
import re
import tiktoken
from collections import Counter
cl100k_base = tiktoken.get_encoding("cl100k_base")

def generate_response_vllm(llm, sampling_params, prompts):
    outputs = llm.generate(prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]

def check_results_vllm(solutions, truth, llm, sampling_params, template="llama-3", question=None):
    prompts = []
    for i in range(len(solutions)):
        conv = get_conv_template(template)
        conv.set_system_message("You are a helpful, respectful and honest assistant.")

        solution = solutions[i]
        if not question:
            prompt = f"The solution given by the language model is: {solution}.\nThe truth is: {truth}.\nIs the solution correct? Yes or No."
        else:
            prompt = f"Given the question: {question}, the solution given by the language model is: {solution}.\nThe correct answer is: {truth}.\nDo you think the solution is correct (same with the correct answer)? Yes or No."
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompts.append(conv.get_prompt())
    correctness_lst = generate_response_vllm(llm, sampling_params, prompts)
    return [True if 'yes' in c[:20].lower() else False for c in correctness_lst]


def extract_last_number(input_string):
    # Find all numerical values in the string
    numerical_values = re.findall(r"[-+]?\d*\.\d+|\d+", input_string)
    rewrite = ""
    if numerical_values:
        rewrite = "Answer:\\boxed{" + str(numerical_values[-1]) + '}'
    return (numerical_values if numerical_values else None, rewrite)

def _last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    left_brace_idx = None
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
            if left_brace_idx is None:
                left_brace_idx = i
        elif string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break

        i += 1

    if left_brace_idx is None or right_brace_idx is None:
        return None

    return string[left_brace_idx + 1: right_brace_idx].strip()

def match_answer(response):
    is_matched = False
    # Find boxed
    ans_boxed = _last_boxed_only_string(response)
    if ans_boxed:
        is_matched = True
        response = ans_boxed

    # Grade
    response = response.replace('$', '')
    response = response.replace(',', '')
    response = response.replace('-', ' - ')
    try:
        responses = [int(num) if num.isdigit() else float(num) for num in re.findall(r'-?\d+\.?\d*', response)]
        res = []
        for r in [responses[0], responses[-1]]:
            r = str(r)
            res.append(r)
            r = r.replace('.00', '').replace('.00%', '').replace('.0%', '').replace('.0', '').replace('%', '')
            res.append(r)
    except:
        res = []
    return is_matched, res, ans_boxed

def check_results_em(solutions, truth):
    correctness = []
    if '.' == truth[0]:
        truth = '0' + truth
    truth = truth.replace('.00', '').replace('.00%', '').replace('.0%', '').replace('.0', '').replace(',', '').replace('%', '').strip()
    for i in range(len(solutions)):
        solution = solutions[i].strip()
        answers = None
        _, answers, ans_boxed = match_answer(solution)
        correctness.append(False)
        if answers:
            if truth[0].isdigit():
                for answer in answers:
                    if answer == truth:
                        correctness[-1] = True
                        break
            else:
                if ans_boxed and truth.lower() in ans_boxed.lower():
                    correctness[-1] = True
    return correctness, []

def _filter_do_not_know(solution):
    if "don't know" in solution:
        return False
    
    if "impossible" in solution:
        return False

    return True

def find_json_files(string_names, directory, module=True, redone=True, reverse=True):
    # List to store filenames with "aaa" in their names
    matching_files = []
    string_names_redone = [string_names[0], 'redone'] if redone else None
    string_names_reverse = [string_names[0], 'reverse'] if reverse else None

    # Loop through all files in the specified directory
    for filename in os.listdir(directory):
        # Check if the file is a JSON file and contains "aaa" in its name
        if module and filename.endswith('.jsonl') and all([string_name in filename for string_name in string_names]):
            if "redone" not in filename and "reverse" not in filename:
                matching_files.append(folder + '/' + filename)
        if string_names_redone and filename.endswith('.jsonl') and all([string_name in filename for string_name in string_names_redone]):
            matching_files.append(folder + '/' + filename)
        if string_names_reverse and filename.endswith('.jsonl') and all([string_name in filename for string_name in string_names_reverse]):
            matching_files.append(folder + '/' + filename)

    return matching_files

def find_step(lst):
    idx = None
    for i, l in enumerate(lst):
        if '1.' in l:
            if '2.' in ''.join(lst[i:]):
                idx = i
    return idx

if __name__ == "__main__":
    folder = "./"  # folder we save the previous round reasoning path
    max_number = 5  # number of paths for each question you want to keep
    file_names = find_json_files(string_names="gsm8k", directory=folder)

    # load all the raw data
    with open(file_names[0], "r") as f:
        json_list = list(f)
    for i in range(1, len(file_names)):
        with open(file_names[i], "r") as f:
            json_list += list(f)

    instructions_reasons = defaultdict(list)
    instructions_solutions = defaultdict(list)
    instructions_truth = dict()
    instructions_save = defaultdict(int)
    instructions_all = set()

    # load raw data to dict
    for json_str in json_list:
        try:
            result = json.loads(json_str)
            truth = result['answer']
            solution = result['reasoning']
            correct = result['correctness']
            instructions_all.add(result['instruction'])
            if correct and "boxed" in solution:  # len(result['module']) < 1000 and
                instructions_truth[result['instruction']] = truth
                instructions_solutions[result['instruction']].append(solution)
                instructions_reasons[result['instruction']].append((result['module'], result['reasoning']))
        except:
            pass
    print('After LLM check the correctness: ')
    print('>>> We will use ', len(instructions_truth), ' out of ', len(instructions_all))


    res = []
    keys = list(instructions_truth.keys())
    module_stats = defaultdict(int)
    length_stats = defaultdict(int)
    total, have = 0, 0
    for i in tqdm(range(len(keys))):
        ins = keys[i]
        solutions = instructions_solutions[ins]
        reasons = [element[1] for element in instructions_reasons[ins]]
        modules = [element[0] for element in instructions_reasons[ins]]
        truth = instructions_truth[ins]
        correctness, solutions_rewrite = check_results_em(solutions, truth)
        res_one_ins, raw_one_ins = [], []
        total += 1
        if sum(correctness) > 0:
            have += 1
        for idx, c in enumerate(correctness):
            try:
                if c and len(cl100k_base.encode(solutions[idx])) < 800:
                    s = solutions[idx].split('\n\n')
                    s = [re.sub(r'\s+', ' ', _s.strip()) for _s in s if _s.strip()]
                    s = [_s for _s in s if _s.strip()]
                    if s:
                        idx_s = find_step(s)
                        if idx_s:
                            write = "Let's think step by step.\n\n" + '\n\n'.join(s[idx_s:])
                        else:
                            write = None
                        if write:
                            res_one_ins.append({"instruction": ins,
                                                "answer":write,
                                                "truth": truth})
            except:
                pass

        if len(res_one_ins) > max_number:
            res_one_ins = random.sample(res_one_ins, max_number)
        res += res_one_ins
                
        instructions_save[ins] += len(res_one_ins)

    with open("save-path-to.json", 'w') as f:
        json.dump(res, f)