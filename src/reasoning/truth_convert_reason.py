from fastchat.conversation import get_conv_template, conv_templates
from vllm import LLM, SamplingParams
from tqdm import tqdm
import json

def template_construct(template, prompt):
    conv = get_conv_template(template)
    conv.set_system_message("You are a helpful, respectful and honest assistant.")
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()

def generate_response_vllm(llm, sampling_params, prompts):
    outputs = llm.generate(prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]

# STAGE 1

def select_reasoning_modules(task_descriptions, reasoning_modules, llm, sampling_params, template_usage="mistral"):
    """
    Step 1: SELECT relevant reasoning modules for the task.
    """
    prompts = []
    for task_description in task_descriptions:
        prompt = f"Given the task: {task_description}, which of the following reasoning modules are relevant? Do not elaborate on why.\n\n" + "\n".join(reasoning_modules)
        prompts.append(template_construct(template_usage, prompt))
    selected_modules_lst = generate_response_vllm(llm, sampling_params, prompts)
    return selected_modules_lst


def adapt_reasoning_modules(selected_modules_lst, task_example, solution, llm, sampling_params, template_usage="mistral"):
    """
    Step 2: ADAPT the selected reasoning modules to be more specific to the task.
    """
    prompts = []
    for selected_modules in selected_modules_lst:
        prompt = f"Without working out the solution: {solution}, adapt the following reasoning modules to be specific to our task:\n{selected_modules}\n\nOur task:\n{task_example}"
        prompts.append(template_construct(template_usage, prompt))
    adapted_modules_lst = generate_response_vllm(llm, sampling_params, prompts)
    return adapted_modules_lst


def implement_reasoning_structure(adapted_modules_lst, task_description, solution, llm, sampling_params, template_usage="mistral"):
    """
    Step 3: IMPLEMENT the adapted reasoning modules into an actionable reasoning structure.
    """
    prompts = []
    for adapted_modules in adapted_modules_lst:
        prompt = f"Without working out the solution: {solution}, create an actionable and concise reasoning structure step by step for the task using these adapted reasoning modules:\n{adapted_modules}\n\nTask Description:\n{task_description}\n"
        prompts.append(template_construct(template_usage, prompt))
    reasoning_structure = generate_response_vllm(llm, sampling_params, prompts)
    return reasoning_structure


# STAGE 2

def execute_reasoning_structure(reasoning_structure_lst, task_instance, solution, llm, sampling_params, template_usage="mistral"):
    """
    Execute the reasoning structure to solve a specific task instance.
    """
    prompts = []
    solution = "\\boxed{" + solution + "}"
    for reasoning_structure in reasoning_structure_lst:
        prompt = f"Using the following reasoning structure: {reasoning_structure}\n\nTask: {task_instance}\n\nSolve this task step by step (less than 8 steps) based on the reasoning structure, and at last present your final answer: {solution}. You must present the solution in <---box--->."
        prompts.append(template_construct(template_usage, prompt).replace('<---box--->', "\\boxed{Your Answer}"))
    solution = generate_response_vllm(llm, sampling_params, prompts)
    return solution


def check_results(solutions, truth, llm, sampling_params, question=None, template_usage="mistral"):
    prompts = []
    for solution in solutions:
        if not question:
            prompt = f"The solution given by the language model is: {solution}\n\nThe truth is: {truth}.\n\nIs the solution correct? Yes or No."
        else:
            prompt = f"Given the question: {question}, the solution given by the language model is: {solution}\n\nThe correct answer is: {truth}.\n\nDo you think the solution is correct (same with the correct answer)? Yes or No."
        prompts.append(template_construct(template_usage, prompt))
    correctness = generate_response_vllm(llm, sampling_params, prompts)
    return [True if 'yes' in c[:20].lower() else False for c in correctness]


# Example usage
if __name__ == "__main__":
    reasoning_modules = ["How could I devise an experiment to help solve that problem?",
    "Make a list of ideas for solving this problem, and apply them one by one to the problem to see if any progress can be made.",
    "How could I measure progress on this problem?",
    "How can I simplify the problem so that it is easier to solve?",
    "How can I break down this problem into smaller, more manageable parts?",
    "Critical Thinking: This style involves analyzing the problem from different perspectives, questioning assumptions, and evaluating the evidence or information available. It focuses on logical reasoning, evidence-based decision-making, and identifying potential biases or flaws in thinking.",
    "Try creative thinking, generate innovative and out-of-the-box ideas to solve the problem. Explore unconventional solutions, thinking beyond traditional boundaries, and encouraging imagination and originality.",
    "Use systems thinking: Consider the problem as part of a larger system and understanding the interconnectedness of various elements. Focuses on identifying the underlying causes, feedback loops, and interdependencies that influence the problem, and developing holistic solutions that address the system as a whole.",
    "Use Reflective Thinking: Step back from the problem, take the time for introspection and self-reflection. Examine personal biases, assumptions, and mental models that may influence problem-solving, and being open to learning from past experiences to improve future approaches.",
    "What is the core issue or problem that needs to be addressed?",
    "What are the potential obstacles or challenges that might arise in solving this problem?",
    "Are there any relevant data or information that can provide insights into the problem? If yes, what data sources are available, and how can they be analyzed?",
    "How can progress or success in solving the problem be measured or evaluated?",
    "What indicators or metrics can be used?",
    "Is the problem a technical or practical one that requires a specific expertise or skill set? Or is it more of a conceptual or theoretical problem?",
    "Does the problem involve decision-making or planning, where choices need to be made under uncertainty or with competing objectives?",
    "Is the problem an analytical one that requires data analysis, modeling, or optimization techniques?",
    "Is the problem a design challenge that requires creative solutions and innovation?",
    "Does the problem require addressing systemic or structural issues rather than just individual instances?",
    "What kinds of solution typically are produced for this kind of problem specification?",
    "Let's think step by step.",
    "Let's make a step by step plan and implement it with good notation and explanation."
]

    with open("./step3.json", 'r') as f:  # last round's
        dataset = json.load(f)


    sampling_params = SamplingParams(temperature=0.85, top_p=0.9, max_tokens=2048)
    llm = LLM(model="/export/atlas-eval/huggingface/Mistral-7B-Instruct-v0.3")  #, tensor_parallel_size=2)  # tensor_parallel_size=4
    r = [210, 270]
    start_idx = 0
    outfile_all = open('step4.jsonl', 'w')

    for j in tqdm(range(r[0], min(r[1], len(dataset["instruction"])))):
        task_example = dataset["instruction"][j]
        truth = dataset["answer"][j]

        selected_modules = select_reasoning_modules([task_example] * 20, reasoning_modules, llm, sampling_params)

        adapted_modules = adapt_reasoning_modules(selected_modules, task_example, truth, llm, sampling_params)

        reasoning_structure = implement_reasoning_structure(adapted_modules, task_example, truth, llm, sampling_params)

        results_lst = execute_reasoning_structure(reasoning_structure, task_example, dataset["answer"][j], llm, sampling_params)

        correctness = check_results(results_lst, truth, llm, sampling_params, question=task_example)

        for i in range(len(results_lst)):
            entry = dict()
            entry["instruction"] = task_example
            entry["answer"] = truth
            entry["adapted_module"] = adapted_modules[i]
            entry["reasoning"] = results_lst[i]
            entry["reasoning_structure"] = reasoning_structure[i]
            entry["module_id"] = i + start_idx
            entry["module"] = selected_modules[i]
            entry["correctness"] = correctness[i]
            json.dump(entry, outfile_all)
            outfile_all.write('\n')

    outfile_all.close()