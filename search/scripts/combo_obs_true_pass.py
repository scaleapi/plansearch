import json
import os
import datasets
LOG_DIRECTORY = os.path.join("logs",
                                "combo_observation_07-25T08:24:33"
                             )
ORIG_NUM_PROBLEMS = 226

def get_observation_files(log_directory, iter_num: int):
    return [f for f in os.listdir(log_directory) if f.startswith(f"observation_{iter_num}")]
def get_code_files(log_directory):
    return [f for f in os.listdir(log_directory) if f.startswith("codes")]

first_observation_files = get_observation_files(LOG_DIRECTORY, 0)
curr_iter = 0
while len(get_observation_files(LOG_DIRECTORY, curr_iter)):
    final_observation_files = get_observation_files(LOG_DIRECTORY, curr_iter)
    curr_iter += 1

code_files = get_code_files(LOG_DIRECTORY)

num_problems = 0

for file in first_observation_files:
    with open(os.path.join(LOG_DIRECTORY, file), "r") as f:
        obs = json.load(f)
    num_problems += len(obs)

num_codes = 0
for file in code_files:
    with open(os.path.join(LOG_DIRECTORY, file), "r") as f:
        obs = json.load(f)
    num_codes += len(obs)

print(f"RATIO: {num_codes / num_problems}")

assert num_problems % ORIG_NUM_PROBLEMS == 0
n_completions = num_problems // ORIG_NUM_PROBLEMS
print("n_completions:", n_completions)
codes = []
for file in code_files:
    with open(os.path.join(LOG_DIRECTORY, file), "r") as f:
        code_results = json.load(f)
    for code_result in code_results:
        codes.append((code_result["problem_str"], code_result["parsed_codes"]))

lite_ds = datasets.load_dataset("codegenning/livecodebench_lite_v2")
from base_classes import Problem, Test


problems = []
problem_hashes = {}
for row in lite_ds["test"]:
    problems.append(Problem.from_coderm_item(row["question"], row["starter_code"], json.loads(row["public_input_output"]), json.loads(row["input_output"])))
    problem_hashes[row["question"]] = {"problem": problems[-1], "codes": []}
for code in codes:
    assert code[0] in problem_hashes
    problem_hashes[code[0]]["codes"].append(code[1])

lens = []
codes_to_run = []
tests_to_run = []
expanded_to_orig_idxs = []

for i, problem_dict in enumerate(problem_hashes.values()):
    codes_to_run.extend(problem_dict["codes"])
    tests_to_run.extend([problem_dict["problem"].private_tests] * len(problem_dict["codes"]))
    expanded_to_orig_idxs.extend([i] * len(problem_dict["codes"]))

timeouts = [90] * len(tests_to_run)

from exec_utils import run_tests_per_code
results = run_tests_per_code(codes_to_run, tests_to_run, timeouts)
write_data = [{"results": []} for _ in range(ORIG_NUM_PROBLEMS)]
for orig_idx, (status, _) in zip(expanded_to_orig_idxs, results):
    write_data[orig_idx]["results"].append({"passing": status})
    
with open(os.path.join(LOG_DIRECTORY, "results_per_code.json"), "w") as f:
    json.dump(write_data, f)
