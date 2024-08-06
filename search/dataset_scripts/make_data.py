from IPython import embed
from datasets import load_dataset, Dataset, DatasetDict
import json

"""
data = load_dataset("codegenning/taco_cleaned_exec_filtered")

def map_taco_to_diff(difficulties: list[str]):
    DIFF_TO_DIFF = {"EASY": "easy", "MEDIUM": "medium", "MEDIUM_HARD": "medium_hard", "HARD": "hard", "VERY_HARD": "very_hard", "UNKNOWN_DIFFICULTY": "unknown"}
    return [DIFF_TO_DIFF[difficulty] for difficulty in difficulties]

import json

all_tests = []
for filter_input_output in data["train"]["input_output"]:
    row = json.loads(filter_input_output)
    new_outputs = []
    if row.get("fn_name", None) is None:
        for t in row["outputs"]:
            if isinstance(t, str):
                new_outputs.append(t)
            else:
                assert isinstance(t, list)
                new_outputs.append('\n'.join(t) + '\n')
    else:
        new_outputs = row["outputs"]
    row["outputs"] = new_outputs
    all_tests.append(json.dumps(row))

new_dataset = Dataset.from_dict({"question": data["train"]["question"], 
                                 "starter_code": data["train"]["starter_code"],
                                 "input_output": all_tests,
                                 "difficulty": map_taco_to_diff(data["train"]["difficulty"]),
                                 "solutions": data["train"]["solutions"],

                                 "raw_tags": data["train"]["raw_tags"],
                                 "source": data["train"]["source"],
                                 "date": data["train"]["date"],
                                 "tags": data["train"]["tags"],
                                 "skill_types": data["train"]["skill_types"],
                                 "time_limit": data["train"]["time_limit"],
                                 "memory_limit": data["train"]["memory_limit"],
                                 "Expected Auxiliary Space": data["train"]["Expected Auxiliary Space"],
                                 "Expected Time Complexity": data["train"]["Expected Time Complexity"],
                                })

dd = DatasetDict({"train": new_dataset})
dd.push_to_hub("codegenning/F_taco_execclean")
"""

data = load_dataset("codegenning/livecodebench_lite_v2_lite35")

new_dataset = Dataset.from_dict({"question": data["test"]["question"], 
                                 "starter_code": data["test"]["starter_code"],
                                 "input_output": data["test"]["input_output"],
                                 "public_input_output": data["test"]["public_input_output"],
                                 "difficulty": data["test"]["difficulty"],

                                 "source": data["test"]["source"],
                                 "date": data["test"]["date"],
                                 "id": data["test"]["id"],
                                })

dd = DatasetDict({"test": new_dataset})
dd.push_to_hub("codegenning/F_livecodebench_lite_v2_lite35", private=True)
"""
# data = load_dataset("codegenning/taco_with_gpt-4o-mini_gennedsols")
data = load_dataset("codegenning/taco-rl_meta-llama-Meta-Llama-3.1-8B-Instruct_sols_M")

new_dataset = Dataset.from_dict({"question": data["train"]["problem_str"], 
                                 "starter_code": data["train"]["starter_code"],
                                 "input_output": data["train"]["tests"],
                                 "solutions": data["train"]["correct_solutions"],
                                 "fail_codes": data["train"]["fail_solutions"],
                                })

dd = DatasetDict({"train": new_dataset})
dd.push_to_hub("codegenning/F_taco-rl_Llama-3.1-8B-I_sols")
"""

"""
data = load_dataset("codegenning/taco-rl")
new_data = {}

def filter_prompt(prompt: str, delimiter: str = '\"""') -> str:
    return prompt.split(delimiter)[1].split(delimiter)[0].strip()

for split in ["train", "test"]:
    all_tests = []
    for filter_input_output in data[split]["input_output"]:
        row = json.loads(filter_input_output)
        new_outputs = []
        if row.get("fn_name", None) is None:
            for t in row["outputs"]:
                if isinstance(t, str):
                    new_outputs.append(t)
                else:
                    assert isinstance(t, list)
                    new_outputs.append('\n'.join(t) + '\n')
        else:
            new_outputs = row["outputs"]
        row["outputs"] = new_outputs
        all_tests.append(json.dumps(row))

    new_data[split] = Dataset.from_dict({"question": [filter_prompt(prompt) for prompt in data[split]["prompt"]], 
                                     "starter_code": data[split]["starter_code"],
                                     "input_output": all_tests,
                                     "solutions": data[split]["solutions"],

                                     "raw_tags": data[split]["raw_tags"],
                                     "source": data[split]["source"],
                                     "date": data[split]["date"],
                                     "tags": data[split]["tags"],
                                     "skill_types": data[split]["skill_types"],
                                     "time_limit": data[split]["time_limit"],
                                     "memory_limit": data[split]["memory_limit"],
                                     "Expected Auxiliary Space": data[split]["Expected Auxiliary Space"],
                                     "Expected Time Complexity": data[split]["Expected Time Complexity"],
                                    })

dd = DatasetDict(new_data)
dd.push_to_hub("codegenning/F_taco_execclean")
"""

