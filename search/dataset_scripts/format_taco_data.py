import json
from datasets import load_dataset, Dataset, DatasetDict
import argparse


def map_taco_to_diff(difficulties: list[str]):
    DIFF_TO_DIFF = {"EASY": "easy", "MEDIUM": "medium", "MEDIUM_HARD": "medium_hard", "HARD": "hard", "VERY_HARD": "very_hard", "UNKNOWN_DIFFICULTY": "unknown"}
    return [DIFF_TO_DIFF[difficulty] for difficulty in difficulties]

def main(args: argparse.Namespace):
    d4 = load_dataset(args.dataset)
    dd = {}

    splits = []
    if args.split == "both":
        splits = ["train", "test"]
    elif args.split == "train":
        splits = ["train"]
    else:
        splits = ["test"]
 
    for split in splits:
        all_tests = []
        for filter_input_output in d4[split]["input_output"]:
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

        new_dataset = Dataset.from_dict({"question": d4[split]["question"], 
                                        "starter_code": d4[split]["starter_code"],
                                        "input_output": all_tests,
                                        "difficulty": map_taco_to_diff(d4[split]["difficulty"]),
                                        "solutions": d4[split]["solutions"],

                                        "raw_tags": d4[split]["raw_tags"],
                                        "source": d4[split]["source"],
                                        "date": d4[split]["date"],
                                        "tags": d4[split]["tags"],
                                        "skill_types": d4[split]["skill_types"],
                                        "time_limit": d4[split]["time_limit"],
                                        "memory_limit": d4[split]["memory_limit"],
                                        "Expected Auxiliary Space": d4[split]["Expected Auxiliary Space"],
                                        "Expected Time Complexity": d4[split]["Expected Time Complexity"],
                                        })
        dd[split] = new_dataset

    dd = DatasetDict(dd)
    dd.push_to_hub(args.output, private=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Path to the input Huggingface dataset')
    parser.add_argument('--output', type=str, required=True, help='Path to the output Huggingface dataset')
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test", "both"],
        required=True,
        help="Split of the dataset to evaluate/generate from"
    )
    args = parser.parse_args()

    main(args)
