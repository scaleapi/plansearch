from datasets import Dataset, DatasetDict
from search.basic_prompting import SimplePromptModel
from search.dataset_utils import parse_dataset
import argparse

def main(args: argparse.Namespace):
    model = SimplePromptModel("model_configs/gpt-4o-mini.json", num_shot=1)

    all_data = {}
    splits = []
    if args.split == "both":
        splits = ["train", "test"]
    elif args.split == "train":
        splits = ["train"]
    else:
        splits = ["test"]

    for split in splits:
        new_data = {"chosen": [], "rejected": []}
        problems = parse_dataset(args.dataset, split)
        
        for problem in problems:
            if len(problem.solutions) < args.min_len or len(problem.fail_codes) < args.min_len:
                continue
            
            prompt = model.format_problem_to_prompt(problem)
            for i, (sol_code, fail_code) in enumerate(zip(problem.solutions, problem.fail_codes)):
                if i == args.max_len:
                    break
                sol_code_str = f"```python\n{sol_code}\n```"
                fail_code_str = f"```python\n{fail_code}\n```"
                new_data["chosen"].append(prompt + [{"role": "assistant", "content": sol_code_str}])
                new_data["rejected"].append(prompt + [{"role": "assistant", "content": fail_code_str}])

        all_data[split] = Dataset.from_dict(new_data)

    if len(splits) == 1:
        dd = all_data[split].train_test_split(args.test_prop)
    else:
        dd = DatasetDict(all_data)

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
    parser.add_argument(
        "--min-len",
        type=int,
        default=1,
        help="Min number of both solutions and fails to include in dataset"
    )
    parser.add_argument(
        "--max-len",
        type=int,
        required=True,
        help="Max number of solutions and fails to include in the dataset per problem"
    )
    parser.add_argument(
        "--test-prop",
        type=float,
        default=0.01,
        help="Proportion of train samples to split into test. (Only applicable if split is not both)"
    )
    args = parser.parse_args()

    main(args)


