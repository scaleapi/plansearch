from datasets import Dataset, DatasetDict

import argparse
import random

from search.one_prompt_models import BasicPromptModel
from search.dataset_utils import parse_dataset
from coderm.prompts import py_prompt
from search.reward_model_utils import RewardModel


def main(args: argparse.Namespace):
    model = BasicPromptModel("model_configs/gpt-4o-mini.json", num_shot=1)

    all_data = {}
    if args.split == "both":
        splits = ["train", "test"]
    elif args.split == "train":
        splits = ["train"]
    else:
        splits = ["test"]

    for split in splits:
        new_data = {"chosen": [], "rejected": []}
        problems = parse_dataset(args.dataset, split)
        
        if args.reward_model is None:
            ordered_sols_per_problem = []
            ordered_fails_per_problem = []

            for i, problem in enumerate(problems):
                assert isinstance(problem.solutions, list)
                assert isinstance(problem.fail_codes, list)

                ordered_sols_per_problem.append(problem.solutions)
                ordered_fails_per_problem.append(problem.fail_codes)
        else:
            rm = RewardModel(args.reward_model,)
            problem_groups = ["solutions", "fail_codes"]

            scores_codes_per_problem = {}

            for problem_group in problem_groups:
                problem_idxs = []
                scores = []
                questions = []
                codes = []

                for i, problem in enumerate(problems):
                    assert isinstance(problem.solutions, list)
                    assert isinstance(problem.fail_codes, list)
                    if problem_group == "solutions":
                        code_group = problem.solutions
                    else:
                        code_group = problem.fail_codes

                    codes.extend(code_group)
                    questions.extend([problem.problem_str] * len(code_group))
                    problem_idxs.extend([i] * len(code_group))

                scores = rm.get_scores(questions, codes)

                scores_codes_per_problem[problem_group] = [[] for _ in range(len(problems))]
                for orig_idx, score, code in zip(problem_idxs, scores, codes):
                    scores_codes_per_problem[problem_group][orig_idx].append((score, code))

                for i in range(len(problems)):
                    scores_codes_per_problem[problem_group][i] = sorted(scores_codes_per_problem[problem_group][i], reverse=True)

            ordered_sols_per_problem = [[code for _, code in score_codes] for score_codes in scores_codes_per_problem["solutions"]]
            ordered_fails_per_problem = [[code for _, code in score_codes] for score_codes in scores_codes_per_problem["fail_codes"]]


        assert len(problems) == len(ordered_sols_per_problem) == len(ordered_fails_per_problem)

        for problem, sols, fails in zip(problems, ordered_sols_per_problem, ordered_fails_per_problem):
            if len(sols) < args.min_len or len(fails) < args.min_len:
                continue
            
            prompt = model.format_problem_to_prompt(problem)
            for i, (sol_code, fail_code) in enumerate(zip(sols, fails)):
                if i == args.max_len:
                    break

                sol_code_str = f"```python\n{sol_code}\n```"
                fail_code_str = f"```python\n{fail_code}\n```"
                assert isinstance(prompt, list)
                new_data["chosen"].append(prompt + [{"role": "assistant", "content": sol_code_str}])
                new_data["rejected"].append(prompt + [{"role": "assistant", "content": fail_code_str}])

        all_data[split] = Dataset.from_dict(new_data)

    if len(splits) == 1:
        dd = all_data[splits[0]].train_test_split(args.test_prop)
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
    parser.add_argument(
        "--reward-model",
        type=str,
        default=None,
        help="Reward model to use"
    )
    args = parser.parse_args()

    main(args)


