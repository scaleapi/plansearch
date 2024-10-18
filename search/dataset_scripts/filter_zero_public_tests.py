import json
from datasets import load_dataset, Dataset, DatasetDict
import argparse


def map_taco_to_diff(difficulties: list[str]):
    DIFF_TO_DIFF = {"EASY": "easy", "MEDIUM": "medium", "MEDIUM_HARD": "medium_hard", "HARD": "hard", "VERY_HARD": "very_hard", "UNKNOWN_DIFFICULTY": "unknown"}
    return [DIFF_TO_DIFF[difficulty] for difficulty in difficulties]

def main(args: argparse.Namespace):
    data = load_dataset(args.dataset)
    dd = {}

    splits = []
    if args.split == "both":
        splits = ["train", "test"]
    elif args.split == "train":
        splits = ["train"]
    else:
        splits = ["test"]
 
    for split in splits:
        good_public_test_idxs = [i for i, pio_str in enumerate(data[split]["public_input_output"]) if len(json.loads(pio_str)["inputs"]) != 0]
        new_dataset = data[split].select(good_public_test_idxs)
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
