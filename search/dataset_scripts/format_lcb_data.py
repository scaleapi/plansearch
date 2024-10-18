from datasets import load_dataset, Dataset, DatasetDict
import argparse

def main(args: argparse.Namespace):
    data = load_dataset(args.dataset)

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
    dd.push_to_hub(args.output, private=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Path to the input Huggingface dataset')
    parser.add_argument('--output', type=str, required=True, help='Path to the output Huggingface dataset')
    args = parser.parse_args()

    main(args)
