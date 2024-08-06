from datasets import load_dataset, Dataset, DatasetDict
import json

d3 = load_dataset("codegenning/taco_cleaned_exec_filtered")
BAD_INDS = [161, 225, 236, 252, 332, 337, 645, 663, 762, 782, 929, 1030, 1227, 1253, 1296, 1572, 1625, 1832, 1845, 1867, 1928, 2352, 2531, 2749, 3076, 3314, 3327, 3473, 3524, 3575, 3691, 3708, 3728, 3732, 3816, 3910, 3944, 4026, 4132, 4274, 4366, 4401, 4746, 4834, 4974, 5004, 5135, 5484, 5821, 6067, 6265, 6376, 6484, 6509, 6570, 6638, 7146, 7161, 2682, 3616, 4924, 2399]
good_inds = [i for i in range(len(d3["train"])) if i not in BAD_INDS]

filtered = d3["train"].select(good_inds)

new_dd = filtered.train_test_split(0.01)
new_dd.push_to_hub("codegenning/taco_cleaner", private=True)

# To find BAD_INDS, I manually went through and combed through flagged problems based on a heuristic:
def flag_weird(i: int, s: str) -> bool:
    MIN_LEN = 200
    if len(s) <= MIN_LEN:
        return True
    if "<image>" in s.lower() or "[image]" in s.lower():
        return True
    if s.strip().startswith("<image>\n\nInput"):
        return True
    if s.strip().startswith("-----Input-----") or s.strip().startswith("Example") or s.strip().startswith("Input"):
        return True
    if s.strip().endswith("Examples"):
        return True
    return False
    
# print(len([(i, be) for i, be in enumerate(d3["train"]["question"]) if flag_weird(i, be)]))
# [(i, be) for i, be in enumerate(d3["train"]["question"]) if flag_weird(i, be)]
