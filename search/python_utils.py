import os
from pathlib import Path
from typing import Any, Optional, Union, TypeVar
import json
import random
import sys


def random_print(print_str: str, p: float = 1e-3):
    if random.random() < p:
        print(print_str)

T = TypeVar('T')
def chunk(lst: list[T], n: int):
    """Yield successive n-sized chunks from lst. From StackOverflow."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def safe_equals(x1: Any, x2: Any) -> bool:
    try:
        return bool(x1 == x2)
    except:
        return False

def safe_repr(arg: Any):
    if safe_equals(arg, float("inf")):
        return "float('inf')"
    return repr(arg)

def safe_iter(potentially_iter: Any):
    if isinstance(potentially_iter, str):
        return (potentially_iter,)
    if not hasattr(potentially_iter, "__iter__"):
        return (potentially_iter,)
    return potentially_iter

def wrap_list(potentially_iter: Any) -> list:
    if isinstance(potentially_iter, str):
        return [potentially_iter,]
    if not hasattr(potentially_iter, "__iter__"):
        return list(potentially_iter)
    return list(potentially_iter)

def log_to_dir(base_dir: str, file_name_to_data: dict[str, Union[str, dict]]):
    if base_dir is None:
        return
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    for file_name, data in file_name_to_data.items():
        with open(os.path.join(base_dir, file_name), "w") as f:
            if file_name.endswith(".json"):
                json.dump(data, f, indent=2)
            else:
                f.write(data)

# From CodeRM
def autodetect_dtype_str() -> str:
    import torch
    if torch.cuda.is_bf16_supported():
        return "bfloat16"
    else:
        return "auto"

if __name__ == "__main__":
    pass
