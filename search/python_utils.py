import os
from pathlib import Path
from typing import Any, Optional, Union, TypeVar, List, Callable
import json
import random
import sys
from enum import Enum


T = TypeVar('T')
U = TypeVar('U')

RNestListT = Union[T, list["RNestListT"]]
RNestListU = Union[U, list["RNestListU"]]
LiRNestListT = list[RNestListT]
LiRNestListU = list[RNestListU]


def nested_list_len(li: LiRNestListT) -> int:
    assert isinstance(li, list)
    cnt = 0
    for el in li:
        if isinstance(el, list):
            cnt += nested_list_len(el)
        else:
            cnt += 1
    return cnt

class Action(Enum):
    UP = "up"
    DOWN = "down"
    PUSH = "push"

def merge_nested_lists(li1: RNestListT, li2: RNestListT) -> RNestListU:
    assert isinstance(li1, list) == isinstance(li2, list)
    if not isinstance(li1, list):
        return (li1, li2)

    output_list = []
    assert isinstance(li1, list) and (len(li1) == len(li2))
    for sub_li1, sub_li2 in zip(li1, li2):
        output_list.append(merge_nested_lists(sub_li1, sub_li2))
    return output_list

def map_nary_fn_on_nested_list(fn, *args: *tuple[RNestListT[T]]) -> RNestListU[U]:
    assert len(args) > 0
    # Checking types of args
    for li in args:
        if isinstance(args[0], list):
            assert isinstance(li, list)
            assert len(args[0]) == len(li)
        else:
            assert not isinstance(li, list)

    if not isinstance(args[0], list):
        return fn(*args)

    output_list = []
    for sub_lists in zip(*args):
        output_list.append(map_nary_fn_on_nested_list(fn, *sub_lists))
    return output_list

def index_nested_list(li: LiRNestListT, items: list[T], actions: list[Action]) -> None:
    for i, el in enumerate(li):
        if isinstance(el, list):
            actions.append(Action.UP)
            index_nested_list(el, items, actions)
            actions.append(Action.DOWN)
        else:
            actions.append(Action.PUSH)
            items.append(li[i])

def _format_from_actions(outputs: list[T], actions: list[Action]) -> LiRNestListT:
    formatted_output: LiRNestListT = []

    # push on right, pop on right
    stack: list[LiRNestListT] = []
    stack.append(formatted_output)

    curr_output_idx = 0
    for action in actions:
        if action is Action.PUSH:
            assert curr_output_idx < len(outputs)
            stack[-1].append(outputs[curr_output_idx])
            curr_output_idx += 1
        elif action is Action.UP:
            stack.append([])
        else:
            assert action is Action.DOWN
            top_of_stack = stack.pop()
            stack[-1].append(top_of_stack)

    assert len(stack) == 1 and stack[-1] == formatted_output
    assert curr_output_idx == len(outputs)

    return formatted_output

def batch_map_on_nested_list(li: LiRNestListT, fn: Callable[[list[T]], list[U]]) -> LiRNestListU:
    assert isinstance(li, list)
    items = []
    actions = []

    index_nested_list(li, items, actions)
    assert nested_list_len(li) == sum(action is Action.PUSH for action in actions)

    outputs = fn(items)
    formatted_outputs = _format_from_actions(outputs, actions)
    assert isinstance(formatted_outputs, list)

    return formatted_outputs


def stringify(x: Any) -> str:
    try:
        return json.dumps(x)
    except:
        return repr(x)

def unstringify(s: str) -> Any:
    try:
        return json.loads(s)
    except:
        return eval(s)

def fn_arg_join(args: list[Any]) -> str:
    return ", ".join([repr(a) for a in args])

def random_print(print_str: str, p: float = 1e-3):
    if random.random() < p:
        print(print_str)

def convert_to_cmd_arg(attr: str) -> str:
    return attr.replace('_', '-')

def str_to_bool(value: Union[str, bool]) -> bool:
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

def chunk(lst: list[T], n: Optional[int]):
    """Yield successive n-sized chunks from lst. From StackOverflow."""
    if n is None:
        yield lst
    else:
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

def log_to_dir(base_dir: Optional[str], file_name_to_data: dict[str, Union[str, Any]]):
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
    lol = [[3, ["2"]], ["a"], "0", [], ["b", [3, [[[5]]]], ["1", 2, 3]]]
    lol1 = [[1, [2]], ["a"], "1", [], ["b", [3, [[[5]]]], ["1", 2, 3]]]
    print(lol)
    # print(batch_map_on_nested_list(lol, lambda x: [str(s) for s in x]))
    print(merge_nested_lists(lol, lol1))
    print(map_nary_fn_on_nested_list(lambda x1, x2: str(x1) + "|" + str(x2), lol, lol1))
