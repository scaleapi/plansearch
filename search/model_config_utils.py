import argparse
import json
import os
import datetime

from search.query_clients import CLIENT_TYPE_TO_CLASS
from search.python_utils import str_to_bool, convert_to_cmd_arg


def add_model_config_args(parser: argparse.ArgumentParser, model_config_name: str = "model"):
    model_config_name = convert_to_cmd_arg(model_config_name)
    parser.add_argument(
        f"--{model_config_name}-config-path",
        type=str,
        default=None,
        help=f"{model_config_name}: Model config path to use",
    )
    parser.add_argument(
        f"--{model_config_name}-client-type",
        type=str,
        default=None,
        help=f"{model_config_name}: Client type",
    )
    parser.add_argument(
        f"--{model_config_name}-model-name",
        type=str,
        default=None,
        help=f"{model_config_name}: Model name",
    )
    parser.add_argument(
        f"--{model_config_name}-is-chat",
        type=str_to_bool,
        default=None,
        help=f"{model_config_name}: Is chat format?",
    )
    parser.add_argument(
        f"--{model_config_name}-is-batched",
        type=str_to_bool,
        default=None,
        help=f"{model_config_name}: Should batch?",
    )
    parser.add_argument(
        f"--{model_config_name}-batch-size",
        type=int,
        default=None,
        help=f"{model_config_name}: Batch size (if batched)",
    )
    parser.add_argument(
        f"--{model_config_name}-num-workers",
        type=int,
        default=None,
        help=f"{model_config_name}: Number of workers (if not batched)",
    )
    parser.add_argument(
        f"--{model_config_name}-price-per-input-output",
        type=float,
        nargs=2,
        default=None,
        help=f"{model_config_name}: Price per input/output token",
    )
    parser.add_argument(
        f"--{model_config_name}-base-url",
        type=str,
        default=None,
        help=f"{model_config_name}: Base URL (if SGLang)",
    )
    add_vllm_args(parser, model_config_name)


def add_vllm_args(parser: argparse.ArgumentParser, model_config_name: str):
    parser.add_argument(
        f"--{model_config_name}-tensor-parallel-size",
        type=int,
        default=None,
        help=f"{model_config_name}: Tensor parallel size (if vLLM)",
    )
    parser.add_argument(
        f"--{model_config_name}-gpu-memory-utilization",
        type=float,
        default=None,
        help=f"{model_config_name}: GPU memory utilization (if vLLM)",
    )
    parser.add_argument(
        f"--{model_config_name}-max-model-len",
        type=int,
        default=None,
        help=f"{model_config_name}: Max model length (if vLLM)",
    )
    parser.add_argument(
        f"--{model_config_name}-dtype",
        type=str,
        default=None,
        help=f"{model_config_name}: Data type (if vLLM)",
    )
    parser.add_argument(
        f"--{model_config_name}-enforce-eager",
        type=bool,
        default=None,
        help=f"{model_config_name}: Enforce eager (if vLLM)",
    )

"""
client_type
model_name
is_chat
is_batched
batch_size
price_per_input_output
num_workers
batch_size

base_url

tensor_parallel_size
gpu_memory_utilization
max_model_len
dtype
enforce_eager
"""

def maybe_overwrite_default(args: argparse.Namespace, config: dict[str], model_config_name: str, attr: str):
    val_in_args = getattr(args, f"{model_config_name}_{attr}", None)
    if val_in_args is not None:
        config[attr] = val_in_args
    return config.get(attr, None)

def parse_args_for_model_client(args: argparse.Namespace, model_config_name: str = "model", temp_folder_base: str = "temp_model_configs") -> str:
    json_config = getattr(args, f"{model_config_name}_config_path", None)
    if json_config is not None:
        with open(json_config, "r") as f:
            config = json.load(f)
    else:
        config = {}
    
    assert isinstance(config, dict)

    client_type = maybe_overwrite_default(args, config, model_config_name, "client_type")
    for param in CLIENT_TYPE_TO_CLASS[client_type].PARAMS:
        maybe_overwrite_default(args, config, model_config_name, param)

    new_json_path = os.path.join(temp_folder_base, f"{model_config_name}_{datetime.datetime.now().strftime('%m%dT%H:%M')}.json")
    with open(new_json_path, "w") as f:
        json.dump(config, f, indent=2)
    return new_json_path


if __name__ == "__main__":
    lol = 3
    parser = argparse.ArgumentParser()
    parser.add_argument(
        f"--bruh{lol}",
        type=float,
        nargs=2,
        required=True
    )
    parser.add_argument(
        "--b",
        type=str,
        required=True
    )
    args = parser.parse_args()
    print(args)
    print(args.b)
    print(args.bruh.hi)
