# PlanSearch: Planning In Natural Language Improves LLM Search For Code Generation

## Overview

PlanSearch is a novel search algorithm designed to boost large language models' (LLMs) performance by increasing diversity in generated solutions to competitive programming problems. It achieves strong performance on benchmarks such as HumanEval+, MBPP+, and LiveCodeBench. Using PlanSearch on top of Claude 3.5 Sonnet achieves a pass@200 of 77.0% on LiveCodeBench, which outperforms tboth the best pass-rate achieved without any search (pass@1 = 41.4%) and using standard repeated sampling on top of existing non-search models (pass@200 = 60.6%).

This repository contains the code and scripts used to reproduce the experiments in PlanSearch, including both the actual runs themselves as well as diversity measurement scores.


## Setup

### Installation

Clone the repository and install the necessary dependencies. This git repo also uses submodules. 

```bash
git clone https://github.com/evanzwang/plansearch.git --recurse-submodules
cd plansearch
pip install -e .
pip install -e CodeRM
```

`plansearch/CodeRM` contains code for execution. Code execution can be set up to run locally by running:
```bash
pushd CodeRM/coderm/code_exec_server
./build_and_run.sh
popd
```
(these instructions are also in `CodeRM/README.md`)

Running on a custom cluster is possible but may require some modifications to the codebase.

## Usage

### Datasets

The datasets can be found at `https://huggingface.co/codegenning`. 
- LiveCodeBench split: `codegenning/F_livecodebench_lite_v3` (associated testbank: `codegenning/B_livecodebench_lite_v3`)
- MBPP+ split: `codegenning/F_mbpp_plus` (associated testbank: `codegenning/B_mbpp_plus_v2`)
- HumanEval+ split: `codegenning/F_human_eval_plus` (associated testbank: `codegenning/B_huan_eval_plus_v2`)

#### Dataset Formats

F_ format mandatory columns:
- `question`: String representing the question
- `starter_code`: A string representing the starter code to the problem, such as LeetCode's `Solution: ...`. If no starter code required, is the empty string.
- `input_output`: A JSON string containing a dictionary with keys `inputs`, `outputs`, and optionally `fn_name` or `exec_string`.
    - `inputs`: A list of inputs, one for each test case. Same length as `outputs`.
    - `outputs`: A list of outputs, one for each test case.
    - `fn_name`: If provided, is a string citing the entrypoint (i.e., function name) the test will run. Otherwise, the both input and output are strings and accepted through standard in/out.
    - `exec_string`: If provided, will simply run `exec_string` instead of the default test harness. Every other key will be ignored.

F_ format optional columns:
- `public_input_output`: String the same as `input_output` format, but describing the public tests instead

### Running PlanSearch and Baselines

To run experiments, you can use the provided `eval.py`. It takes in an environment variable, `SEARCH_ALG`, which determines the search baseline to use. Options include:
- `basic_prompting` - Repeated Sampling
- `simple_idea` - IdeaSearch
- `combo_observation` - PlanSearch
- `backtranslate` - Backtranslation experiments

No matter the search algorithm, `eval.py` takes in several arguments:
- `experiment-directory`: Optional string. Specifies where to save experiment and query logs. If None, will save to a default logs directory.
- `cache-file`: Must end in `.json`, and tracks where to cache queries. If already exists, will use and update that cache file when querying.
- `dataset`: Path to a Hugging Face dataset
- `split`: Choose between `train`, `test`, or `both`. These specify which dataset split to evaluate on. 
- `exec-type`: Choose between `none`, `private`, or `both`. These specify whether to execute no tests, only private tests, or both public and private tests.
- `output-dataset`: Optional string. Specifies where to output a dataset of generations (will upload to Hugging Face). Will not output to a dataset if None specified.
- `output`: Where to output the results file. 
- `num-repeats`: How many repeats to do, i.e., how many copies of a problem to be provided to a model to create `num-repeats` independent completions.
- `num-completions-from-model`: For each problem fed to the model, how many completions to generate. (Set to $-1$ if the model determines the number of completions itself, like PlanSearch.)
- `global-batch-size`: The global batch size for `LLMQuerier`, i.e., how many queries to do before saving to cache.

In addition, there are arguments for the code execution:
- `exec-batch-size`: How many total number of code/test pairs to execute at a time
- `exec-num-processes`: Number of processes to separate `exec-batch-size` requests into
- `executor`: The server URL for executing the code. (None if local)
- `testbank`: Optional string that if provided, points to a Hugging Face testbank which is a dataset of hash to test which is sent to the server for caching.
- `timeout`: The timeout (in s) for each code/test pair to run. Default is $60$ s.


#### Example PlanSearch run
```bash
SEARCH_ALG="combo_observation" \
python eval.py \
--idea-model-config-path model_configs/gpt-4o.json \
--code-model-config-path model_configs/gpt-4o.json \
--output final_results/livecodebench_lite_v3/plansearch_gpt-4o_temp0.9 \
--experiment-directory final_logs/livecodebenchlite_v3/plansearch_gpt-4o_temp0.9 \
--dataset codegenning/F_livecodebench_lite_v3 \
--num-completions-from-model -1 \
--idea-temperature 0.9 \
--code-temperature 0.9 \
--top-p 0.95 \
--split test \
--testbank codegenning/B_livecodebench_lite_v3 \
--cache-file caches/gpt-4o_livecodebench_lite_v3_plansearch_cache.json \
--exec-type both \
--exec-batch-size 1800 \
--exec-num-processes 180 \
--num-observation-layers 2 \
--num-repeats 1 \
```

See [specific arguments](#specific-search-algorithm-arguments) for the specific arguments which may be distinct between search algorithms.

## Misc. Info

Results are saved as `.json.gz` files, and if tests were executed, pass@k can be computed using CodeRM's pass@k script in `coderm/coderm/eval/metrics.py`. 

Plotting utilities can be found in `notebooks/graph_notebooks`, with final versions of the graphs found in the paper produced by `notebooks/graph_notebooks/final_2_pass_k.ipynb`.

Parse dataset name into list of Problems (parse_dataset_utils.parse_dataset)

### Diversity

Diversity measure evaluation is run using `scripts/check_similar.py`, and returns the results to a `results.npy` file within the log directory.  This script takes arguments:
- `model-config-path`: String pointing to a `.json` file providing the model for diversity evaluations
- `results-file`: Where the generated codes are saved (i.e., the file passed into `gunzip_json_read`.)
- `comparison-type`: Choose between `default`, `right-wrong`, `right`, `wrong` to compare diversity over different subsets of the generations. Default is from all generations, right-wrong compares the diversity across correct vs wrong generations, right is only on generations that are correct, and similarly for wrong. (These results are not reported in the paper but served as a sanity check for diversity measure.)
- `log-directory`: Where to log. (defaults to `other_logs/similar_logs/...`)
- `seed`: Seed for RNG. (default 42)
- `max-to-consider`: Maximum number of codes to consider in the diversity estimation
- `max-fill-idea-attempts`: Maximum number of times to attempt querying for ideas, since the request may be rejected. (default 3)
- `cache-file`: `.json` file describing where caches are stored.

### Model Configs

Model config examples can be found in `model_configs/`. You can also add your own configuration files by following the examples. Currently, OpenAI, Anthropic, DeepSeek API, vLLM, SGLang, Fireworks, and Together are supported.

### Specific search algorithm arguments
- Repeated Sampling/`basic_prompting`:
    - `model-config-path`: string pointing to a `.json` file providing the model information to use for prompting
    - `max-tokens`: Max tokens per response (defaults to 4096)
    - `temperature`: Temperature (defaults to 0)
    - `top-p`: Top-p (defaults to 0.9)
- IdeaSearch/`simple_idea`:
    - `idea-model-config-path`: string pointing to a `.json` file providing the model for the idea generation
    - `code-model-config-path`: string pointing to a `.json` file providing the model for the idea to code step
    - `idea-temperature`: Temperature for the idea model generations
    - `code-temperature`: Temperature for the code model generations
    - `top-p`: same
- PlanSearch/`combo_observation`:
    - `idea-model-config-path`: string pointing to a `.json` file providing the model for the idea generation
    - `code-model-config-path`: string pointing to a `.json` file providing the model for the idea to pseudocode and pseudocode to code steps
    - `idea-temperature`: Temperature for the idea model generations
    - `code-temperature`: Temperature for the code model generations
    - `top-p`: same
    - `seed`: Seed for RNG
    - `max-observation-k`: $S$ as described in the paper; the maximum subset size when generating new observations (default 2)
    - `num-observations-to-generate`: How many observations to start off generating in the first layer (default 10)
    - `num-observation-layers`: How many layers of observations to do (default 2)
    - `without-fix-idea`: If specified, will not do the 'fix solution sketch' step
    - `without-pseudocode`: If specified, will skip the pseudocode step and go directly from natural language solution sketch to output code
    - `without-idea`: If specified, will skip the solution sketch and pseudocode step. Goes directly from observation combination to output code



