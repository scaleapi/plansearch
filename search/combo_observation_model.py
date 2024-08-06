from typing import Any, Optional, Union
import argparse
import os
import itertools
import random
import datetime

from search.base_classes import Problem, SearchModel
from search.parsing_utils import markdown_codeblock_extract
from search.exec_utils import run_tests_per_code
from search.python_utils import log_to_dir


class ComboObservationModel(SearchModel):
    import search.prompts.combo_observation_prompts as prompts
    def __init__(self, idea_model_config_path: str, code_model_config_path: str, experiment_directory: Optional[str] = None, cache_file: Optional[str] = None, querier_batch_size: Optional[int] = 12_288, max_observation_k: int = 2, num_observations_to_generate: int = 10, frequency_penalty: Optional[float] = None, logit_bias: Optional[dict[str, int]] = None, max_tokens: Optional[int] = None, presence_penalty: Optional[float] = None, seed: Optional[int] = None, stop: Union[Optional[str], list[str]] = None, idea_temperature: Optional[float] = None, code_temperature: Optional[float] = None, top_p: Optional[float] = None, timeout: int = 30, num_workers: Optional[int] = os.cpu_count(), testbank: Optional[str] = None, executor: str = "http://127.0.0.1:8000"):
        super().__init__("observation", experiment_directory=experiment_directory, cache_file=cache_file, querier_batch_size=querier_batch_size)

        self.idea_model = idea_model_config_path
        self.code_model = code_model_config_path

        self.frequency_penalty = frequency_penalty
        self.logit_bias = logit_bias
        self.max_tokens = max_tokens
        self.presence_penalty = presence_penalty
        self.seed = seed
        self.stop = stop
        self.code_temperature = code_temperature
        self.idea_temperature = idea_temperature
        self.top_p = top_p
        random.seed(self.seed)

        self.max_observation_k = max_observation_k
        self.num_observations_to_generate = num_observations_to_generate
        self.timeout = timeout
        self.num_workers = num_workers
        self.testbank = testbank
        self.executor = executor

  
    def query_model(self, model_type: str, prompts: list[list[dict[str, str]]], temperature: Optional[float] = None) -> list[str]:
        assert model_type in {"idea", "code"}
        if model_type == "idea":
            model = self.idea_model
            use_temperature = self.idea_temperature
        else:
            model = self.code_model
            use_temperature = self.code_temperature
        if temperature is not None:
            use_temperature = temperature

        outputs = self.querier.generate(model, 
                              prompts,
                              frequency_penalty=self.frequency_penalty,
                              logit_bias=self.logit_bias,
                              max_tokens=self.max_tokens,
                              presence_penalty=self.presence_penalty,
                              seed=self.seed,
                              stop=self.stop,
                              temperature=use_temperature,
                              top_p=self.top_p,
                              requery=True,
                              )
        assert len(outputs) == len(prompts)
        return outputs
 

    def get_first_observations_prompt(self, problem: Problem) -> list[dict[str, str]]:
        convo = [{"role": "system", "content": self.prompts.SYSTEM_PROMPT_OBSERVATION},
                 {"role": "user", "content": self.prompts.get_observation(problem.problem_str, self.num_observations_to_generate)}]
        return convo
    
    def get_combined_observations_prompt(self, problem: Problem, observation_combos: tuple[str]) -> list[dict[str, str]]:
        convo = [{"role": "system", "content": self.prompts.SYSTEM_PROMPT_OBSERVATION2},
                 {"role": "user", "content": self.prompts.combine_observations(problem.problem_str, observation_combos)}]
        return convo
  
    def get_observations_strs(self, problems: list[Problem], observation_combos: Optional[list[tuple[str]]] = None, iter_num: int = 0) -> list[list[str]]:
        if iter_num == 0:
            assert observation_combos is None
            get_observations_prompts = [self.get_first_observations_prompt(problem) for problem in problems]
        elif iter_num == 1:
            assert observation_combos is not None
            assert len(problems) == len(observation_combos)
            get_observations_prompts = [self.get_combined_observations_prompt(problem, obs_combo) for problem, obs_combo in zip(problems, observation_combos)]
        else:
            raise ValueError(f"iter_num {iter_num} not supported")

        observations_strs = self.query_model("idea", get_observations_prompts)

        get_parse_into_list_prompts = [prompts + 
                      [{"role": "assistant", "content": observations_str},
                       {"role": "user", "content": self.prompts.FORMAT_INTO_LIST_PROMPT}]
                      for prompts, observations_str in zip(get_observations_prompts, observations_strs)]
        observations_strlists = self.query_model("idea", get_parse_into_list_prompts)


        filter_observations_prompts = [prompts + 
                      [{"role": "assistant", "content": observations_strlist},
                       {"role": "user", "content": self.prompts.FILTER_TO_USEFUL_LIST_PROMPT}]
                      for prompts, observations_strlist in zip(get_parse_into_list_prompts, observations_strlists)]
        filtered_obs_strlists = self.query_model("idea", filter_observations_prompts)


        parse_to_python_prompts = [prompts + 
                      [{"role": "assistant", "content": filtered_obs_strlist},
                       {"role": "user", "content": self.prompts.PARSE_INTO_PYTHON_LIST_PROMPT}]
                      for prompts, filtered_obs_strlist in zip(filter_observations_prompts, filtered_obs_strlists)]
        
        MAX_PARSE_TRIES = 3
        python_obs_lists = [None] * len(filtered_obs_strlists)
        unused_idxs = list(range(len(filtered_obs_strlists)))

        log_attempted_parses = [[] for _ in range(len(filtered_obs_strlists))]

        for iter in range(MAX_PARSE_TRIES):
            to_query = [parse_to_python_prompts[i] for i in unused_idxs]
            parsed_obs_pythonlists = self.query_model("idea", to_query, temperature=iter * 0.2)

            for orig_idx, parse in zip(unused_idxs, parsed_obs_pythonlists):
                log_attempted_parses[orig_idx].append(parse)

            for orig_idx, parsed_obs_python in zip(unused_idxs, parsed_obs_pythonlists):
                try:
                    attempted_parse = eval(markdown_codeblock_extract(parsed_obs_python))
                    assert isinstance(attempted_parse, list)
                    assert all(isinstance(parse, str) for parse in attempted_parse)
                    python_obs_lists[orig_idx] = attempted_parse
                except:
                    pass
            
            unused_idxs = [i for i, el in enumerate(python_obs_lists) if el is None]
            if len(unused_idxs) == 0:
                break

        if any(el is None for el in python_obs_lists):
            print("Warning: Python parsing of observation lists failed.")

        python_obs_lists = [[] if el is None else el for el in python_obs_lists]

        logs = [{
            "problem_str": problems[i].problem_str,
            "observations_str": observations_strs[i],
            "observations_strlist": observations_strlists[i],
            "filtered_strlist": filtered_obs_strlists[i],
            "attempted_parses": log_attempted_parses[i],
            "python_observation_list": python_obs_lists[i]}
            for i in range(len(problems))]
        log_to_dir(self.experiment_directory, {f"observation_{iter_num}_{datetime.datetime.now().strftime('%m-%dT%H:%M:%S')}.json": logs})

        return python_obs_lists


    def split_into_observation_combos(self, observations: list[str]) -> list[tuple[str]]:
        max_k = min(self.max_observation_k, len(observations))

        observation_combos = []
        for k in range(max_k+1):
            observation_combos.extend(itertools.combinations(observations, k))
        
        random.shuffle(observation_combos)
        return observation_combos


    def get_nl_sols_prompt(self, problem: Problem, observation_combo: tuple[str]) -> list[dict[str, str]]:
        convo = [{"role": "system", "content": self.prompts.SYSTEM_PROMPT_NL_SOL_FROM_OBS_COMBO},
                 {"role": "user", "content": self.prompts.get_nl_solution_from_obs_combo(problem.problem_str, observation_combo)}]
        return convo

    def get_pseudocode_prompt(self, problem: Problem, nl_solution: str) -> list[dict[str, str]]:
        convo = [{"role": "system", "content": self.prompts.SYSTEM_PROMPT_PSEUDOCODE},
                 {"role": "user", "content": self.prompts.get_pseudocode(problem.problem_str, nl_solution)}]
        return convo

    def pseudocode_to_code_solution_prompt(self, problem: Problem, pseudocode: str) -> list[dict[str, str]]:
        convo = [{"role": "system", "content": self.prompts.SYSTEM_PROMPT_GENERATE},
                 {"role": "user", "content": self.prompts.generate_code_sol(problem.problem_str, pseudocode, problem.starter_code)}]
        return convo
    
    def get_code_solution_from_obs_combos(self, expanded_problems: list[Problem], observation_combos: list[tuple[str]]) -> tuple[list[str], Any]:
        assert len(expanded_problems) == len(observation_combos)
        get_nl_sols_prompt = [self.get_nl_sols_prompt(problem, observation_combo)
                              for problem, observation_combo in zip(expanded_problems, observation_combos)]
        nl_solutions = self.query_model("idea", get_nl_sols_prompt)

        get_pseudocode_prompt = [self.get_pseudocode_prompt(problem, nl_solution)
                                 for problem, nl_solution in zip(expanded_problems, nl_solutions)]
        pseudocodes = self.query_model("code", get_pseudocode_prompt)

        get_code_prompt = [self.pseudocode_to_code_solution_prompt(problem, pseudocode)
                                 for problem, pseudocode in zip(expanded_problems, pseudocodes)]
        output_codes = self.query_model("code", get_code_prompt)
        parsed_codes = [markdown_codeblock_extract(genned).strip() for genned in output_codes]

        logs = [{
            "problem_str": expanded_problems[i].problem_str,
            "nl_solution": nl_solutions[i],
            "pseudocode": pseudocodes[i],
            "output_codes": output_codes[i],
            "parsed_codes": parsed_codes[i]}
                for i in range(len(expanded_problems))]
        return parsed_codes, logs


    def generate_solutions(self, problems: list[Problem], *args, **kwargs) -> list[str]:
        observations_lists = self.get_observations_strs(problems, iter_num=0)

        observation_combos: list[tuple[str]] = []
        observation_combos_to_orig_problem_idx: list[int] = []
        for i, observations in enumerate(observations_lists):
            new_observation_combos = self.split_into_observation_combos(observations)

            observation_combos.extend(new_observation_combos)
            observation_combos_to_orig_problem_idx.extend([i] * len(new_observation_combos))
        
        # Secondary observations
        new_observation_lists = self.get_observations_strs([problems[orig_idx] for orig_idx in observation_combos_to_orig_problem_idx], observation_combos, iter_num=1)
        for orig_idx, observations in zip(observation_combos_to_orig_problem_idx, new_observation_lists):
            new_observation_combos = self.split_into_observation_combos(observations)

            observation_combos.extend(new_observation_combos)
            observation_combos_to_orig_problem_idx.extend([orig_idx] * len(new_observation_combos))

        code_sols, code_logs = self.get_code_solution_from_obs_combos([problems[orig_idx] for orig_idx in observation_combos_to_orig_problem_idx], observation_combos)
        assert len(code_sols) == len(observation_combos) == len(observation_combos_to_orig_problem_idx)

        # remap back to orig index
        results = run_tests_per_code(code_sols, [problems[orig_idx].public_tests for orig_idx in observation_combos_to_orig_problem_idx], [self.timeout] * len(code_sols), num_workers=self.num_workers, testbank=self.testbank, executor=self.executor)
        assert len(results) == len(code_sols)

        selected_codes = ["# No successful generation\n"] * len(problems) 

        for i, (orig_idx, result, code) in enumerate(zip(observation_combos_to_orig_problem_idx, results, code_sols)):
            result_good, result_error = result
            if result_good:
                selected_codes[orig_idx] = code
            
            code_logs[i]["tests"] = [test.to_repr_dict() for test in problems[orig_idx].public_tests]
            code_logs[i]["gens"] = {"passed": result_good, "error": result_error}

        log_to_dir(os.path.join(self.experiment_directory), {f"codes_{datetime.datetime.now().strftime('%m-%dT%H:%M:%S')}.json": code_logs})

        return selected_codes


def add_combo_observation_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--idea-model-config-path",
        required=True,
        help="Model config to use for ideas"
    )
    parser.add_argument(
        "--code-model-config-path",
        required=True,
        help="Model config to use for implementation"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max new tokens to generate"
    )
    parser.add_argument(
        "--idea-temperature",
        type=float,
        default=0.,
        help="Temperature for sampling ideas"
    )
    parser.add_argument(
        "--code-temperature",
        type=float,
        default=0.,
        help="Temperature for sampling codes"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3,
        help="Random seed"
    )
    parser.add_argument(
        "--max-observation-k",
        type=int,
        default=2,
        help="Max K in choosing K-sized subsets of generated observations"
    )
    parser.add_argument(
        "--num-observations-to-generate",
        type=int,
        default=10,
        help="Number of observations to generate"
    )

def get_combo_observation_model(args: argparse.Namespace) -> SearchModel:
    return ComboObservationModel(args.idea_model_config_path, args.code_model_config_path, args.experiment_directory, cache_file=args.cache_file, querier_batch_size=args.global_batch_size, seed=args.seed, max_observation_k=args.max_observation_k, num_observations_to_generate=args.num_observations_to_generate, idea_temperature=args.idea_temperature, code_temperature=args.code_temperature, top_p=args.top_p, max_tokens=args.max_tokens, timeout=args.timeout, num_workers=args.num_workers, testbank=args.testbank, executor=args.executor)
