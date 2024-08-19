from typing import Any, Optional, Union
import argparse
import os
import itertools
import random
import datetime

from search.base_classes import Problem, SearchModel
from search.parsing_utils import markdown_codeblock_extract
from search.exec_utils import run_tests_per_code
from search.python_utils import log_to_dir, batch_map_on_nested_list, map_on_nested_list, nested_list_len, index_nested_list, merge_nested_lists
from search.model_config_utils import add_model_config_args, parse_args_for_model_client


class ComboObservationModel(SearchModel):
    import search.prompts.combo_observation_prompts as prompts
    COMPLETION_FROM_MODEL_SUPPORTED = True
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
                              timeout=150,
                              )
        assert len(outputs) == len(prompts)
        return outputs
 

    def get_first_observations_prompt(self, problem: Problem) -> list[dict[str, str]]:
        convo = [{"role": "system", "content": self.prompts.SYSTEM_PROMPT_OBSERVATION},
                 {"role": "user", "content": self.prompts.get_observation(problem.problem_str, self.num_observations_to_generate)}]
        return convo
    
    def get_combined_observations_prompt(self, problem: Problem, observation_combos: tuple[str, ...]) -> list[dict[str, str]]:
        convo = [{"role": "system", "content": self.prompts.SYSTEM_PROMPT_OBSERVATION2},
                 {"role": "user", "content": self.prompts.combine_observations(problem.problem_str, observation_combos)}]
        return convo
  
    def get_observations_strs(self, problems_obs: list[tuple[Problem, Optional[tuple[str, ...]]]], iter_num: int = 0) -> list[tuple[tuple[str, ...], dict[str]]]:
        problems = [problem for problem, _ in problems_obs]
        observation_combos = [obs for _, obs in problems_obs]

        if iter_num == 0:
            assert all(combs is None for combs in observation_combos)
            get_observations_prompts = [self.get_first_observations_prompt(problem) for problem in problems]
        elif iter_num == 1:
            assert all(isinstance(combs, tuple) for combs in observation_combos)
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

        python_obs_lists = [() if el is None else tuple(el) for el in python_obs_lists]

        logs = [{
            "problem_str": problems[i].problem_str,
            "observations_str": observations_strs[i],
            "observations_strlist": observations_strlists[i],
            "filtered_strlist": filtered_obs_strlists[i],
            "attempted_parses": log_attempted_parses[i],
            "python_observation_list": python_obs_lists[i]}
            for i in range(len(problems))]

        assert len(python_obs_lists) == len(logs)
        return [(obs_list, log) for obs_list, log in zip(python_obs_lists, logs)]


    def split_into_observation_combos(self, observations: tuple[str, ...]) -> list[tuple[str, ...]]:
        max_k = min(self.max_observation_k, len(observations))

        observation_combos = []
        for k in range(max_k+1):
            observation_combos.extend(itertools.combinations(observations, k))
        
        random.shuffle(observation_combos)
        return observation_combos


    def get_nl_sols_prompt(self, problem: Problem, observation_combo: tuple[str, ...]) -> list[dict[str, str]]:
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
    
    def get_code_solution_from_obs_combos(self, problem_observation_combos: list[tuple[Problem, tuple[str, ...]]]) -> list[tuple[Problem, str, dict[str]]]:
        expanded_problems = [problem for problem, _ in problem_observation_combos]
        observation_combos = [obs for _, obs in problem_observation_combos]

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
            "output_code": output_codes[i],
            "parsed_code": parsed_codes[i]}
                for i in range(len(expanded_problems))]
        return [(problem, code, log) for problem, code, log in zip(expanded_problems, parsed_codes, logs)]

    def generate_solutions(self, problems: list[Problem], *args, **kwargs) -> list[str]:
        completions_from_model = kwargs.get("completions_from_model", False)
        num_completions = 1
        if completions_from_model:
            num_completions = kwargs.get("num_completions", 1)

        observations_lists_logs = self.get_observations_strs([(problem, None) for problem in problems], iter_num=0)
        observations1_lists = [obs_list for obs_list, _ in observations_lists_logs]
        observations1_logs = [log for _, log in observations_lists_logs]
        assert len(observations1_lists) == len(problems)

        log_to_dir(self.experiment_directory, {f"observation_0_{datetime.datetime.now().strftime('%m-%dT%H:%M:%S')}.json": observations1_logs})

        # List of len(problem). Each element is (problem, list of tuples each representing an observation combo)
        problem_observation_combos = [[(problem, combo)
                                       for combo in self.split_into_observation_combos(observations)]
                                      for problem, observations in zip(problems, observations1_lists)]

        # List L1 of length len(problems), each element corresponds to an observation combo, and contains a list of new observations
        #           problems
        #              / \
        # .. | an observation combo | ..
        #              /      \
        #     ... | tuple of 2nd observations from that combo | ...
        observations2_lists_logs = batch_map_on_nested_list(problem_observation_combos, lambda li: self.get_observations_strs(li, 1))
        assert len(observations2_lists_logs) == len(problems)
        assert all(len(obs2_lists_logs_for_prob) == len(prob_obs_list1_for_problem)
                   for prob_obs_list1_for_problem, obs2_lists_logs_for_prob in zip(problem_observation_combos, observations2_lists_logs))

        observations2_lists = map_on_nested_list(observations2_lists_logs, lambda x: x[0])
        observations2_logs = map_on_nested_list(observations2_lists_logs, lambda x: x[1])

        log_to_dir(self.experiment_directory, {f"observation_1_{datetime.datetime.now().strftime('%m-%dT%H:%M:%S')}.json": observations2_logs})

        new_observation_combos = map_on_nested_list(observations2_lists, self.split_into_observation_combos)
        problem_observation_combos = [ # problem layer
            [ # observation 1 layer
                [ # observation 2 layer
                    [(problem, o) for o in [obs1] + obs_list2] for (problem, obs1), obs_list2 in zip(prob_obs_list1_for_problem, obs_list2_for_problem)
                ]
            ] for prob_obs_list1_for_problem, obs_list2_for_problem in zip(problem_observation_combos, new_observation_combos)]

        code_sols_and_logs = batch_map_on_nested_list(problem_observation_combos, self.get_code_solution_from_obs_combos)
        code_probs = map_on_nested_list(code_sols_and_logs, lambda x: x[0])
        code_sols = map_on_nested_list(code_sols_and_logs, lambda x: x[1])
        code_logs = map_on_nested_list(code_sols_and_logs, lambda x: x[2])

        prob_sols = merge_nested_lists(code_probs, code_sols)
        tot_len = nested_list_len(code_sols)
        results = batch_map_on_nested_list(prob_sols, 
            lambda li: run_tests_per_code([code for _, code in li], [prob.get_test_public() for prob, _ in li], 
                                          [self.timeout] * tot_len, num_workers=self.num_workers, testbank=self.testbank, executor=self.executor)
        )

        assert len(results) == len(problems)

        # Logging
        code_logs_results = merge_nested_lists(code_logs, results)
        def temp(x: tuple[dict[str], tuple[bool, str]]):
            x[0]["gens"] = {"passed": x[1][0], "error": x[1][1]}
        map_on_nested_list(code_logs_results, temp)

        code_logs_probs = merge_nested_lists(code_logs, code_probs)
        def temp(x: tuple[dict[str], Problem]):
            public_test = x[1].get_test_public()
            x[0]["tests"] = public_test if isinstance(public_test, str) else [test.to_repr_dict() for test in public_test]
        map_on_nested_list(code_logs_probs, temp)


        selected_codes = ["# No successful generation\n"] * len(problems) * num_completions
        sols_results = merge_nested_lists(code_sols, results)

        for i, sols_results_for_problem in enumerate(sols_results):
            flattened_code_result_for_problem: list[tuple[str, tuple[bool, str]]] = []
            index_nested_list(sols_results_for_problem, flattened_code_result_for_problem, [])

            good_codes = []

            for code, (result_good, _) in flattened_code_result_for_problem:
                if result_good:
                    good_codes.append(code)
            
            if len(good_codes) == 0:
                continue
            
            random.shuffle(good_codes)
            for completion_idx in range(num_completions):
                selected_codes[i + len(problems) * completion_idx] = good_codes[completion_idx % len(good_codes)]

        log_to_dir(os.path.join(self.experiment_directory), {f"codes_{datetime.datetime.now().strftime('%m-%dT%H:%M:%S')}.json": code_logs})

        return selected_codes


def add_combo_observation_args(parser: argparse.ArgumentParser):
    add_model_config_args(parser, "idea_model")
    add_model_config_args(parser, "code_model")
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
    idea_model_path = parse_args_for_model_client(args, model_config_name="idea_model", temp_folder_base=args.experiment_directory)
    code_model_path = parse_args_for_model_client(args, model_config_name="code_model", temp_folder_base=args.experiment_directory)
    return ComboObservationModel(idea_model_config_path=idea_model_path, code_model_config_path=code_model_path, experiment_directory=args.experiment_directory, cache_file=args.cache_file, querier_batch_size=args.global_batch_size, seed=args.seed, max_observation_k=args.max_observation_k, num_observations_to_generate=args.num_observations_to_generate, idea_temperature=args.idea_temperature, code_temperature=args.code_temperature, top_p=args.top_p, max_tokens=args.max_tokens, timeout=args.timeout, num_workers=args.exec_batch_size, testbank=args.testbank, executor=args.executor)
