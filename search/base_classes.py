from abc import ABC, abstractmethod
from typing import List, Any, Optional, Union
import datetime
import os
import json

from coderm.model import BaseModel, Completion
from coderm.prompts import Prompt

from search.python_utils import wrap_list, stringify
from search.queriers import LLMQuerier


class Test:
    def __init__(self, input: tuple[list[Any], dict[str, Any]], output: Any, fn_name: Optional[str], has_Solution: Optional[bool]):
        assert isinstance(input, tuple) and len(input) == 2 and isinstance(input[0], list)
        if fn_name is None:
            assert (len(input[0]) == 1) and isinstance(input[0][0], str) and isinstance(output, str)

        self.input = input
        self.output = output
        assert (fn_name is None) == (has_Solution is None)
        self.fn_name = fn_name
        self.has_Solution = has_Solution

    def __repr__(self):
        return f"Test({self.input}, {self.output}, {self.fn_name}, {self.has_Solution})"
       
    def switch_fn_name(self, fn_name: str = "", has_Solution: bool = False):
        self.fn_name = fn_name
        self.has_Solution = has_Solution
    
    def get_input_no_kwargs(self) -> Union[str, list[Any]]:
        if self.fn_name is None:
            return self.input[0][0]
        else:
            return self.input[0]

    def to_repr_dict(self):
        return {
            "input_repr": repr(self.get_input_no_kwargs()),
            "output_repr": repr(self.output),
            "fn_name": self.fn_name,
        }
    
    def to_simple_assert(self):
        assert self.fn_name is not None or self.fn_name != ""
        return f"assert {self.fn_name}(*{self.get_input_no_kwargs()}) == {repr(self.output)}\n"


# Evan Wang, adapted from previously written code while at Caltech
class Problem:
    def __init__(self, problem_str: str, starter_code: str = "", public_tests: Optional[list[Test]] = None, private_tests: Optional[list[Test]] = None, public_exec_string: Optional[str] = None, private_exec_string: Optional[str] = None, fn_name: Optional[str] = None, solutions: Optional[list[str]] = None, fail_codes: Optional[list[str]] = None) -> None:
        self.problem_str = problem_str
        self.starter_code = starter_code
        self.has_Solution = "class Solution:" in starter_code if starter_code != "" else None

        if public_tests is None:
            self.public_tests: list[Test] = []
        else:
            self.public_tests = public_tests

        if private_tests is None:
            self.private_tests: list[Test] = []
        else:
            self.private_tests = private_tests

        self.public_exec_string = public_exec_string
        self.private_exec_string = private_exec_string

        # Make sure not both exec_string and tests are filled
        assert self.public_exec_string is None or len(self.public_tests) == 0
        assert self.private_exec_string is None or len(self.private_tests) == 0

        all_tests = self.public_tests + self.private_tests
        if len(all_tests):
            prev_fn_name = all_tests[0].fn_name
        else:
            prev_fn_name = fn_name
        
        assert len(all_tests) or private_exec_string is not None
        # Input validation
        for test in all_tests:
            assert isinstance(test, Test)
            assert prev_fn_name == test.fn_name
            assert test.has_Solution == self.has_Solution

            prev_fn_name = test.fn_name
            if prev_fn_name is None:
                assert starter_code == "" or starter_code is None
                assert isinstance(test.input[0][0], str) and isinstance(test.output, str)
            else:
                assert starter_code is not None and len(starter_code)

        self.og_fn_name = prev_fn_name
        self.fn_name = prev_fn_name
        
        self.solutions = solutions
        self.fail_codes = fail_codes

    def convert_stdio_to_fn_input(self):
        assert self.fn_name is None
        self.fn_name = ""
        for test in self.public_tests:
            test.switch_fn_name(self.fn_name, False)
        for test in self.private_tests:
            test.switch_fn_name(self.fn_name, False)
   
    def has_starter_code(self):
        return not (self.starter_code == "" or self.starter_code is None)
    
    def get_starter_code_fn(self) -> Optional[str]:
        if self.starter_code == None or self.starter_code == "":
            return None
        starter_split = self.starter_code.splitlines()
        if self.has_Solution:
            assert len(starter_split) == 3
            fn_line = 1
        else:
            assert len(starter_split) == 2
            fn_line = 0
        return starter_split[fn_line].split("def ")[1].rstrip()
    
    def get_starter_code_fn_name(self) -> Optional[str]:
        starter_code_fn = self.get_starter_code_fn()
        if starter_code_fn is None:
            return None
        return starter_code_fn.split("(")[0]

    def tests_to_dict(self, use_private: bool = True) -> dict[str, Any]:
        out_dict = {"inputs": [], "outputs": []}
        tests = self.private_tests if use_private else self.public_tests
        for test in tests:
            out_dict["inputs"].append(test.get_input_no_kwargs())
            out_dict["outputs"].append(test.output)

        if self.og_fn_name is not None:
            out_dict["fn_name"] = self.og_fn_name

        return out_dict
    
    def get_test_private(self) -> Union[str, list[Test]]:
        if self.private_exec_string is None:
            return self.private_tests
        return self.private_exec_string

    def get_test_public(self) -> Union[str, list[Test]]:
        if self.public_exec_string is None:
            return self.public_tests
        return self.public_exec_string

    def to_dict(self) -> dict[str, Any]:
        out_dict: dict[str, Union[str, list[str]]] = {
                "question": self.problem_str,
                "starter_code": self.starter_code,
                }

        private_test_dict = self.tests_to_dict(use_private=True)
        if self.private_exec_string is not None:
            private_test_dict["exec_string"] = self.private_exec_string

        public_test_dict = self.tests_to_dict(use_private=False)
        if self.public_exec_string is not None:
            public_test_dict["exec_string"] = self.public_exec_string
        
        out_dict["input_output"] = stringify(private_test_dict)
        out_dict["public_input_output"] = stringify(private_test_dict)

        if self.solutions is not None and self.fail_codes is not None and (len(self.solutions) or len(self.fail_codes)):
            out_dict["solutions"] = self.solutions
            out_dict["fail_codes"] = self.fail_codes

        return out_dict

    
    @staticmethod
    def from_coderm_item(question: str, starter_code: str, public_tests: Optional[dict[str, Any]], tests: Optional[dict[str, Any]], solutions: Optional[list[str]] = None, fail_codes: Optional[list[str]] = None) -> "Problem":
        if public_tests is None:
            assert tests is not None
            public_tests = {"fn_name": tests.get("fn_name", None), "inputs": [], "outputs": [], "exec_string": None}
        if tests is None:
            assert public_tests is not None
            tests = {"fn_name": public_tests.get("fn_name", None), "inputs": [], "outputs": [], "exec_string": None}

        public_tests["fn_name"] = public_tests.get("fn_name", None)
        public_tests["exec_string"] = public_tests.get("exec_string", None)
        tests["fn_name"] = tests.get("fn_name", None)
        tests["exec_string"] = tests.get("exec_string", None)

        assert public_tests["fn_name"] == tests["fn_name"]
        fn_name = tests["fn_name"]
        has_Solution = ("class Solution:" in starter_code) if (fn_name is not None) else None

        assert len(public_tests["inputs"]) == len(public_tests["outputs"])
        public_test_list = [Test((wrap_list(inp), {}), out, fn_name, has_Solution) for inp, out in zip(public_tests["inputs"], public_tests["outputs"])]
        
        assert len(tests["inputs"]) == len(tests["outputs"])
        test_list = [Test((wrap_list(inp), {}), out, fn_name, has_Solution) for inp, out in zip(tests["inputs"], tests["outputs"])]

        return Problem(question, starter_code=starter_code, public_tests=public_test_list, private_tests=test_list, fn_name=fn_name, public_exec_string=public_tests["exec_string"], private_exec_string=tests["exec_string"], solutions=solutions, fail_codes=fail_codes)


class SearchModel(BaseModel, ABC):
    COMPLETION_FROM_MODEL_SUPPORTED = False
    def __init__(self, model_config_path: str, experiment_directory: Optional[str], cache_file: Optional[str], querier_batch_size: Optional[int]):
        str_model_path = model_config_path.replace('/', '-')
        super().__init__(str_model_path)
        self.experiment_directory = (experiment_directory if experiment_directory is not None 
                                     else f"logs/{datetime.datetime.now().strftime('%m%dT%H%M%S')}_{str_model_path}")
        self.querier = LLMQuerier(os.path.join(self.experiment_directory, "queries"), cache_file=cache_file, global_batch_size=querier_batch_size)

    def format_prompt(self, question: str, code: str = "", public_tests: Optional[dict[str, Any]] = None, tests: Optional[dict[str, Any]] = None, solutions: Optional[list[str]] = None) -> str | list[dict[str, Any]]:
        print("Warning: `format_prompt` is misused right now.")
        return [{"problem_str": question, "starter_code": code, "public_tests": public_tests, "tests": tests, "solutions": solutions}]

    @abstractmethod
    def generate_solutions(self, problems: list[Problem], *args, **kwargs) -> list[list[str]]:
        pass
    
    def generate_with_info(self, prompts: List[Prompt], **kwargs) -> List[Completion]:
        if len(kwargs):
            print("Warning: kwargs not used")
        problems = [Problem.from_coderm_item(prompt[0]["problem_str"], starter_code=prompt[0]["starter_code"], public_tests=prompt[0]["public_tests"], tests=prompt[0]["tests"], solutions=prompt[0]["solutions"]) for prompt in prompts]
        raise NotImplementedError("Code branch not maintained")
        solutions = self.generate_solutions(problems)
        return [Completion(code, -1, -1) for code in solutions]
    
    def prefix_starter_code(self) -> bool:
        return False
