from abc import ABC, abstractmethod
from typing import List, Any, Optional, Union
import datetime
import os

from python_utils import wrap_list
from coderm.model import BaseModel, Completion
from coderm.prompts import Prompt
from queriers import LLMQuerier


class Test:
    def __init__(self, input: tuple[list[Any], dict[str, Any]], output: Any, fn_name: Optional[str]):
        assert isinstance(input, tuple) and len(input) == 2 and isinstance(input[0], list)
        if fn_name is None:
            assert (len(input[0]) == 1) and isinstance(input[0][0], str) and isinstance(output, str)

        self.input = input
        self.output = output
        self.fn_name = fn_name

    def __repr__(self):
        return f"Test({self.input}, {self.output}, {self.fn_name})"
       
    def switch_fn_name(self, fn_name: str = ""):
        self.fn_name = fn_name
    
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
    def __init__(self, problem_str: str, starter_code: str = "", public_tests: Optional[list[Test]] = None, private_tests: Optional[list[Test]] = None, solutions: Optional[list[str]] = None) -> None:
        self.problem_str = problem_str

        if public_tests is None:
            self.public_tests: list[Test] = []
        else:
            self.public_tests = public_tests

        if private_tests is None:
            self.private_tests: list[Test] = []
        else:
            self.private_tests = private_tests

        all_tests = self.public_tests + self.private_tests
        assert len(all_tests)
        prev_fn_name = all_tests[0].fn_name

        # Input validation
        for test in all_tests:
            assert isinstance(test, Test)
            assert prev_fn_name == test.fn_name
            prev_fn_name = test.fn_name
            if prev_fn_name is None:
                assert starter_code == "" or starter_code is None
                assert isinstance(test.input[0][0], str) and isinstance(test.output, str)
            else:
                assert starter_code is not None and len(starter_code)

        self.fn_name = prev_fn_name
        self.starter_code = starter_code
        self.solutions = solutions

    def convert_stdio_to_fn_input(self):
        assert self.fn_name is None
        self.fn_name = ""
        for test in self.public_tests:
            test.switch_fn_name(self.fn_name)
        for test in self.private_tests:
            test.switch_fn_name(self.fn_name)
   
    def has_starter_code(self):
        return not (self.starter_code == "" or self.starter_code is None)
    
    def get_starter_code_fn(self) -> Optional[str]:
        if self.starter_code == None or self.starter_code == "":
            return None
        
        starter_split = self.starter_code.splitlines()
        assert len(starter_split) == 3
        return starter_split[1].split("def ")[1].rstrip()
    
    def get_starter_code_fn_name(self) -> Optional[str]:
        starter_code_fn = self.get_starter_code_fn()
        if starter_code_fn is None:
            return None
        return starter_code_fn.split("(")[0]

    @staticmethod
    def from_coderm_item(question: str, starter_code: str, public_tests: Optional[dict[str, Any]], tests: Optional[dict[str, Any]], solutions: Optional[list[str]] = None) -> "Problem":
        assert (public_tests is not None) or (tests is not None)
        if public_tests is None:
            public_tests = {"fn_name": tests.get("fn_name", None), "inputs": None, "outputs": None}
        if tests is None:
            tests = {"fn_name": public_tests.get("fn_name", None), "inputs": None, "outputs": None}

        assert public_tests.get("fn_name", None) == tests.get("fn_name", None)
        fn_name = tests.get("fn_name", None)

        if public_tests.get("inputs", None) is None:
            assert public_tests.get("outputs", None) is None
            public_test_list = []
        else:
            assert len(public_tests["inputs"]) == len(public_tests["outputs"])
            public_test_list = [Test((wrap_list(inp), {}), out, fn_name) for inp, out in zip(public_tests["inputs"], public_tests["outputs"])]
        
        if tests.get("inputs", None) is None:
            assert tests.get("outputs", None) is None
            test_list = []
        else:
            assert len(tests["inputs"]) == len(tests["outputs"])
            test_list = [Test((wrap_list(inp), {}), out, fn_name) for inp, out in zip(tests["inputs"], tests["outputs"])]

        return Problem(question, starter_code=starter_code, public_tests=public_test_list, private_tests=test_list, solutions=solutions)


class SearchModel(BaseModel, ABC):
    def __init__(self, model_name: str, experiment_directory: str = None, cache_file: str = None, querier_batch_size: Optional[int] = 16384):
        super().__init__(model_name)
        self.experiment_directory = (experiment_directory if experiment_directory is not None 
                                     else f"logs/{datetime.datetime.now().strftime('%m%dT%H%M%S')}_{model_name}")
        self.querier = LLMQuerier(os.path.join(self.experiment_directory, "queries"), cache_file=cache_file, batch_size=querier_batch_size)

    def format_prompt(self, question: str, code: str = "", public_tests: Optional[dict[str, Any]] = None, tests: Optional[dict[str, Any]] = None, solutions: Optional[list[str]] = None) -> str | list[dict[str, str]]:
        print("Warning: `format_prompt` is misused right now.")
        return [{"problem_str": question, "starter_code": code, "public_tests": public_tests, "tests": tests, "solutions": solutions}]

    @abstractmethod
    def generate_solutions(self, problems: list[Problem], *args, **kwargs) -> list[str]:
        pass
    
    def generate_with_info(self, prompts: List[Prompt], **kwargs) -> List[Completion]:
        if len(kwargs):
            print("Warning: kwargs not used")
        problems = [Problem.from_coderm_item(prompt[0]["problem_str"], starter_code=prompt[0]["starter_code"], public_tests=prompt[0]["public_tests"], tests=prompt[0]["tests"], solutions=prompt[0]["solutions"]) for prompt in prompts]
        solutions = self.generate_solutions(problems)
        return [Completion(code, -1, -1) for code in solutions]
    
    def prefix_starter_code(self) -> bool:
        return False
