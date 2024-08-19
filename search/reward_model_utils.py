import torch
from torch.nn.parallel import DataParallel as DP
from transformers import AutoTokenizer
from openrlhf.models import get_llm_for_sequence_regression
from tqdm import tqdm

from typing import Optional

from coderm.prompts import py_prompt
from search.python_utils import chunk

import argparse
import os


class RewardModel:
    def __init__(self, model_path: str, max_tokens: int = 4096, batch_size: Optional[int] = 8) -> None:
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.batch_size = batch_size

        self.model = None
        self.tokenizer = None
        self.is_loaded = False

    def load_model(self):
        if self.is_loaded:
            return

        self.model = get_llm_for_sequence_regression(self.model_path, "reward", normalize_reward=True, bf16=True,).cuda()
        if torch.cuda.device_count() > 1:
            self.model = DP(self.model)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.is_loaded = True

    def get_scores(self, questions: list[str], codes: list[str]) -> list[float]:
        if not self.is_loaded:
            self.load_model()
        assert self.model is not None
        assert self.tokenizer is not None
        assert len(questions) == len(codes)

        codes_with_context = [py_prompt(question, code) for question, code in zip(questions, codes)]
        
        if self.batch_size is None:
            batched_texts = [codes_with_context]
        else:
            batched_texts = list(chunk(codes_with_context, self.batch_size))

        scores = []
        with tqdm(total=len(codes_with_context)) as pbar:
            for text_batch in batched_texts:
                with torch.no_grad():
                    input_tokens = self.tokenizer(
                        text_batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.max_tokens,
                        ).to("cuda")

                    batch_scores = self.model(**input_tokens)

                    assert batch_scores.shape == (len(text_batch),)
                    scores.extend(batch_scores.detach().cpu().tolist())
                    pbar.update(len(text_batch))

        return scores

if __name__ == "__main__":
    REW_PATH = "/mnt/efs/evanwang/model_weights/reward/star"
    rm = RewardModel(REW_PATH)
    LENS = 30000
    print("SCORES: ", rm.get_scores(["Given a list of integers, take their sum.", "Given a list of integers, take their sum."] * (LENS // 2), ["def sum_list(l):\n    return sum(l)", "bruh hello?"] * (LENS // 2)))

