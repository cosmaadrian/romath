import re
import torch
import pprint
import transformers
from transformers import pipeline
from transformers import AutoModelForCausalLM
import re

from evaluate.math_equivalence import is_equivalent

from .utils import complete_prompts

from evaluate.prompts.judge_english_prompt import PROMPT as ENGLISH_PROMPT
from evaluate.prompts.judge_romanian_prompt import PROMPT as ROMANIAN_PROMPT

class SolutionJudge:
    """
        A class checks the correctness of a solution to a problem.
        It uses either `math_equivalence.py` for solutions that are not proofs (have a single final answer)
            or using a target LLM for judging the correctness of proofs.

        Problems that have single answers should have their generated final answers contained in a boxed format (`\boxed{<answer>}`), otherwise it will check the whole output.
    """
    def __init__(self, model_name, prompt):
        super().__init__()
        self.model_name = model_name

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map = "auto",
            torch_dtype = torch.float16,
            trust_remote_code = True
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        self.text_generator = pipeline("text-generation", model = self.model, tokenizer = self.tokenizer)

        self.template = ENGLISH_PROMPT if prompt == 'en' else ROMANIAN_PROMPT if prompt == 'ro' else None

    def evaluate(self, question: str, true: str, prediction: str, has_single_answer: bool = False) -> int:
        if has_single_answer:
            if r'\boxed{' in prediction:
                prediction = re.sub(r'\\boxed\{(.+?)\}', r'\1', prediction)
                return int(is_equivalent(true, prediction))

            # if the prediction does not contain the boxed format, we will check the whole output
        messages = complete_prompts(self.template, question = question, true = true, prediction = prediction)
        response = self.text_generator(
            messages,
            do_sample = False, max_new_tokens = 32, temperature = None, top_k = None, top_p = None
        )

        content = response[0]['generated_text'][-1]['content']
        # pprint.pprint(content)

        try:
            score = int(re.search(r'\d', content).group())
            if score != 1 and score != 0:
                raise Exception(score)

            return score
        except Exception as e:
            print(f"Exception: {e}")
            print(f"WARNING: Could not extract score from the response. ({content})")
            return -1

if __name__ == "__main__":
    judge = SolutionJudge(model_name = "Qwen/Qwen2-1.5B-Instruct")
    response = judge.evaluate(question="What is the sum of 2 and 3?", true = "5", prediction = "4", has_single_answer = False)
    print(response)