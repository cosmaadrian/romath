import ollama
import string
from copy import deepcopy

if __name__ == "__main__":
    from math_equivalence import is_equivalent
else:
    from evaluate.math_equivalence import is_equivalent

import re

class SolutionJudge:
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

        self.llm_client = ollama.Client(host = 'http://localhost:11434')
        self.template = EVALUATION_PROMPT

    def evaluate(self, question: str, true: str, prediction: str, has_single_answer: bool = False) -> int:
        if has_single_answer: # TODO probably need to change this to a more robust check / extract the \boxed{} part, etc.
            if is_equivalent(true, prediction):
                return 1
            return 0

        messages = self._complete_prompts(question = question, true = true, prediction = prediction)

        response = self.llm_client.chat(
            model = self.model_name,
            messages = messages
        )

        content = response['message']['content'].strip()

        try:
            score = int(re.findall(r'\d+', content)[0])
            return score
        except Exception as e:
            print(f"WARNING: Could not extract score from the response. ({content})")
            return -1

    def _complete_prompts(self, **kwargs) -> list:
        if type(self.template) is list:
            current_messages = deepcopy(self.template)
            for i in range(len(current_messages)):

                if isinstance(current_messages[i]['content'], str):
                    try:
                        current_messages[i]['content'] = current_messages[i]['content'].format(**kwargs)
                    except KeyError:
                        variables = self.get_variables_from_fstring(current_messages[i]['content'])
                        not_provided = [var for var in variables if var not in kwargs]
                        raise ValueError(f"Missing required arguments: {not_provided} when calling the LLMAgent.")
                elif isinstance(current_messages[i]['content'], list):
                    for j in range(len(current_messages[i]['content'])):
                        try:
                            current_messages[i]['content'][j]['content'] = current_messages[i]['content'][j][
                                'content'].format(**kwargs)
                        except KeyError:
                            variables = self.get_variables_from_fstring(current_messages[i]['content'][j]['content'])
                            not_provided = [var for var in variables if var not in kwargs]
                            raise ValueError(f"Missing required arguments: {not_provided} when calling the LLMAgent.")
            return current_messages

    @staticmethod
    def get_variables_from_fstring(fstring):
        formatter = string.Formatter()
        return [name for _, name, _, _ in formatter.parse(fstring) if name is not None]


# TODO make it in Romanian
EVALUATION_PROMPT = [
    {
        "role": "system",
        "content":
        """
        Assume the role of a math teacher tasked with evaluating student responses against the provided solutions, which may include exact values, multiple-choice answers, or numerical approximations. The question is provided as: {question}, the correct answer is provided as: {true}.

        ## Evaluation Criteria:
        1. **Mathematical Equivalence**: Evaluate answers based on deep mathematical equivalence, not just numerical accuracy. Use advanced tools or techniques to verify if different algebraic or symbolic expressions are equivalent. Tools like symbolic computation software (e.g., Wolfram Alpha, SymPy) should be used to confirm equivalences such as \\( \\frac{{\\sqrt{{6}}-\\sqrt{{2}}}}{{2}} \\) being equivalent to \\( \\sqrt{{2 - \\sqrt{{3}}}} \\).
        2. **Scoring**: Assign a score of '1' for any answer that matches or is equivalent to the provided solution, whether it is an exact value, a choice label (e.g., A, B, C), or a correctly rounded numerical approximation. Assign a score of '0' for incorrect answers. Do not provide any explanatory feedback in your evaluation.
        3. **Handling Multiple Choices**: If the solution provided is a choice (e.g., A, B, C, D, E, F) and the student identifies this choice correctly, treat it as correct. If the solution is an exact value and the student provides the corresponding choice that reflects this value correctly according to the problem's context, also treat it as correct.
        4. **Numerical Equivalence**: Treat numerical answers as equivalent if they are correct to at least two decimal places or more, depending on the precision provided in the solution. For instance, both 0.913 and 0.91 should be accepted if the solution is accurate within two decimal places.
        5. **Symbolic and Algebraic Identities**: Recognize and accept equivalent algebraic forms, such as \\( \\sin^2(x) + \\cos^2(x) = 1 \\) or \\( e^{{i\\pi}} + 1 = 0 \\), as correct.
        6. **Trigonometric and Logarithmic Forms**: Accept equivalent trigonometric and logarithmic expressions, acknowledging identities and transformations that might alter the form but not the value.
        7. **Comprehensive Evaluation**: Encourage the use of computational tools to check for equivalence in cases where expressions are too complex for straightforward visual inspection.

        ## Expected Output Format:
            Present your final answer with a score of '1' or '0' only. Do not include any additional information or feedback in your response.

        Please evaluate the student's response with precision to ensure accurate and fair grading.
        """
    },
    {"role": "user", "content":
    """
    The student answer is {prediction}.
    """
    }
]

if __name__ == "__main__":
    judge = SolutionJudge(model_name = 'phi3')
    response = judge.evaluate(question="What is the sum of 2 and 2?", true = "4", prediction = "4", has_single_answer = False)
    print(response)