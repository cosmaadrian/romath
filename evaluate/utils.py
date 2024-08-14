from copy import deepcopy
import string


def complete_prompts(template, **kwargs) -> list:
    current_messages = deepcopy(template)
    for i in range(len(current_messages)):
        if isinstance(current_messages[i]['content'], str):
            try:
                current_messages[i]['content'] = current_messages[i]['content'].format(**kwargs)
            except KeyError:
                variables = get_variables_from_fstring(current_messages[i]['content'])
                not_provided = [var for var in variables if var not in kwargs]
                raise ValueError(f"Missing required arguments: {not_provided} when calling the LLMAgent.")
        elif isinstance(current_messages[i]['content'], list):
            for j in range(len(current_messages[i]['content'])):
                try:
                    current_messages[i]['content'][j]['content'] = current_messages[i]['content'][j][
                        'content'].format(**kwargs)
                except KeyError:
                    variables = get_variables_from_fstring(current_messages[i]['content'][j]['content'])
                    not_provided = [var for var in variables if var not in kwargs]
                    raise ValueError(f"Missing required arguments: {not_provided} when calling the LLMAgent.")
    return current_messages

@staticmethod
def get_variables_from_fstring(fstring):
    formatter = string.Formatter()
    return [name for _, name, _, _ in formatter.parse(fstring) if name is not None]

