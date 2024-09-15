from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pandas as pd
import argparse
from tqdm import tqdm
import os
import re
import string

TOKEN = os.environ.get('HF_TOKEN', None)

def translate_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors = "pt").to("cuda")
    translated_tokens = model.generate(**inputs, forced_bos_token_id = tokenizer.convert_tokens_to_ids("eng_Latn"))
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens = True)[0]

def translate_original(text, tokenizer, model):
    text = text.replace("\\n", "\n")
    sentences = re.split(r'(?<=[.\n])', text)

    translated_text = ''
    for sentence in sentences:
        if sentence.replace(r"\s+", "") != '' and sentence.strip() != '.' and sentence.translate(str.maketrans('', '', string.punctuation)).strip() != '':
            try:
                math = re.search(r'\\\([\s\S]*?\\\)|\\\\\([\s\S]*?\\\)|\\\[[\s\S]*?\\\]|\\\\\[[\s\S]*?\\\]', sentence).group()
                if math != sentence and math.strip() != sentence.strip():
                    translated_text += translate_text(sentence, tokenizer, model)
                else:
                    translated_text += sentence
            except:
                translated_text += translate_text(sentence, tokenizer, model)
        else:
            translated_text += sentence

    return translated_text

def translate_unchanged_math(text, tokenizer, model):
    text = text.replace("\\n", "\n")
    # translate text, by keeping numbers and special characters between \( and \) unchanged
    # get indices of special characters
    special = []
    for m in re.finditer(r'\\\([\s\S]*?\\\)|\\\\\([\s\S]*?\\\)|\\\[[\s\S]*?\\\]|\\\\\[[\s\S]*?\\\]', text):
        special.append((m.start(), m.end()))

    # translate text that is not special, and keep special unchanged
    translated_text = ''
    start = 0

    for s, e in special:
        if text[start:s].replace(r"\s+", "").replace('\n', "") != '' and text[start:s].strip() not in string.punctuation:
            translated_sentence = translate_original(text[start:s], tokenizer, model)

            if translated_sentence.strip() != '' and translated_sentence.strip() != '.':
                # if text[start:s] starts or ends with period, add period to translated text
                if text[start:s].strip()[-1] == '.' and translated_sentence.strip()[-1] != '.':
                    translated_sentence += '.'

                if text[start:s].strip()[-1] != '.' and translated_sentence.strip()[-1] == '.':
                    translated_sentence = translated_sentence.strip()[:-1]

                if text[start:s].strip()[0] == '.' and translated_sentence.strip()[0] != '.':
                    translated_sentence = '.' + translated_sentence

                if text[start:s].strip()[0] != '.' and translated_sentence.strip()[0] == '.':
                    translated_sentence = translated_sentence.strip()[1:]

            translated_text += translated_sentence
        else:
            translated_text += text[start:s]

        translated_text += text[s:e]
        start = e

    if text[start:].replace(r"\s+", "").replace('\n', "") != '' and text[start:].strip() not in string.punctuation:
        translated_sentence = translate_original(text[start:], tokenizer, model)

        if translated_sentence.strip() != '' and translated_sentence.strip() != '.':

            if text[start:].strip()[-1] == '.' and translated_sentence.strip()[-1] != '.':
                translated_sentence += '.'

            if text[start:].strip()[-1] != '.' and translated_sentence.strip()[-1] == '.':
                translated_sentence = translated_sentence.strip()[:-1]

            if text[start:].strip()[0] == '.' and translated_sentence.strip()[0] != '.':
                translated_sentence = '.' + translated_sentence

            if text[start:].strip()[0] != '.' and translated_sentence.strip()[0] == '.':
                translated_sentence = translated_sentence.strip()[1:]

        translated_text += translated_sentence
    else:
        translated_text += text[start:]

    return translated_text

def translate_math_token(text, tokenizer, model):
    text = text.replace("\\n", "\n")
    # translate text, by replacing numbers and special characters between \( and \) with [MATH]
    # get indices of special characters
    special = []
    for m in re.finditer(r'\\\([\s\S]*?\\\)|\\\\\([\s\S]*?\\\)|\\\[[\s\S]*?\\\]|\\\\\[[\s\S]*?\\\]', text):
        special.append((m.start(), m.end()))

    # replace special characters with [MATH]
    text_with_math_token = ''
    start = 0
    for s, e in special:
        text_with_math_token += text[start:s]
        text_with_math_token += '[MATH]'
        start = e

    text_with_math_token += text[start:]

    # translate sentence by sentence
    sentences = re.split(r'(?<=[.\n])', text_with_math_token)

    translated_text = ''
    for sentence in sentences:
        if sentence.replace(r"\s+", "") != '' and sentence.strip().replace("[MATH]", "").strip() != '' and sentence.strip().replace("[MATH]", "").strip() != '.'  and sentence.strip() != '.':
            translated_text += translate_text(sentence, tokenizer, model)
        else:
            translated_text += sentence


    translated_text = translated_text.strip()
    # replace [MATH] with special characters
    math = []
    for m in re.finditer(r'\[MATH\]|\[math\]', translated_text):
        math.append((m.start(), m.end()))

    translated_text_with_math = ''
    start = 0
    for (s_math, e_math), (s_spec, e_spec) in zip(math, special):
        translated_text_with_math += translated_text[start:s_math]
        translated_text_with_math += text[s_spec:e_spec]
        start = e_math

    translated_text_with_math += translated_text[start:]

    return translated_text_with_math

def translate_romath(dataset_df, model_name):
    dataset_df = dataset_df.copy()
    tokenizer = AutoTokenizer.from_pretrained(model_name, token = TOKEN, src_lang = "ron_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token = TOKEN).to("cuda")

    for i, row in tqdm(dataset_df.iterrows(), total = len(dataset_df)):
        dataset_df.loc[i, 'translated_problem'] = translate_original(row['problem'].strip(), tokenizer, model)
        dataset_df.loc[i, 'translated_solution'] = translate_original(row['solution'].strip(), tokenizer, model)

        # translate text, by keeping numbers and special characters between \( and \) unchanged
        dataset_df.loc[i, 'translated_problem_unchanged_math'] = translate_unchanged_math(row['problem'].strip(), tokenizer, model)
        dataset_df.loc[i, 'translated_solution_unchanged_math'] = translate_unchanged_math(row['solution'].strip(), tokenizer, model)

        # translate text, by replacing numbers and special characters between \( and \) with [MATH]
        dataset_df.loc[i, 'translated_problem_math_token'] = translate_math_token(row['problem'].strip(), tokenizer, model)
        dataset_df.loc[i, 'translated_solution_math_token'] = translate_math_token(row['solution'].strip(), tokenizer, model)

    return dataset_df

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Annotate answer')
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--model', type=str, help='Model name')

    args = parser.parse_args()

    dataset_df = pd.read_csv(args.dataset)
    dataset_df = translate_romath(dataset_df, model_name = args.model)

    os.makedirs('translated', exist_ok = True)
    basename = os.path.basename(args.dataset)

    dataset_df.to_csv(f'translated/{basename.split(".csv")[0]}_model_{args.model.split("/")[-1]}.csv', index=False)
