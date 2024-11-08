import os
import re
import argparse
import pandas as pd
import numpy as np
import spacy
import math
from pathlib import Path
from openai import OpenAI
import nltk
from nltk.corpus import stopwords
from collections import Counter
from sklearn.metrics import balanced_accuracy_score

from tasks import *

client = OpenAI(api_key=os.environ.get(""))
nlp = spacy.load('en_core_web_sm')
nltk.download('stopwords')
gpt_model = "gpt-4o"

stop_words = set(stopwords.words('english'))

# clean word string
def clean_word(word):
    word = word.strip()
    new_word = re.sub(r'[^a-zA-Z]+', '', word)
    return new_word

def clean_words_list(wordlist):
    return [clean_word(element) for element in wordlist]

# remove nan
def remove_nan(data_list):
    filtered_list = [item for item in data_list if item is not None and not (isinstance(item, float) and math.isnan(item))]
    return filtered_list

# call gpt for evaluation
def call_gpt(prompt, gpt_model):
    while True:
        try:
            response = client.chat.completions.create(
                model = gpt_model,
                messages = [
                    {"role": "user", "content": prompt}
                ],
                max_tokens = 100
            )

            gpt_output = response.choices[0].message.content
            break

        except Exception as e:
            print("Sleeping due to timeout.")
            time.sleep(300)

    return gpt_output

# for tasks needing exact match evaluation
def exact_match(model_predictions, gold_predictions):

    model_predictions = [item.lower() for item in model_predictions]
    gold_predictions = [item.lower() for item in gold_predictions]

    # create binary lists
    model_predictions_bin = []
    gold_predictions_bin = []

    for item in gold_predictions:
        gold_predictions_bin.append(1)

    # if item in gold prediction matches model prediction, 1 else 0
    print(len(gold_predictions))
    print(len(model_predictions))
    for i in range(len(gold_predictions)):
        if gold_predictions[i] in model_predictions[i]:
            model_predictions_bin.append(1)
        else:
            model_predictions_bin.append(0)

    # get exact match balanced accuracy
    score = balanced_accuracy_score(gold_predictions_bin, model_predictions_bin)
    return score

# get pos tag
def pos_tag(text):
  doc = nlp(text)
  result = [(token.text.lower(),token.pos_) for token in doc]
  return result

def evaluate_pos_correctness(model_generation_list, approved_pos_list, approved_word_list):
    approved_pos_pairs = []
    for alternate, alternate_pos in zip(approved_word_list,approved_pos_list):
        approved_pos_pairs.append((alternate, alternate_pos))

    model_generation_list_pos = []
    for sent in model_generation_list:
        model_generation_list_pos.append(pos_tag(sent))

    correct_hits = 0

    for pairs, tagged_sent in zip(approved_pos_pairs,model_generation_list_pos):
        if pairs in tagged_sent:
            correct_hits += 1

    score = correct_hits/len(model_generation_list)
    return score

def evaluate_definition_correctness(model_generation_list, approved_definition_list, approved_word_list):

    base_prompt = """
    Sentence: {{sentence}}
    Word: {{word}}
    Approved Definition: {{approved_definition}}

    Given the information above, judge if the given word is used in the sentence with respect to its approved definition. Answer directly with YES or NO. If the target word is not found in the sentence, answer NO. 

    """

    gpt_responses = []

    for sent, definition, word in zip(model_generation_list, approved_definition_list, approved_word_list):
        prompt = base_prompt.replace("{{sentence}}",sent)
        prompt = prompt.replace("{{word}}", word)
        prompt = prompt.replace("{{approved_definition}}",definition)

        #print(prompt)

        gpt_responses.append(call_gpt(prompt, gpt_model))

    gpt_responses = [item.lower() for item in gpt_responses]

    correct_hits = 0
    for item in gpt_responses:
        if 'yes' in item.lower():
            correct_hits += 1

    score = correct_hits/len(model_generation_list)
    return score

# for CEFR task 'identify_wrong_words_category'
def evaluate_wrong_words(model_predictions, gold_predictions):
    #model_predictions_splitted = [x.lower() for x in model_predictions]
    gold_predictions_splitted = [x.split(',') for x in gold_predictions]

    score = 0.0

    for model_predictions_text, gold_words_list in zip(model_predictions, gold_predictions_splitted):
        temp_score = sum(1 for item in gold_words_list if item in model_predictions_text.lower())
        temp_score = temp_score / len(gold_words_list)
        score += temp_score

    score = score / len(model_predictions)
    return score

# read CEFR wordlists 
def read_category_words(category):
    category_file = category + '.txt'
    category_path = 'data/oxford_wordlists_acc/' + category_file
    with open(Path(category_path), 'r', encoding='utf-8') as file:
        category_contents = file.read()
        category_contents_list = category_contents.split(",")
    return category_contents_list


# for CEFR task rewrite and generate
def evaluate_percentage(model_predictions, category_word_list, stop_words, n):

    #n = n-1 # allowance?
    score_list = []

    for text, category in zip(model_predictions, category_word_list):

        category_words = read_category_words(category.strip())
        #cleaned_words = [word.lower() for word in text.split() if word.lower() not in stop_words]
        cleaned_words = [word.lower() for word in text.split()]
        cleaned_words = clean_words_list(cleaned_words)
        category_words = clean_words_list(category_words)

        temp_score = sum(1 for word in cleaned_words if word.lower() in category_words)
        temp_score = temp_score / len(cleaned_words)
        score_list.append(temp_score)

    #score = sum(1 for x in score_list if x >= n)
    score = sum(score_list) / len(score_list)
    return score

# EVALUATION DIRECTORY
def evaluate(task, model_task_generation_list, task_test_df):
    if 'check' in task or 'identify' in task:
        gold_predictions = task_test_df['answer'].tolist()
        gold_predictions = remove_nan(gold_predictions)
        model_task_generation_list = remove_nan(model_task_generation_list)
        score = exact_match(model_task_generation_list, gold_predictions)

    elif 'rewrite' in task and 'definition' in task and 'pos' in task:
        approved_pos_list = task_test_df['alternative_approved_pos'].tolist()
        approved_alternative_list = task_test_df['alternative'].tolist()
        model_task_generation_list = remove_nan(model_task_generation_list)
        approved_pos_list = remove_nan(approved_pos_list)
        approved_alternative_list = remove_nan(approved_alternative_list)
        pos_score = evaluate_pos_correctness(model_task_generation_list, approved_pos_list, approved_alternative_list)

        approved_definition_list = task_test_df['approved_definition'].tolist()
        approved_word_list = task_test_df['alternative'].tolist()
        approved_definition_list = remove_nan(approved_definition_list)
        approved_word_list = remove_nan(approved_word_list)
        def_score = evaluate_definition_correctness(model_task_generation_list, approved_definition_list, approved_word_list)

        score = (pos_score+def_score)/2

    elif 'rewrite' in task and 'definition' in task:
        approved_definition_list = task_test_df['approved_definition'].tolist()
        approved_word_list = task_test_df['word'].tolist()
        model_task_generation_list = remove_nan(model_task_generation_list)
        approved_definition_list = remove_nan(approved_definition_list)
        approved_word_list = remove_nan(approved_word_list)
        score = evaluate_definition_correctness(model_task_generation_list, approved_definition_list, approved_word_list)

    elif 'rewrite' in task and 'pos' in task:
        approved_pos_list = task_test_df['alternative_approved_pos'].tolist()
        approved_alternative_list = task_test_df['alternative'].tolist()
        model_task_generation_list = remove_nan(model_task_generation_list)
        approved_pos_list = remove_nan(approved_pos_list)
        approved_alternative_list = remove_nan(approved_alternative_list)
        score = evaluate_pos_correctness(model_task_generation_list, approved_pos_list, approved_alternative_list)

    elif 'generate' in task and 'definition' in task and 'pos' in task:
        approved_pos_list = task_test_df['approved_word_pos'].tolist()
        approved_word_list = task_test_df['word'].tolist()
        model_task_generation_list = remove_nan(model_task_generation_list)
        approved_pos_list = remove_nan(approved_pos_list)
        approved_word_list = remove_nan(approved_word_list)
        pos_score = evaluate_pos_correctness(model_task_generation_list, approved_pos_list, approved_word_list)

        approved_definition_list = task_test_df['approved_definition'].tolist()
        approved_definition_list = remove_nan(approved_definition_list)
        approved_word_list = remove_nan(approved_word_list)
        def_score = evaluate_definition_correctness(model_task_generation_list, approved_definition_list, approved_word_list)

        score = (pos_score+def_score)/2

    elif 'generate' in task and 'pos' in task:
        approved_pos_list = task_test_df['approved_word_pos'].tolist()
        approved_word_list = task_test_df['word'].tolist()
        model_task_generation_list = remove_nan(model_task_generation_list)
        approved_pos_list = remove_nan(approved_pos_list)
        approved_word_list = remove_nan(approved_word_list)
        score = evaluate_pos_correctness(model_task_generation_list, approved_pos_list, approved_word_list)

    elif 'generate' in task and 'definition' in task:
        approved_word_list = task_test_df['word'].tolist()
        approved_definition_list = task_test_df['approved_definition'].tolist()
        model_task_generation_list = remove_nan(model_task_generation_list)
        approved_word_list = remove_nan(approved_word_list)
        approved_definition_list = remove_nan(approved_definition_list)
        score = evaluate_definition_correctness(model_task_generation_list, approved_definition_list, approved_word_list)

    elif 'check' in task and 'category' in task:
        gold_predictions = task_test_df['answer'].tolist()
        gold_predictions = remove_nan(gold_predictions)
        model_task_generation_list = remove_nan(model_task_generation_list)
        score = exact_match(model_task_generation_list, gold_predictions)

    elif 'identify' in task and 'correct_category' in task:
        gold_predictions = task_test_df['answer'].tolist()
        gold_predictions = remove_nan(gold_predictions)
        model_task_generation_list = remove_nan(model_task_generation_list)
        score = exact_match(model_task_generation_list, gold_predictions)

    elif 'identify' in task and 'wrong_words' in task:
        gold_predictions = task_test_df['answer'].tolist()
        gold_predictions = remove_nan(gold_predictions)
        model_task_generation_list = remove_nan(model_task_generation_list)
        score = evaluate_wrong_words(model_task_generation_list, gold_predictions)

    elif ('generate' in task or 'rewrite' in task) and ('category' in task):
        category_word_list = task_test_df['category'].tolist()
        model_task_generation_list = remove_nan(model_task_generation_list)
        if '95' in task:
            n = 0.95
        else:
            n = 1.00
        score = evaluate_percentage(model_task_generation_list, category_word_list, stop_words, n)

    return score

# read model generation file
model_generation_file = input("Enter model generation file:")
model_generation_file = str(model_generation_file)

# CHANGE FOR STE OR CEFR
model_generation_path = Path('results/cleaned/shot/' + model_generation_file)
#model_generation_path = Path('results/cleaned/cefr/' + model_generation_file)

model_generation_df = pd.read_csv(model_generation_path)

# selected task
SELECTED_TASKS = [
    "rewrite_text_correct_definition",
    "rewrite_text_correct_pos_and_definition"
]

# iterate over all task
model_score_dict = {}
for task in SELECTED_TASKS:
    print(task)
    task_path = 'tasks/' + task + '/' + 'test.xlsx'
    task_test_df = pd.read_excel(Path(task_path))

    model_task_generation = model_generation_df[task].tolist()
    model_score_dict[task] = evaluate(task, model_task_generation, task_test_df)

print(model_score_dict)
score_df = pd.DataFrame(model_score_dict,index=[0])
output_file = model_generation_file.replace('.csv','_scores.csv')
score_df.to_csv(output_file, index=False)