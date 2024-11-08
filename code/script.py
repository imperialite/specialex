import os
import argparse
import pandas as pd
import numpy as np
import requests
import torch

# External script files
from model_utils import *
from tasks import *
from method import *

from huggingface_hub import login

def evaluate(
	task_group_list,
	file_output_name,
	test_file_name,
	prompt_file_name,
	model_api_url,
	max_length,
	top_p,
	inference_method,
	access_token_read,
	quantize,
	accelerate
	):

	# ICL METHOD
	if inference_method == 'icl':

		task_generations = {}

		# Load model
		model, tokenizer, device = load_model_and_tokenizer(
			model_name = model_api_url,
			access_token = access_token_read,
			quantize = quantize
		)

		print("FINISH LOADING MODELS")

		for task in task_group_list:

			# Prepare prompt list for model
			prompt_list = prompt_preparation(task, test_file_name, prompt_file_name)

			# Inference
			task_output = icl_generate(
				prompt_list,
				model,
				tokenizer,
				device,
				max_length,
				top_p,
				model_api_url,
				accelerate
				)

			task_generations[task] = task_output

		
		# Populate NaNs for to list so they can be exported together
		max_length = max(len(lst) for lst in task_generations.values())
		for key, lst in task_generations.items():
			task_generations[key] = lst + [np.nan] * (max_length - len(lst))

		# Export dictionary to csv
		task_generations_df = pd.DataFrame(task_generations)
		task_generations_df.to_csv(file_output_name, index='False')


	elif inference_method == 'icl-openai':

		task_generations = {}

		# Load model
		model = model_api_url

		print("FINISH LOADING MODELS")

		for task in task_group_list:

			# Prepare prompt list for model
			prompt_list = prompt_preparation(task, test_file_name, prompt_file_name)

			# Inference
			task_output = icl_openai(
				prompt_list,
				model
				)

			task_generations[task] = task_output

		
		# Populate NaNs for to list so they can be exported together
		max_length = max(len(lst) for lst in task_generations.values())
		for key, lst in task_generations.items():
			task_generations[key] = lst + [np.nan] * (max_length - len(lst))

		# Export dictionary to csv
		task_generations_df = pd.DataFrame(task_generations)
		task_generations_df.to_csv(file_output_name, index='False')

	
	elif inference_method == 'icl-hf-api':

		task_generations = {}

		print("FINISH LOADING MODELS")

		for task in task_group_list:

			# Prepare prompt list for model
			prompt_list = prompt_preparation(task, test_file_name, prompt_file_name)

			# Inference
			task_output = icl_hf_api(
				prompt_list,
				model_api_url,
				access_token_read
				)

			task_generations[task] = task_output

		
		# Populate NaNs for to list so they can be exported together
		max_length = max(len(lst) for lst in task_generations.values())
		for key, lst in task_generations.items():
			task_generations[key] = lst + [np.nan] * (max_length - len(lst))

		# Export dictionary to csv
		task_generations_df = pd.DataFrame(task_generations)
		task_generations_df.to_csv(file_output_name, index='False')

	# RAG w/ PLAN METHOD
	#elif inference_method == 'rag-plan':
	#	print("TODO")


def main(args):
	# Dump args
    print_args(args) 
    
    # Login Huggingface
    access_token_read = args.auth_token
    login(token = access_token_read)

    data_group = args.data_group
    task_group = args.task_group

    # Task and data selection
    if data_group == 'ste' and task_group  == 'checking':
    	task_group_list = STE_CHECK_TASKS
    elif data_group == 'ste' and task_group  == 'identification':
    	task_group_list = STE_IDENTIFICATION_TASKS
    elif data_group == 'ste' and task_group  == 'rewriting':
    	task_group_list = STE_REWRITING_TASKS
    elif data_group == 'ste' and task_group  == 'generation':
    	task_group_list = STE_GENERATION_TASKS
    elif data_group == 'ste' and task_group  == 'all':
    	task_group_list = STE_TASKS
    elif data_group == 'vxgl' or data_group  == 'cefr' and task_group  == 'all':
    	task_group_list = VXGL_CEFR_TASKS

    # Test file selection
    if data_group == 'ste':
    	test_file_name = 'test.xlsx'
    elif data_group == 'cefr' or data_group == 'vxgl':
    	test_file_name = 'test.xlsx' # use default while vxgl data is not ready yet

    prompt_file_name = args.prompt_file_name
    model_api_url = args.model_api_url
    max_length = args.max_length
    top_p = args.top_p
    inference_method = args.method
    quantize = args.quantize
    accelerate = args.accelerate

    # Name file output
    if '/' in model_api_url:
    	model_short_name = model_api_url.split('/')[-1]
    else:
    	model_short_name = model_api_url

    file_output_name = str(model_short_name) + '_' + str(task_group) + '_' + str(data_group) + '_' + str(inference_method)
    if 'enriched' in prompt_file_name:
    	file_output_name = file_output_name + '_enriched'
    file_output_name = file_output_name + '.csv'
    print("File output name:",file_output_name)

    evaluate(
        task_group_list,
        file_output_name,
        test_file_name,
        prompt_file_name,
        model_api_url,
        max_length,
        top_p,
        inference_method,
        access_token_read,
        quantize,
        accelerate
    )

if __name__ == '__main__':

	# Clear cache
	torch.cuda.empty_cache()

	parser = argparse.ArgumentParser()

	# Setup params
	parser.add_argument("--task_group", type=str, default="checking") # checking, identification, generation, rewriting
	parser.add_argument("--data_group", type=str, default="ste") # ste, vxgl, cambridge

	# File params
	parser.add_argument("--prompt_file_name", type=str, default="base_prompt.txt")

	# Model params
	parser.add_argument("--model_api_url", type=str, required=True)
	parser.add_argument("--max_length", type=int, default=300)
	parser.add_argument("--top_p", type=float, default=1.0)
	parser.add_argument("--auth_token", type=str, default="")
	parser.add_argument("--quantize", type=bool, default=True)
	parser.add_argument("--accelerate", type=bool, default=True)


	# Method param
	parser.add_argument("--method", type=str, default="icl")

	args = parser.parse_args()
	main(args)