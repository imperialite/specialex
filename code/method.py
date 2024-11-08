import pandas as pd
from pathlib import Path
import torch
from accelerate import Accelerator
from openai import OpenAI
import os
import time
import requests

torch.cuda.empty_cache()

# Call gpt for evaluation
def call_gpt(prompt, gpt_model):

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    while True:
        try:
            response = client.chat.completions.create(
                model = gpt_model,
                messages = [
                    {"role": "user", "content": prompt}
                ],
                temperature = 0.0
            )

            gpt_output = response.choices[0].message.content
            break

        except Exception as e:
            print("Sleeping due to timeout.")
            time.sleep(300)

    return gpt_output

# Function to replace tags with dataframe values
def process_template(row, template):
    for column in row.index:
        tag = '{{' + column + '}}'

        # Check if column name is in prompt template
        if tag in template:
        	template = template.replace(tag, str(row[column]))
    return template

# Prompt formatter, takes prompt template (txt) and test file (dataframe)
def prompt_preparation(task, test_file_name, prompt_file_name):

	prompt_list = []

	task_path = Path('tasks/' + task)
	prompt_template_path = task_path / prompt_file_name
	test_file_path = task_path / test_file_name

	print("Prompt template path:",prompt_template_path)
	print("Test template path:",test_file_path)

	file = open(prompt_template_path, "r")
	prompt_template = file.read()
	file.close()
	test_file_df = pd.read_excel(test_file_path)

	# Applying the template processing for each row
	for index, row in test_file_df.iterrows():
	    prompt_list.append(process_template(row, prompt_template))

	return prompt_list

def icl_generate(prompt_list, model, tokenizer, device, max_length, top_p, model_api_url, accelerate):

	model_generations = []
	counter = 0

	# Iterate over prepared prompts
	for prompt in prompt_list:

		# Format prompts
		if "llama-3" in model_api_url.lower() or "qwen" in model_api_url.lower():
			messages = [
				{"role": "system", "content": "You are a powerful AI that can follow complex instructions, answer questions, perform tasks, and learn patterns of output if a few examples are given. You provide your responses directly without justifying your output."},
				{"role": "user", "content": prompt}
			]
			input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)

		if "mistral" in model_api_url.lower() or "cohere" in model_api_url.lower() or "allen" in model_api_url.lower() or "gemma" in model_api_url.lower():
			messages = [
				{"role": "user", "content": prompt}
			]
			input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

		else:
			input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
		
		# Generation part
		model.eval()
		with torch.no_grad():
			output_undecoded = model.generate(
				input_ids,
				max_new_tokens = max_length,
				do_sample=False
			)

		# Llama 3 specific output
		if "llama-3" in model_api_url.lower():
			response = output_undecoded[0][input_ids.shape[-1]:]
			output_decoded = tokenizer.decode(response, skip_special_tokens=True)
			model_generations.append(output_decoded)
		
		else:
			prompt_length = len(tokenizer.encode(prompt))
			output_decoded = tokenizer.decode(output_undecoded[0][prompt_length:], skip_special_tokens=True)
			model_generations.append(output_decoded)

		counter += 1
		print(counter)

		# clear cache
		torch.cuda.empty_cache()

	return model_generations

def icl_openai(prompt_list, model):

	model_generations = []
	counter = 0

	# Iterate over prepared prompts
	for prompt in prompt_list:

		response = call_gpt(prompt, model)
		model_generations.append(response)

		counter += 1
		print(counter)

	return model_generations

# HF Inference API
def icl_hf_api(prompt_list, API_URL, api_token):

	api_token = "Bearer " + api_token
	headers = {"Authorization": api_token}

	def query(payload):
		print(headers)
		print(API_URL)
		response = requests.post(API_URL, headers=headers, json=payload)
		return response.json()

	model_generations = []
	counter = 0

	# Iterate over prepared prompts
	for prompt in prompt_list:

		while True:
			try:
				response = query({
				'inputs': prompt,  # Your input prompt
				'parameters': {
					'do_sample': False,
					'return_full_text': False,
					'temperature': 0.01
					}
				})
				model_generations.append(response[0]['generated_text'])
				break

			except KeyError as e:
				print("Sleeping due to timeout.")
				time.sleep(600)

			except Exception as e:
				print("Sleeping due to timeout.")
				time.sleep(600)
		
		counter += 1
		print(counter)

	return model_generations