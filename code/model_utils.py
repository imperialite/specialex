# This code is derived from https://github.com/AI21Labs/in-context-ralm/blob/main/ralm/model_utils.py

import logging
import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

def load_tokenizer(model_name, access_token):
    return AutoTokenizer.from_pretrained(model_name, token=access_token)

def load_model_and_tokenizer(model_name, access_token, quantize):
    
    device = torch.device("cuda")
    device_count = torch.cuda.device_count()
    print(device_count)

    quantization_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)

    if "flan" in model_name or "coedit" in model_name or "T0" in model_name:
        if quantize == True:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, attn_implementation="flash_attention_2", torch_dtype=torch.float16, quantization_config=quantization_config, device_map="auto", token=access_token)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", token=access_token)
    else:
        if quantize == True:
            model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="flash_attention_2", torch_dtype=torch.float16, quantization_config=quantization_config, device_map="auto", token=access_token)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", token=access_token)
    
    #model = torch.nn.DataParallel(model)
    tokenizer = load_tokenizer(model_name, access_token)

    return model, tokenizer, device

logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter("[%(thread)s] %(asctime)s [%(levelname)s] %(name)s: %(message)s")
console = logging.StreamHandler()
console.setFormatter(log_formatter)
logger.addHandler(console)

def print_args(args):
    logger.info(" **************** CONFIGURATION **************** ")
    for key, val in sorted(vars(args).items()):
        keystr = "{}".format(key) + (" " * (30 - len(key)))
        logger.info("%s -->   %s", keystr, val)
    logger.info(" **************** CONFIGURATION **************** ")