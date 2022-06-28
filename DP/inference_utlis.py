import torch
import torch.nn as nn

import os
import random
import time
import json
import numpy as np
import nltk
import os
import sys
import random
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import operator
from operator import itemgetter
import progressbar
import argparse

import re
def restore_text(text, mode):
    if mode == 'bs':
        text = re.sub(' is ', ' ', text)
    elif mode == 'da':
        pass
    else:
        raise Exception('Wrong Restore Mode!!!')
    text = re.sub(' , ', ' ', text)
    text = ' '.join(text.split()).strip()
    return text

def erase_error(text):
    # [value - -> [value-
    # [ value -> [value
    # [value- -> [value_
    text = re.sub(r"\[value -", r"[value\-", text)
    text = re.sub(r"\[ value", r"[value", text)
    text = re.sub(r"\[value-", r"[value_", text)
    return text

def batch_generate(model, one_inference_batch, data):
    is_cuda = next(model.parameters()).is_cuda
    if is_cuda: 
        #device = next(model.parameters()).device
        device = torch.device('cuda')
        if torch.cuda.device_count() > 1: # multi-gpu training 
            model = model.module
        else: # single gpu training
            pass
    else:
        device = 0

    max_span_len, max_response_len = 80, 120
    tokenizer = data.tokenizer
    da_batch, parse_dict_batch = one_inference_batch
    batch_size = len(parse_dict_batch)
    res_batch_parse_dict = parse_dict_batch

    # we need to generate the dialogue action
    # the da input already contains the ref db result
    da_tensor, da_mask = data.pad_batch(da_batch)
    if is_cuda:
        da_tensor = da_tensor.cuda(device)
        da_mask = da_mask.cuda(device)
    batch_da_text = model.batch_generate(da_tensor, da_mask, generate_mode='da', max_decode_len=max_span_len)
    for idx in range(batch_size):
        res_batch_parse_dict[idx]['aspn_gen'] = batch_da_text[idx]
        
    return res_batch_parse_dict
