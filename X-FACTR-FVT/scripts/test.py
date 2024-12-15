import sys
from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from typing import List, Dict, Tuple, Set, Union
import traceback
import torch
from transformers import *
import transformers
import json
import numpy as np
import os
from tqdm import tqdm
import argparse
import logging
from collections import defaultdict
import pandas
import csv
import time
import re
from prompt import Prompt
from check_gender import load_entity_gender, Gender
from check_instanceof import load_entity_instance, load_entity_is_cate
from entity_lang import Alias, MultiRel
from tokenization_kobert import KoBertTokenizer

from fvt.fvt import FastVocabularyTransfer
def get_tokenizer(lang: str, name: str):
    if lang == 'en':
        tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    elif lang == 'zh':
        tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    elif lang == 'es':
        tokenizer = AutoTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')
    elif lang == 'tr':
        tokenizer = AutoTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased')
    if name in {'xlm-mlm-100-1280', 'xlm-roberta-base'}:
        tokenizer = AutoTokenizer.from_pretrained(name)
    print(tokenizer.__class__.__name__, ', Vocab Size', tokenizer.vocab_size)
    return tokenizer
    
tokenizer = get_tokenizer('es', 'bert')
original_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')#AutoTokenizer.from_pretrained(LM)
pretrained_model = AutoModelWithLMHead.from_pretrained('bert-base-multilingual-cased')



fvt = FastVocabularyTransfer()
model = fvt.transfer(
    in_tokenizer=tokenizer,
    gen_tokenizer=original_tokenizer,
    gen_model=pretrained_model
) 